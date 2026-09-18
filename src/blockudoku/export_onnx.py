"""Export a trained network to ONNX for onnxruntime-web.

    python -m blockudoku.export_onnx runs/small                  # -> web/model/
    python -m blockudoku.export_onnx runs/small --checkpoint model.eqx --out web/model

Output directory:
    model.onnx   fp32 graph: inputs planes [N,4,9,9], pieces [N,75] -> q [N,243]
    model.yaml   provenance: run, checkpoint, eval score, network config, I/O layout

The graph is written by hand from the Equinox parameters (noise-free means of the
noisy layers) and mirrors `RainbowNet.__call__` op for op. tests/test_export.py
checks numerical parity against JAX with onnxruntime; tests/test_web.py checks the
browser code path with onnxruntime-web.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import yaml
from onnx import TensorProto, helper, numpy_helper

from blockudoku import checkpoint
from blockudoku.config import Config
from blockudoku.env import NUM_ACTIONS, NUM_CELLS, NUM_PLANES, NUM_SLOTS, PIECE_FEATURES
from blockudoku.network import (
    BOX_OF_CELL,
    LN_EPS,
    PIECE_PIXELS,
    RainbowNet,
    rope_tables,
    value_support,
)

OPSET = 17  # LayerNormalization needs >= 17
IR_VERSION = 8  # conservative, loads in older onnxruntime-web builds too
ROOT = Path(__file__).resolve().parents[2]


def i64(values) -> np.ndarray:
    return np.asarray(values, np.int64)


def f32(values) -> np.ndarray:
    return np.asarray(values, np.float32)


class _Graph:
    def __init__(self):
        self.nodes: list[onnx.NodeProto] = []
        self.inits: list[onnx.TensorProto] = []
        self._n = 0

    def const(self, value, name: str | None = None) -> str:
        self._n += 1
        name = name or f"c{self._n}"
        self.inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def op(self, op_type: str, inputs: list[str], out: str | None = None, **attrs) -> str:
        self._n += 1
        out = out or f"{op_type.lower()}_{self._n}"
        self.nodes.append(helper.make_node(op_type, inputs, [out], name=f"n{self._n}_{op_type}", **attrs))
        return out

    def linear(self, x: str, w, b, name: str) -> str:
        """y = x @ w.T + b for any leading dims (w is (out, in) as in Equinox)."""
        y = self.op("MatMul", [x, self.const(f32(w).T, f"{name}.w")])
        return self.op("Add", [y, self.const(f32(b), f"{name}.b")])

    def reshape(self, x: str, shape) -> str:
        return self.op("Reshape", [x, self.const(i64(shape))])

    def layer_norm(self, x: str, ln, name: str) -> str:
        return self.op("LayerNormalization", [x, self.const(f32(ln.weight), f"{name}.w"),
                                              self.const(f32(ln.bias), f"{name}.b")],
                       axis=-1, epsilon=LN_EPS)

    def gelu(self, x: str) -> str:  # exact: 0.5 x (1 + erf(x / sqrt 2))
        erf = self.op("Erf", [self.op("Mul", [x, self.const(f32(1 / np.sqrt(2)))])])
        return self.op("Mul", [self.op("Mul", [x, self.const(f32(0.5))]),
                               self.op("Add", [erf, self.const(f32(1.0))])])

    def slice_tokens(self, x: str, start: int, end: int) -> str:
        return self.op("Slice", [x, self.const(i64([start])), self.const(i64([end])), self.const(i64([1]))])


def build_onnx(net: RainbowNet, cfg: Config) -> onnx.ModelProto:
    g = _Graph()
    n_atoms, dim, heads = cfg.net.num_atoms, cfg.net.dim, cfg.net.heads
    hd, regs = dim // heads, cfg.net.num_registers
    T = NUM_CELLS + NUM_SLOTS + regs

    # tokens: cells [N,81,D], pieces [N,3,D], registers broadcast to [N,R,D]
    c = g.op("Transpose", [g.reshape("planes", [0, NUM_PLANES, NUM_CELLS])], perm=[0, 2, 1])
    c = g.op("Add", [g.linear(c, net.cell_embed.weight, net.cell_embed.bias, "cell_embed"),
                     g.const(f32(net.box_embed)[BOX_OF_CELL], "box_embed")])
    p = g.reshape("pieces", [0, NUM_SLOTS, PIECE_PIXELS])
    p = g.op("Add", [g.linear(p, net.piece_embed.weight, net.piece_embed.bias, "piece_embed"),
                     g.const(f32(net.slot_embed), "slot_embed")])
    batch = g.op("Slice", [g.op("Shape", ["pieces"]), g.const(i64([0])), g.const(i64([1]))])
    reg_shape = g.op("Concat", [batch, g.const(i64([regs, dim]))], axis=0)
    r = g.op("Expand", [g.const(f32(net.registers)[None], "registers"), reg_shape])
    x = g.op("Concat", [c, p, r], axis=1)

    cos, sin, rot = rope_tables(T, hd, cfg.net.rope_base)
    cos, sin, rot = g.const(cos, "rope_cos"), g.const(sin, "rope_sin"), g.const(rot, "rope_rot")
    scale = g.const(f32(1 / np.sqrt(hd)))

    def heads_split(y):  # [N,T,D] -> [N,H,T,hd]
        return g.op("Transpose", [g.reshape(y, [0, T, heads, hd])], perm=[0, 2, 1, 3])

    def rope(y):
        return g.op("Add", [g.op("Mul", [y, cos]), g.op("Mul", [g.op("MatMul", [y, rot]), sin])])

    for i, blk in enumerate(net.blocks):
        at = blk.attn
        h = g.layer_norm(x, blk.ln1, f"b{i}.ln1")
        q = rope(heads_split(g.linear(h, at.q.weight, at.q.bias, f"b{i}.q")))
        k = rope(heads_split(g.linear(h, at.k.weight, at.k.bias, f"b{i}.k")))
        v = heads_split(g.linear(h, at.v.weight, at.v.bias, f"b{i}.v"))
        s = g.op("Mul", [g.op("MatMul", [q, g.op("Transpose", [k], perm=[0, 1, 3, 2])]), scale])
        o = g.op("MatMul", [g.op("Softmax", [s], axis=-1), v])
        o = g.reshape(g.op("Transpose", [o], perm=[0, 2, 1, 3]), [0, T, dim])
        x = g.op("Add", [x, g.linear(o, at.o.weight, at.o.bias, f"b{i}.o")])
        h = g.layer_norm(x, blk.ln2, f"b{i}.ln2")
        h = g.gelu(g.linear(h, blk.fc1.weight, blk.fc1.bias, f"b{i}.fc1"))
        x = g.op("Add", [x, g.linear(h, blk.fc2.weight, blk.fc2.bias, f"b{i}.fc2")])
    x = g.layer_norm(x, net.ln_f, "ln_f")

    # advantage from cell tokens -> [N, 243, A] (action = slot*81 + cell)
    a = g.slice_tokens(x, 0, NUM_CELLS)
    a = g.op("Relu", [g.linear(a, net.adv_hidden.weight_mu, net.adv_hidden.bias_mu, "adv_hidden")])
    a = g.linear(a, net.adv_out.weight_mu, net.adv_out.bias_mu, "adv_out")  # [N,81,3A]
    a = g.op("Transpose", [g.reshape(a, [0, NUM_CELLS, NUM_SLOTS, n_atoms])], perm=[0, 2, 1, 3])
    a = g.reshape(a, [0, NUM_ACTIONS, n_atoms])

    # value from register tokens -> [N, 1, A]
    v = g.reshape(g.slice_tokens(x, NUM_CELLS + NUM_SLOTS, T), [0, regs * dim])
    v = g.op("Relu", [g.linear(v, net.value_hidden.weight_mu, net.value_hidden.bias_mu, "value_hidden")])
    v = g.linear(v, net.value_out.weight_mu, net.value_out.bias_mu, "value_out")
    v = g.op("Unsqueeze", [v, g.const(i64([1]))])

    # dueling combine, softmax over atoms, expectation over the (log-spaced) support
    mean_a = g.op("ReduceMean", [a], axes=[1], keepdims=1)
    logits = g.op("Add", [v, g.op("Sub", [a, mean_a])], out="logits")
    probs = g.op("Softmax", [logits], axis=-1)
    support = value_support(cfg.v_min, cfg.v_max, n_atoms, cfg.support_scale)[:, None]
    q = g.op("MatMul", [probs, g.const(support, "support")])
    g.op("Reshape", [q, g.const(i64([0, NUM_ACTIONS]))], out="q")

    graph = helper.make_graph(
        g.nodes, "blockudoku_rainbow_vit",
        inputs=[
            helper.make_tensor_value_info("planes", TensorProto.FLOAT, ["batch", NUM_PLANES, 9, 9]),
            helper.make_tensor_value_info("pieces", TensorProto.FLOAT, ["batch", PIECE_FEATURES]),
        ],
        outputs=[helper.make_tensor_value_info("q", TensorProto.FLOAT, ["batch", NUM_ACTIONS])],
        initializer=g.inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)],
                              producer_name="blockudoku-rl")
    model.ir_version = IR_VERSION
    onnx.checker.check_model(model, full_check=True)
    return model


def export(run_dir: str | Path, out_dir: str | Path, ckpt_name: str = "model.eqx",
           metadata: dict | None = None) -> Path:
    net, cfg = checkpoint.load(run_dir, ckpt_name)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save(build_onnx(net, cfg), out_dir / "model.onnx")
    info = {
        **(metadata or {}),
        "inputs": {"planes": [NUM_PLANES, 9, 9], "pieces": [PIECE_FEATURES]},
        "outputs": {"q": [NUM_ACTIONS]},
        "action_index": "slot*81 + row*9 + col (docs/RULES.md section 6)",
        "value_support": {"v_min": cfg.v_min, "v_max": cfg.v_max, "scale": cfg.support_scale,
                          "reward_scale": cfg.reward_scale},
        "net": cfg.to_dict()["net"],
    }
    (out_dir / "model.yaml").write_text(yaml.safe_dump(info, sort_keys=False))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--checkpoint", default="best.eqx")
    ap.add_argument("--out", default=str(ROOT / "web" / "model"))
    args = ap.parse_args()
    meta = {"run": Path(args.run_dir).name, "checkpoint": args.checkpoint}
    metrics = Path(args.run_dir) / "metrics.jsonl"
    if metrics.exists():
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        evals = [r for r in rows if "eval_score_mean" in r]
        if evals:
            best = max(evals, key=lambda r: r["eval_score_mean"])
            meta["eval_score_mean"] = best["eval_score_mean"]
            meta["env_steps"] = best["env_steps"]
    out = export(args.run_dir, args.out, args.checkpoint, meta)
    print(f"wrote {out / 'model.onnx'} and {out / 'model.yaml'}")


if __name__ == "__main__":
    main()
