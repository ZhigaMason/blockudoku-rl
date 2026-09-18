"""Browser-side parity: web/game.js and web/agent.js (via onnxruntime-web) vs Python.

Needs `node`; the agent test also needs `npm install` (skipped otherwise).
"""

import json
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference import apply_move, legal_actions

from blockudoku import checkpoint, env
from blockudoku.config import load_config
from blockudoku.export_onnx import export
from blockudoku.network import RainbowNet, q_values, value_support
from blockudoku.pieces import NUM_PIECES
from blockudoku.sync_assets import main as sync_main

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def _positions(n, seed):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        board = rng.random((9, 9)) < rng.uniform(0.0, 0.8)
        hand = rng.integers(0, NUM_PIECES, size=3)
        hand[rng.random(3) < 0.2] = -1
        yield board, hand


def _node(*args):
    r = subprocess.run([NODE, *map(str, args)], cwd=ROOT, capture_output=True, text=True,
                       timeout=120, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_assets_in_sync(monkeypatch):
    monkeypatch.setattr("sys.argv", ["sync_assets", "--check"])
    with pytest.raises(SystemExit) as e:
        sync_main()
    assert e.value.code == 0, "run: uv run python -m blockudoku.sync_assets"


def test_game_js_matches_python(tmp_path):
    rng = np.random.default_rng(1)
    cases = []
    for board, hand in _positions(300, 0):
        obs = env.observe(jnp.asarray(board), jnp.asarray(hand, jnp.int32))
        legal = legal_actions(board.tolist(), hand.tolist())
        case = {"board": board.reshape(-1).astype(int).tolist(), "hand": hand.tolist(),
                "mask": np.asarray(obs.planes[1:]).reshape(-1).astype(int).tolist(),
                "planes": np.asarray(obs.planes).reshape(-1).tolist(),
                "pieces": np.asarray(obs.pieces).tolist(), "action": -1}
        if legal:
            a = int(rng.choice(legal))
            nb, nh, pts, _ = apply_move(board.tolist(), hand.tolist(), a)
            case.update(action=a, next_board=np.asarray(nb).reshape(-1).astype(int).tolist(),
                        next_hand=nh, points=pts)
        cases.append(case)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    assert "300/300 ok" in _node("tests/js/parity.mjs", tmp_path / "cases.json")


@pytest.mark.skipif(not (ROOT / "node_modules" / "onnxruntime-web").exists(),
                    reason="run `npm install` for the onnxruntime-web test")
def test_agent_js_matches_jax(tmp_path):
    cfg = load_config("smoke")
    net = RainbowNet(cfg.net, key=jax.random.key(3))
    checkpoint.save(tmp_path / "run", net, cfg)
    export(tmp_path / "run", tmp_path / "model")
    support = jnp.asarray(value_support(cfg.v_min, cfg.v_max, cfg.net.num_atoms, cfg.support_scale))
    cases = []
    for board, hand in _positions(10, 5):
        if (hand < 0).all():
            continue
        obs = env.observe(jnp.asarray(board), jnp.asarray(hand, jnp.int32))
        mask = np.asarray(obs.planes[1:]).reshape(-1) > 0.5
        if not mask.any():
            continue
        q = np.asarray(q_values(net(obs), support))
        cases.append({"board": board.reshape(-1).astype(int).tolist(), "hand": hand.tolist(),
                      "mask": mask.astype(int).tolist(), "q": q.tolist(),
                      "best": int(np.argmax(np.where(mask, q, -np.inf)))})
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    assert "ok" in _node("tests/js/agent.mjs", tmp_path / "model" / "model.onnx", tmp_path / "cases.json")
