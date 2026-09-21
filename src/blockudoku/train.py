"""Train a Rainbow agent from a YAML config.

    python -m blockudoku.train --config small --out runs/small
    python -m blockudoku.train --config configs/full.yaml --out runs/full \
        --set num_envs=128 --set net.dim=256

`--config` takes a preset name from configs/ or a path. `--set` overrides any
field (dotted for nested keys). Writes to --out: config.yaml (the fully
resolved config), model.eqx (latest), best.eqx (best eval score) and
metrics.jsonl (one JSON object per log line). With `wandb.enabled` the same
metrics go to Weights & Biases (credentials from .env, see .env.example).

`--init-from` warm-starts a new run from a checkpoint (a run dir, meaning its
model.eqx, or an .eqx file inside one). Only the network is saved, so optimiser
state, replay buffer and counters start fresh: the buffer refills for
`learning_starts` transitions with the loaded policy, and the LR warms up again.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from blockudoku import checkpoint, evaluate
from blockudoku.config import Config, load_config
from blockudoku.network import num_params
from blockudoku.rainbow import Rainbow, TrainState, greedy_policy, lr_schedule, make_rainbow
from blockudoku.tracking import Tracker


def parse_args(argv=None) -> tuple[Config, Path, float | None, Path | None]:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="preset name in configs/ or path to a YAML file")
    ap.add_argument("--out", required=True, help="run directory")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--time-limit-hours", type=float, default=None,
                    help="stop cleanly (final eval + checkpoint) before this wall-clock budget runs out")
    ap.add_argument("--init-from", type=Path, default=None, metavar="RUN_DIR|FILE.eqx",
                    help="warm-start the network from a checkpoint (default file: model.eqx)")
    args = ap.parse_args(argv)
    return (load_config(args.config, args.overrides), Path(args.out), args.time_limit_hours,
            args.init_from)


def warm_start(rb: Rainbow, ts: TrainState, path: Path, log=print) -> TrainState:
    """Replace the online and target networks of a fresh `ts` with a saved checkpoint."""
    path = Path(path)
    run_dir, name = (path, "model.eqx") if path.is_dir() else (path.parent, path.name)
    net, src = checkpoint.load(run_dir, name)
    cfg = rb.cfg
    # the value heads are only meaningful on the support they were trained with
    same = ("net", "v_min", "v_max", "support_scale")
    diff = [k for k in same if getattr(src, k) != getattr(cfg, k)]
    if diff:
        raise ValueError(f"checkpoint {run_dir / name} differs from the config in {diff}")
    params, _ = eqx.partition(net, eqx.is_array)
    log(f"warm start from {run_dir / name}")
    return ts._replace(params=params, target_params=jax.tree.map(jnp.copy, params),
                       opt_state=rb.optimizer.init(params))


def train(cfg: Config, out: Path, log=print, time_limit_hours: float | None = None,
          init_from: Path | None = None) -> dict:
    """Train until cfg.total_env_steps or, if given, until the next log window would
    overrun `time_limit_hours`; either way the last window is evaluated and saved.
    `init_from` warm-starts the network (see `warm_start`)."""
    start = time.perf_counter()
    deadline = None if time_limit_hours is None else start + time_limit_hours * 3600
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.yaml")
    rb = make_rainbow(cfg)
    schedule = lr_schedule(cfg)
    key, eval_key = jax.random.split(jax.random.key(cfg.seed))
    ts = rb.init(key)
    if init_from is not None:
        ts = warm_start(rb, ts, init_from, log)
    log(f"devices={jax.devices()} params={num_params(rb.model(ts.params)):,} "
        f"replay={cfg.buffer_steps * cfg.num_envs:,} transitions "
        f"loop={'scan' if rb.uses_scan else 'python'} compute={cfg.compute_dtype}")

    steps_per_log = cfg.iterations_per_log * cfg.num_envs
    num_logs = max(cfg.total_env_steps // steps_per_log, 1)
    best = -np.inf
    last: dict = {}
    with open(out / "metrics.jsonl", "a") as metrics, Tracker(cfg, out, log) as tracker:
        for i in range(1, num_logs + 1):
            t0 = time.perf_counter()
            updates_before = int(ts.num_updates)
            ts = rb.run(ts)
            jax.block_until_ready(ts.iteration)
            dt = time.perf_counter() - t0
            s = jax.device_get(ts.stats)
            n = max(int(s.count), 1)
            window_updates = max(int(ts.num_updates) - updates_before, 1)
            row = {
                "env_steps": int(ts.iteration) * cfg.num_envs,
                "updates": int(ts.num_updates),
                "loss": float(ts.loss_sum) / window_updates,
                "grad_norm_mean": float(ts.grad_norm_sum) / window_updates,
                "grad_norm_max": float(ts.grad_norm_max),
                "lr": float(schedule(ts.num_updates)),
                "episodes": int(s.count),
                "train_score_mean": int(s.score_sum) / n,
                "train_score_max": int(s.score_max),
                "train_moves_mean": int(s.moves_sum) / n,
                "sps": steps_per_log / dt,
            }
            # stop if another window of the same length would not fit in the budget
            out_of_time = deadline is not None and time.perf_counter() + dt > deadline
            final = i == num_logs or out_of_time
            if i % cfg.eval_every_logs == 0 or final:
                net = rb.model(ts.params)
                pol = greedy_policy(net, rb.support)
                res = evaluate.play(lambda b, h, k, pol=pol: pol(b, h), eval_key,
                                    cfg.eval_episodes, cfg.eval_max_moves)
                row["eval_score_mean"] = float(res["score"].mean())
                row["eval_score_max"] = int(res["score"].max())
                checkpoint.save(out, net, cfg)
                if row["eval_score_mean"] > best:
                    best = row["eval_score_mean"]
                    checkpoint.save(out, net, cfg, name="best.eqx")
                    tracker.new_best(out, row)
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            tracker.log(row)
            log(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in row.items()))
            last = row
            if out_of_time and i < num_logs:
                log(f"time limit: stopping after {(time.perf_counter() - start) / 3600:.2f} h "
                    f"at env_steps={row['env_steps']}")
                break
    return last


def main() -> None:
    cfg, out, time_limit_hours, init_from = parse_args()
    train(cfg, out, time_limit_hours=time_limit_hours, init_from=init_from)


if __name__ == "__main__":
    main()
