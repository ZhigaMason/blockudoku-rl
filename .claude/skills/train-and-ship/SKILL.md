---
name: train-and-ship
description: Use when training a Blockudoku agent, running a hyperparameter experiment, evaluating a checkpoint, or publishing a new model to the web page (web/model/).
---

# Train → evaluate → export → ship

## 1. Pick or write a config

Presets are in `configs/`:
- `smoke`: seconds, pipeline check.
- `small`: a CPU-sized transformer.
- `full`: one GPU with 32 GB or more; ViT 192×6, bf16, fused scan loop.

For an experiment, create `configs/<name>.yaml` with `extends: full.yaml` and only the
changed keys. Never put hyperparameters in Python. One-off tweaks can use `--set key=value`.

On the GPU machine, run `uv sync --extra cuda`. The full preset logs to Weights & Biases,
using credentials from `.env` (`WANDB_API_KEY`; copy `.env.example`). If the key is missing,
say so rather than inventing one. For a run without logging, use `--set wandb.enabled=false`. The first log line must show a CUDA device,
`loop=scan` and `compute=bfloat16`.

Knobs that matter most:
- Replay ratio (`batch_size * updates_per_iteration / num_envs`) trades sample efficiency
  for speed.
- `learning_rate` together with `lr_warmup_updates`.
- `v_max` must stay above the achievable discounted return.

## 2. Train (in the background)

```bash
uv run python -m blockudoku.train --config <name> --out runs/<name> [--time-limit-hours 12]
```

On MetaCentrum, submit `qsub -v CONFIG=<name> scripts/metacentrum/train_gpu.pbs` instead.
It's a 48 h GPU job that stops cleanly before the walltime. You can't run qsub from this
machine, so give the user the command.

Each run dir gets `config.yaml` (fully resolved), `metrics.jsonl`, `model.eqx` (latest),
and `best.eqx` (best greedy eval). With wandb enabled, the first lines print the run URL.
Read progress from `metrics.jsonl`, which holds the same data as the wandb charts:

- `loss` is the mean C51 cross-entropy of the window (≤ ln(num_atoms)).
- `train_score_mean` uses the noisy policy; `eval_score_mean` is greedy and noise-free.
- `sps` drops once updates start (`updates` > 0). The first window includes compile time.

Warning signs:
- A NaN loss: lower `learning_rate`, lengthen the warm-up, or try `compute_dtype: float32`.
- Eval stuck at the random baseline (≈ 79) long after `learning_starts`.
- `unfinished` games in eval: the agent is hitting `eval_max_moves`, so raise it.

## 3. Evaluate honestly

```bash
make baselines                              # random ≈ 79, greedy ≈ 168
uv run python -m blockudoku.evaluate runs/<name> --episodes 512
```

Report mean and median, plus `unfinished` (games that hit `--max-moves`). Compare to the
greedy baseline, not just random. Evaluate on the machine you trained on; the CPU is slow
for the full model.

## 4. Export and verify

```bash
uv run python -m blockudoku.export_onnx runs/<name>          # best.eqx -> web/model/
uv run pytest -q tests/test_export.py tests/test_web.py
```

## 5. Ship

`web/model/model.onnx` and `model.yaml` are committed; the Pages workflow deploys `web/` on
push to `main`. The full preset exports to about 15 MB (float32), which is fine for Pages. Ask the maintainer to try it (`make serve`) before committing. Do not open
browsers yourself.
