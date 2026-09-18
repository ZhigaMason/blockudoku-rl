# blockudoku-rl

A **Rainbow DQN** agent for **Blockudoku**, trained in **JAX + Equinox** on a GPU,
exported to **ONNX**, and played in a **static web page** (GitHub Pages). In the page,
**onnxruntime-web** runs the network on the visitor's own machine.

```
configs/*.yaml ──► JAX env + Rainbow (GPU) ──► runs/<name>/best.eqx ──► export_onnx ──► web/model/model.onnx
                                                                                        │
                            browser: web/game.js (rules) + web/agent.js (onnxruntime-web) ◄┘
```

## The game

The full normative spec is in [`docs/RULES.md`](docs/RULES.md). In short:

- The board is 9×9, split into rows, columns and 3×3 boxes. The game starts empty.
- You hold 3 pieces, drawn **uniformly at random** from a catalogue of 47 pieces.
  Pieces are never rotated; every orientation is a separate piece.
- Place the pieces in any order. The hand is refilled only once all three have been placed.
- Any row, column or box that becomes full is cleared, all at the same time.
- Scoring per move: `cells placed + 18·k(k+1)/2`, where `k` is the number of regions
  cleared by that move.
- The game ends when none of the remaining pieces fits anywhere.

## Quick start

```bash
make install               # uv sync (Python 3.12, JAX CPU) + npm ci (dev-only JS test deps)
make test                  # ~40 tests: rules oracle, JAX env, replay, Rainbow, ONNX + browser parity
make baselines             # random ≈ 79 points, greedy 1-ply ≈ 168
make serve                 # play at http://127.0.0.1:8000/
```

On the GPU machine:

```bash
uv sync --extra cuda
cp .env.example .env        # add WANDB_API_KEY (the full preset logs to Weights & Biases)
uv run python -m blockudoku.train --config full --out runs/full
uv run python -m blockudoku.evaluate runs/full --episodes 512
uv run python -m blockudoku.export_onnx runs/full      # best.eqx -> web/model/
```

## Model

Tokens (`T = 81 + 3 + R`) → pre-LN transformer → noisy dueling C51 heads.

| part | details |
|------|---------|
| cell tokens (81) | `Dense([filled, legal-anchor slot 0..2]) + box embedding`. The box embedding is needed because 3×3 boxes are absolute positions, which RoPE cannot express |
| piece tokens (3) | `Dense(5×5 bitmap) + slot embedding` |
| register tokens (R) | learned **memory tokens** ([Darcet et al., 2024](https://arxiv.org/abs/2309.16588)); the value head reads them |
| attention | multi-head attention with **2D axial RoPE** on cell tokens: half of every head rotates with the row, half with the column. Piece and register tokens are not rotated |
| advantage head | per cell token, `NoisyLinear → ReLU → NoisyLinear(3 × atoms)`; action `(slot, r, c)` is read from cell `(r, c)` |
| value head | concatenated registers → `NoisyLinear → ReLU → NoisyLinear(atoms)` |
| **Q on a log scale** | C51 atoms are spaced **uniformly in symlog space** between `v_min` and `v_max`: dense near 0, sparse for large returns. The target distribution is projected by linear interpolation on this non-uniform grid, which preserves the mean exactly. Raw game points can therefore be the reward, with no reward scaling to tune |

Rainbow components:
- double Q-learning
- prioritised replay (proportional, with importance-sampling weights)
- dueling heads
- 3-step returns
- distributional (C51) values
- noisy nets (no ε-greedy)

`configs/full.yaml` is sized for a single GPU with 32 GB or more:
- 512 parallel envs, batch 1024, 2 updates per iteration (replay ratio 4).
- An 8.4M-transition replay (~0.9 GB), stored as compact states; observations are rebuilt when sampling.
- ViT with dim 192, depth 6, 6 heads and 4 registers: 3.8M parameters.
- bf16 compute with float32 master weights, and AdamW with linear warm-up.
- Every environment step and update of a log window is fused into one jitted `lax.scan`.

## Configuration

**All** hyperparameters live in YAML under `configs/`:
- `full.yaml` sets every field.
- `small.yaml` (CPU) and `smoke.yaml` (seconds, used by tests) `extends:` it.
- The Python dataclasses are only a schema. Unknown or missing keys are errors.

```bash
python -m blockudoku.train --config full --out runs/exp1 --set net.depth=8 --set learning_rate=1.5e-4
```

A run directory contains:
- `config.yaml`: the fully resolved config.
- `metrics.jsonl`: one line per log window.
- `model.eqx` (latest) and `best.eqx` (best greedy evaluation).

### Weights & Biases

The `wandb:` section of the config controls logging. It's enabled in `full.yaml` and
disabled in `small`/`smoke`.

- **Destination:** `mk-khavil-czech-technical-university-in-prague/blockudoku-rl`, set by
  `wandb.entity` and `wandb.project` in `full.yaml`.
- **Credentials:** read from `.env` in the working directory or a parent directory. Copy
  `.env.example` and set `WANDB_API_KEY`. `.env` is gitignored, and real environment
  variables take precedence over it.
- **What's logged:** every `metrics.jsonl` row, plotted against `env_steps`; the resolved
  config; the best evaluation score in the run summary. With `log_checkpoints: true`, each
  new `best.eqx` is uploaded with `config.yaml` as a model artifact (alias `best`).
- **Toggling:** `--set wandb.enabled=false` for a quick run without logging. With
  `--set wandb.mode=offline`, logs stay local and can be uploaded later with `wandb sync`.

## Web app

`web/` is deployed as-is by `.github/workflows/pages.yml`, with no build step:

- `game.js`: the rules as a DOM-free ES module, checked against Python on 300 random positions.
- `agent.js`: builds the observation and runs `model.onnx` with onnxruntime-web (wasm,
  single-threaded, because GitHub Pages cannot enable cross-origin isolation).
- `app.js`: the UI:
  - human play: click a piece, then the board; you see a placement preview and the regions it would clear
  - agent: **Hint**, **AI move**, **Autoplay**
  - **Q-value heatmap**: shows, for the selected or suggested piece, how good each placement is

To publish:
1. Commit `web/model/model.onnx` and `model.yaml`.
2. Enable GitHub Pages with source "GitHub Actions".
3. Push to `main`.

## Tests

`make check` runs lint, the tests, and a check that generated files are up to date. It's what CI runs.

- `tests/reference.py` is a deliberately naive implementation of RULES.md. The JAX env and
  `game.js` are both compared against it.
- `test_export.py`: the ONNX graph matches JAX under onnxruntime.
- `test_web.py`: the same model, loaded through `web/agent.js` and **onnxruntime-web** in
  Node, gives the same Q-values and the same greedy action as JAX.
- `test_rainbow.py`: the C51 projection matches classic C51 on a uniform grid and preserves
  mass and mean on the log grid; both loop modes and bf16 training run.

## Working with coding agents

- [`AGENTS.md`](AGENTS.md): the map, commands, cross-file contracts, conventions and gotchas.
  [`CLAUDE.md`](CLAUDE.md) imports it.
- `.claude/skills/`: step-by-step procedures for
  - `train-and-ship`
  - `change-game-rules`, which touches spec, env, JS, oracle and tests together
  - `change-network`, which keeps the Equinox model and the hand-built ONNX graph in sync
- `.claude/settings.json`: pre-approves the safe, read-only or test commands.

## Layout

```
docs/RULES.md        game spec (normative)       configs/          YAML presets
src/blockudoku/      env, network, replay, rainbow, config, train, evaluate, export_onnx
web/                 static site (+ generated pieces.json, model/)
tests/               pytest + tests/js (Node)    .claude/, AGENTS.md, CLAUDE.md
```
