---
name: change-network
description: Use when modifying the Equinox RainbowNet (transformer blocks, tokens, RoPE, register tokens, heads, sizes) or NetConfig. Keeps the hand-written ONNX export in lockstep and preserves browser parity.
---

# Changing the network

`src/blockudoku/export_onnx.py` rebuilds `RainbowNet.__call__` op by op in ONNX. There
is no automatic tracing, so any architectural change must be mirrored there.

## Checklist

1. Edit `RainbowNet` in `src/blockudoku/network.py`. New hyperparameters go in
   `NetConfig` (`src/blockudoku/config.py`, schema only, no defaults), **plus a value in
   `configs/full.yaml`**. Override in `small.yaml`/`smoke.yaml` when the CPU presets need
   smaller values.
2. Mirror the change in `build_onnx`:
   - `Dense`/`NoisyLinear` → `g.linear` (MatMul with `W.T`, then Add). Noisy layers export `*_mu` only.
   - `LayerNorm` → `g.layer_norm` (ONNX `LayerNormalization`, `epsilon=LN_EPS`)
   - exact GELU → `g.gelu` (via `Erf`)
   - RoPE → `x*cos + (x @ rot)*sin`, with the constants from `rope_tables` (shared by both sides)
   - Constant tokens (registers) are broadcast to the batch with `Shape`/`Slice`/`Concat`/`Expand`.
   - Keep the layouts: tokens are `[cells 0..80 | pieces 81..83 | registers]`; advantage
     channels are `slot*atoms + atom`; the value head flattens registers token-major.
   - Stay within opset 17 ops that onnxruntime-web's wasm backend supports.
3. Inputs and outputs (`planes`, `pieces` → `q`) are a contract with `web/agent.js`. If they
   change, update agent.js and RULES.md §6.
4. Mind browser cost: every inference runs on a single wasm thread.
   Roughly `num_tokens × params` MACs per move; the full preset is ~340M MACs.

## Verify

```bash
uv run pytest -q tests/test_network.py tests/test_export.py tests/test_web.py tests/test_rainbow.py
```

- `test_export.py` checks JAX against onnxruntime on the smoke and small presets.
- `test_web.py` checks JAX against onnxruntime-web, through the real `agent.js`.
- `test_rainbow.py` covers the scan and python loops and bf16 training.

Old checkpoints will not load into a changed architecture, so retrain before exporting.
