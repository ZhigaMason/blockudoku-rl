@AGENTS.md

## Claude Code specifics

- Project skills in `.claude/skills/`:
  - `train-and-ship`: train → evaluate against baselines → export → verify → publish the model
  - `change-game-rules`: every file to touch when a rule, piece or score changes
  - `change-network`: keeping `network.py` and `export_onnx.py` in lockstep
- Long training runs: start them with `run_in_background` and read `runs/<name>/metrics.jsonl`
  rather than polling the console.
- Do not start browsers or install Playwright; the maintainer tests the page by hand
  (`make serve`, then open http://127.0.0.1:8000/).
