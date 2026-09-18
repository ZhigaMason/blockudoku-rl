---
name: change-game-rules
description: Use when changing Blockudoku rules, scoring, the piece catalogue, hand/refill behaviour, the action index or the observation layout. Lists every file that must change together and how to verify parity.
---

# Changing the game rules

The rules exist in four places that tests keep in agreement. Change them **in one commit**.

## Checklist

1. **Spec first**: edit `docs/RULES.md` (§3 hand, §4 moves, §5 scoring, §6 machine interface).
   If the user's description is ambiguous (for example, when refills happen), ask before coding.
2. **Oracle**: update `tests/reference.py`, the deliberately naive implementation.
3. **JAX env**: `src/blockudoku/env.py` (`step`, `legal_anchors`, `observe`, `move_points`, …).
   Keep it pure and vmap-able; no Python control flow on traced values.
4. **Browser port**: `web/game.js` (`legalMask`, `applyMove`, `encodeObservation`, `Game.step`)
   and, if the UI shows the rule, `web/app.js`.
5. **Pieces only**: edit `src/blockudoku/pieces.py`, then run `make sync` (regenerates
   `web/pieces.json` and the RULES.md appendix). Update the family table and the
   piece count in RULES.md §2 by hand.
6. **Tests**: adjust `tests/test_env.py` and the fixtures in `tests/test_web.py`; add a test for
   the new rule itself.
7. **Observation/action changes** also touch `network.py` input sizes, `export_onnx.py`
   input shapes, `web/agent.js` tensor shapes and RULES.md §6.

## Verify

```bash
make check          # lint, all tests, generated-asset freshness
```

`tests/test_web.py::test_game_js_matches_python` compares game.js with the Python oracle on
300 random positions.

## After a rule change

Every existing checkpoint and `web/model/model.onnx` was trained on the old rules.
Retrain (`train-and-ship` skill) and say clearly in your summary that the shipped model is
stale until you do.
