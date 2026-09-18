"""Regenerate files derived from pieces.py.

    python -m blockudoku.sync_assets          # write web/pieces.json + RULES.md appendix
    python -m blockudoku.sync_assets --check  # exit 1 if anything is stale (used by tests/CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from blockudoku.pieces import NUM_PIECES, PIECE_CELLS, PIECE_NAMES, piece_art, to_json

ROOT = Path(__file__).resolve().parents[2]
PIECES_JSON = ROOT / "web" / "pieces.json"
RULES_MD = ROOT / "docs" / "RULES.md"
BEGIN = "<!-- BEGIN GENERATED PIECES (python -m blockudoku.sync_assets) -->"
END = "<!-- END GENERATED PIECES -->"


def pieces_json_text() -> str:
    return json.dumps(to_json(), indent=1) + "\n"


def appendix_text() -> str:
    lines = ["| id | name | cells | shape |", "|---:|------|------:|-------|"]
    for i in range(NUM_PIECES):
        shape = "<br>".join(f"`{row}`" for row in piece_art(i).split("\n"))
        lines.append(f"| {i} | {PIECE_NAMES[i]} | {PIECE_CELLS[i]} | {shape} |")
    return "\n".join(lines)


def rules_text() -> str:
    text = RULES_MD.read_text()
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{appendix_text()}\n{END}{tail}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    targets = {PIECES_JSON: pieces_json_text(), RULES_MD: rules_text()}
    stale = [p for p, t in targets.items() if not p.exists() or p.read_text() != t]
    if args.check:
        for p in stale:
            print(f"stale: {p.relative_to(ROOT)} (run: python -m blockudoku.sync_assets)")
        sys.exit(1 if stale else 0)
    for p in stale:
        p.write_text(targets[p])
        print(f"wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
