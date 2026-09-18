"""Piece catalogue. Authoritative list referenced by docs/RULES.md (Appendix A).

Each entry is (name, ascii-art). Pieces never rotate in play, so every orientation
is listed explicitly. Order defines the piece id. All pieces are equally likely.
"""

from __future__ import annotations

import numpy as np

PIECE_SIZE = 5  # every piece fits in a 5x5 bounding box

_CATALOGUE: list[tuple[str, str]] = [
    # --- 1 cell -------------------------------------------------------------
    ("mono", "X"),
    # --- 2 cells ------------------------------------------------------------
    ("I2-h", "XX"),
    ("I2-v", "X\nX"),
    ("D2-down", "X.\n.X"),
    ("D2-up", ".X\nX."),
    # --- 3 cells ------------------------------------------------------------
    ("I3-h", "XXX"),
    ("I3-v", "X\nX\nX"),
    ("C3-a", "XX\nX."),
    ("C3-b", "XX\n.X"),
    ("C3-c", ".X\nXX"),
    ("C3-d", "X.\nXX"),
    ("D3-down", "X..\n.X.\n..X"),
    ("D3-up", "..X\n.X.\nX.."),
    # --- 4 cells ------------------------------------------------------------
    ("I4-h", "XXXX"),
    ("I4-v", "X\nX\nX\nX"),
    ("O", "XX\nXX"),
    ("L-0", "X.\nX.\nXX"),
    ("L-90", "XXX\nX.."),
    ("L-180", "XX\n.X\n.X"),
    ("L-270", "..X\nXXX"),
    ("J-0", ".X\n.X\nXX"),
    ("J-90", "X..\nXXX"),
    ("J-180", "XX\nX.\nX."),
    ("J-270", "XXX\n..X"),
    ("T-0", "XXX\n.X."),
    ("T-90", ".X\nXX\n.X"),
    ("T-180", ".X.\nXXX"),
    ("T-270", "X.\nXX\nX."),
    ("S-h", ".XX\nXX."),
    ("S-v", "X.\nXX\n.X"),
    ("Z-h", "XX.\n.XX"),
    ("Z-v", ".X\nXX\nX."),
    # --- 5 cells ------------------------------------------------------------
    ("I5-h", "XXXXX"),
    ("I5-v", "X\nX\nX\nX\nX"),
    ("V-a", "XXX\nX..\nX.."),
    ("V-b", "XXX\n..X\n..X"),
    ("V-c", "..X\n..X\nXXX"),
    ("V-d", "X..\nX..\nXXX"),
    ("BT-0", "XXX\n.X.\n.X."),
    ("BT-90", "..X\nXXX\n..X"),
    ("BT-180", ".X.\n.X.\nXXX"),
    ("BT-270", "X..\nXXX\nX.."),
    ("U-0", "X.X\nXXX"),
    ("U-90", "XX\nX.\nXX"),
    ("U-180", "XXX\nX.X"),
    ("U-270", "XX\n.X\nXX"),
    ("plus", ".X.\nXXX\n.X."),
]


def _parse(art: str) -> np.ndarray:
    rows = art.split("\n")
    grid = np.zeros((PIECE_SIZE, PIECE_SIZE), dtype=bool)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch == "X":
                grid[r, c] = True
            elif ch != ".":
                raise ValueError(f"bad char {ch!r} in piece art")
    return grid


PIECE_NAMES: tuple[str, ...] = tuple(name for name, _ in _CATALOGUE)
PIECES: np.ndarray = np.stack([_parse(art) for _, art in _CATALOGUE])  # (P, 5, 5) bool
NUM_PIECES: int = len(PIECES)
PIECE_CELLS: np.ndarray = PIECES.sum(axis=(1, 2)).astype(np.int32)  # (P,)


def piece_art(piece_id: int) -> str:
    """Render a piece cropped to its bounding box, e.g. 'XX\\nX.'."""
    g = PIECES[piece_id]
    rows = np.where(g.any(1))[0]
    cols = np.where(g.any(0))[0]
    g = g[: rows.max() + 1, : cols.max() + 1]
    return "\n".join("".join("X" if v else "." for v in row) for row in g)


def to_json() -> dict:
    """Catalogue in the format consumed by web/game.js."""
    return {
        "size": PIECE_SIZE,
        "pieces": [
            {"id": i, "name": n, "cells": [[int(r), int(c)] for r, c in zip(*np.nonzero(PIECES[i]))]}
            for i, n in enumerate(PIECE_NAMES)
        ],
    }
