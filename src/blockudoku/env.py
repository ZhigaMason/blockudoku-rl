"""Pure-functional, jit/vmap-friendly Blockudoku environment (spec: docs/RULES.md).

All functions operate on a single game; use `jax.vmap` for batches.

    state = reset(key)
    mask = action_mask(state.board, state.hand)       # (243,) bool
    state, points, done = step(state, action)
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from blockudoku.pieces import NUM_PIECES, PIECE_CELLS, PIECE_SIZE, PIECES

BOARD_SIZE = 9
NUM_SLOTS = 3
NUM_CELLS = BOARD_SIZE * BOARD_SIZE
NUM_ACTIONS = NUM_SLOTS * NUM_CELLS  # 243
NUM_PLANES = 1 + NUM_SLOTS  # board + legal-anchor mask per slot
PIECE_FEATURES = NUM_SLOTS * PIECE_SIZE * PIECE_SIZE  # 75
EMPTY_SLOT = -1
CLEAR_POINTS = 18

_PIECES = jnp.asarray(PIECES)  # (P, 5, 5) bool
PIECE_CELLS_J = jnp.asarray(PIECE_CELLS)  # (P,)


class EnvState(NamedTuple):
    board: jax.Array  # (9, 9) bool, True = filled
    hand: jax.Array  # (3,) int32, piece id or EMPTY_SLOT
    score: jax.Array  # () int32
    num_moves: jax.Array  # () int32
    done: jax.Array  # () bool
    key: jax.Array  # PRNG key used for future deals


class Obs(NamedTuple):
    planes: jax.Array  # (4, 9, 9) float32
    pieces: jax.Array  # (75,) float32


def decode_action(action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    slot, cell = jnp.divmod(action, NUM_CELLS)
    r, c = jnp.divmod(cell, BOARD_SIZE)
    return slot, r, c


def encode_action(slot, r, c):
    return slot * NUM_CELLS + r * BOARD_SIZE + c


def hand_bitmaps(hand: jax.Array) -> jax.Array:
    """(3,) piece ids -> (3, 5, 5) bool bitmaps; empty slots are all False."""
    bitmaps = _PIECES[jnp.maximum(hand, 0)]
    return bitmaps & (hand >= 0)[:, None, None]


def legal_anchors(board: jax.Array, hand: jax.Array) -> jax.Array:
    """(3, 9, 9) bool: True where the slot's piece can be anchored (top-left of bbox)."""
    # Out-of-board cells count as filled, so overhanging placements overlap.
    pad = PIECE_SIZE - 1
    occupied = jnp.pad(board, ((0, pad), (0, pad)), constant_values=True).astype(jnp.float32)
    kernels = hand_bitmaps(hand).astype(jnp.float32)
    overlap = jax.lax.conv_general_dilated(  # cross-correlation, no kernel flip
        occupied[None, None],
        kernels[:, None],
        window_strides=(1, 1),
        padding="VALID",
        precision=jax.lax.Precision.HIGHEST,
    )[0]
    return (overlap < 0.5) & (hand >= 0)[:, None, None]


def action_mask(board: jax.Array, hand: jax.Array) -> jax.Array:
    """(243,) bool legal-action mask, indexed as in docs/RULES.md §6.1."""
    return legal_anchors(board, hand).reshape(NUM_ACTIONS)


def observe(board: jax.Array, hand: jax.Array) -> Obs:
    planes = jnp.concatenate([board[None], legal_anchors(board, hand)]).astype(jnp.float32)
    pieces = hand_bitmaps(hand).reshape(PIECE_FEATURES).astype(jnp.float32)
    return Obs(planes, pieces)


def deal(key: jax.Array, shape=(NUM_SLOTS,)) -> jax.Array:
    """Uniformly random piece ids."""
    return jax.random.randint(key, shape, 0, NUM_PIECES, dtype=jnp.int32)


def reset(key: jax.Array) -> EnvState:
    key, sub = jax.random.split(key)
    return EnvState(
        board=jnp.zeros((BOARD_SIZE, BOARD_SIZE), dtype=bool),
        hand=deal(sub),
        score=jnp.zeros((), jnp.int32),
        num_moves=jnp.zeros((), jnp.int32),
        done=jnp.zeros((), bool),
        key=key,
    )


def place(board: jax.Array, piece_id: jax.Array, r: jax.Array, c: jax.Array) -> jax.Array:
    """Stamp a piece onto the board (no legality check; overhang is dropped)."""
    canvas = jnp.zeros((BOARD_SIZE + PIECE_SIZE - 1,) * 2, dtype=bool)
    canvas = jax.lax.dynamic_update_slice(canvas, _PIECES[piece_id], (r, c))
    return board | canvas[:BOARD_SIZE, :BOARD_SIZE]


def clear_regions(board: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Empty every full row/column/box simultaneously. Returns (board, regions_cleared)."""
    rows = board.all(axis=1)
    cols = board.all(axis=0)
    boxes = board.reshape(3, 3, 3, 3).all(axis=(1, 3))  # (box_row, box_col)
    box_cells = jnp.repeat(jnp.repeat(boxes, 3, axis=0), 3, axis=1)
    cleared = rows[:, None] | cols[None, :] | box_cells
    k = rows.sum() + cols.sum() + boxes.sum()
    return board & ~cleared, k.astype(jnp.int32)


def move_points(cells: jax.Array, k: jax.Array) -> jax.Array:
    return cells + CLEAR_POINTS * k * (k + 1) // 2


def step(state: EnvState, action: jax.Array) -> tuple[EnvState, jax.Array, jax.Array]:
    """Apply one move. Returns (next_state, points, done).

    An illegal action (or stepping a finished game) ends the game with 0 points.
    """
    action = jnp.asarray(action, jnp.int32)
    slot, r, c = decode_action(action)
    legal = action_mask(state.board, state.hand)[action] & ~state.done
    piece = jnp.maximum(state.hand[slot], 0)

    board, k = clear_regions(place(state.board, piece, r, c))
    points = move_points(PIECE_CELLS_J[piece], k)

    key, sub = jax.random.split(state.key)
    hand = state.hand.at[slot].set(EMPTY_SLOT)
    hand = jnp.where((hand == EMPTY_SLOT).all(), deal(sub), hand)  # refill only when all used
    game_over = ~legal_anchors(board, hand).any()

    points = jnp.where(legal, points, 0)
    next_state = EnvState(
        board=jnp.where(legal, board, state.board),
        hand=jnp.where(legal, hand, state.hand),
        score=state.score + points,
        num_moves=state.num_moves + legal.astype(jnp.int32),
        done=~legal | game_over,
        key=key,
    )
    return next_state, points, next_state.done


def step_autoreset(state: EnvState, action: jax.Array):
    """Like `step`, but a finished game is immediately replaced by a fresh one.

    Returns (next_state, points, done, final_state); `final_state` is the
    pre-reset state (useful for reading the final score when done).
    """
    final, points, done = step(state, action)
    fresh = reset(final.key)
    next_state = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, final)
    return next_state, points, done, final


def render(board, hand=None) -> str:
    """Human-readable board (for debugging / agents)."""
    import numpy as np

    from blockudoku.pieces import piece_art

    board = np.asarray(board)
    lines = []
    for r in range(BOARD_SIZE):
        if r and r % 3 == 0:
            lines.append("------+-------+------")
        cells = ["#" if board[r, c] else "." for c in range(BOARD_SIZE)]
        lines.append(" | ".join(" ".join(cells[i : i + 3]) for i in (0, 3, 6)))
    if hand is not None:
        for s, pid in enumerate(np.asarray(hand)):
            art = "(empty)" if pid < 0 else "\n        ".join(piece_art(int(pid)).split("\n"))
            lines.append(f"slot {s}: {art}")
    return "\n".join(lines)
