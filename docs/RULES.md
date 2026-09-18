# Blockudoku — Rules Specification

This document is the **normative definition** of the game as implemented in this
repository. The JAX environment (`src/blockudoku/env.py`), the piece catalogue
(`src/blockudoku/pieces.py`) and the browser port (`web/game.js`) must all agree
with it. If you change a rule, follow the checklist in
`.claude/skills/change-game-rules/SKILL.md`.

Blockudoku is a single-player, stochastic puzzle game that mixes Tetris-style
block placement with the 3×3 box structure of Sudoku.

## 1. Board

- The board is a **9 × 9 grid** of cells. Rows are numbered `0..8` from top to
  bottom and columns `0..8` from left to right. Cell `(r, c)` is row `r`,
  column `c`.
- Each cell is either **empty** or **filled**.
- The board is partitioned into 27 **regions**:
  - 9 **rows** (`r = 0..8`),
  - 9 **columns** (`c = 0..8`),
  - 9 **boxes**: the 3 × 3 sub-grids with top-left corners at
    `(3i, 3j)` for `i, j ∈ {0, 1, 2}`. Box `b = 3i + j`.
- A game starts with an empty board.

## 2. Pieces

- A **piece** is a fixed, non-empty set of cells (a polyomino or a diagonal
  shape) that fits inside a 5 × 5 bounding box.
- Pieces **cannot be rotated or mirrored** by the player. Each orientation of a
  shape is a separate piece in the catalogue.
- A piece's **anchor** is the top-left corner of its bounding box. A piece is
  stored normalised so that its topmost row and leftmost column both contain at
  least one cell.
- The catalogue contains exactly the **47 pieces** listed in
  [Appendix A](#appendix-a--piece-catalogue).

## 3. The hand

- The player holds a **hand of 3 slots** (`slot 0, 1, 2`).
- **Dealing:** at the start of the game, and whenever all three slots are
  empty, all three slots are filled. Each slot independently receives a piece
  drawn **uniformly at random** from the 47-piece catalogue (each piece has
  probability 1/47; draws are with replacement, so duplicates are possible).
- A slot becomes empty when its piece is placed. The hand is **not** refilled
  until all three pieces have been placed; the player may place them in any
  order.

## 4. Moves

A **move** is a pair `(slot, anchor)` where `slot ∈ {0,1,2}` is a non-empty slot
and `anchor = (r, c)` is a board cell. The move is **legal** iff, when the piece's
anchor is put on `(r, c)`, every cell of the piece

1. lies inside the board, and
2. covers an empty cell.

Resolving a legal move happens in this order:

1. **Place** — the covered cells become filled; the slot becomes empty.
2. **Clear** — find every row, column and box that is now completely filled.
   All cells belonging to *any* of those regions become empty *simultaneously*
   (a cell in both a full row and a full box is cleared once). Let `k` be the
   number of full regions found (`0 ≤ k`).
3. **Score** — see §5.
4. **Refill** — if all three slots are now empty, deal a new hand (§3).
5. **Game-over check** — the game ends if **no** piece remaining in the hand
   has a legal move on the resulting board.

The player cannot pass, and there is no undo.

## 5. Scoring

Per move:

```
points = cells_placed + 18 * k * (k + 1) / 2
```

- `cells_placed` — the number of cells in the placed piece (1–5).
- `k` — the number of regions cleared by this move. Clearing one region is
  worth 18, two simultaneously 54, three 108, … (triangular combo bonus).

The final score is the sum of points over all moves. There is no streak bonus
across moves; this keeps the game state Markov in `(board, hand)`.

## 6. Machine interface (for agents)

This section is part of the spec because the trained network, the ONNX export
and the browser all depend on it.

### 6.1 Action index

There are `3 × 81 = 243` actions:

```
action = slot * 81 + r * 9 + c          # slot ∈ 0..2, r, c ∈ 0..8
slot, cell = divmod(action, 81); r, c = divmod(cell, 9)
```

Actions with an empty slot or an illegal anchor are **masked**. The environment
treats an illegal action as terminating the episode with 0 reward (agents are
expected to mask, so this never happens during training).

### 6.2 Observation

Two float32 tensors (batch dimension first when batched):

| name     | shape       | contents |
|----------|-------------|----------|
| `planes` | `(4, 9, 9)` | channel 0: board, 1 = filled. Channels 1–3: legal-anchor mask for slot 0–2 (1 = the slot's piece can be anchored here). An empty slot has an all-zero mask. |
| `pieces` | `(75,)`     | the 3 slots' 5 × 5 piece bitmaps, anchor-aligned, flattened row-major and concatenated: index `slot*25 + r*5 + c`. Empty slot = zeros. |

The legal-action mask is exactly `planes[1:4].reshape(243)`.

### 6.3 Reward

The RL reward for a move is `points` (§5) multiplied by the training config's
`reward_scale`. An episode terminates on game over (§4 step 5).

## Appendix A — piece catalogue

`X` = cell, `.` = empty. Ids are assigned in this order (0-based), and every
piece is equally likely.

<!-- BEGIN GENERATED PIECES (python -m blockudoku.sync_assets) -->
| id | name | cells | shape |
|---:|------|------:|-------|
| 0 | mono | 1 | `X` |
| 1 | I2-h | 2 | `XX` |
| 2 | I2-v | 2 | `X`<br>`X` |
| 3 | D2-down | 2 | `X.`<br>`.X` |
| 4 | D2-up | 2 | `.X`<br>`X.` |
| 5 | I3-h | 3 | `XXX` |
| 6 | I3-v | 3 | `X`<br>`X`<br>`X` |
| 7 | C3-a | 3 | `XX`<br>`X.` |
| 8 | C3-b | 3 | `XX`<br>`.X` |
| 9 | C3-c | 3 | `.X`<br>`XX` |
| 10 | C3-d | 3 | `X.`<br>`XX` |
| 11 | D3-down | 3 | `X..`<br>`.X.`<br>`..X` |
| 12 | D3-up | 3 | `..X`<br>`.X.`<br>`X..` |
| 13 | I4-h | 4 | `XXXX` |
| 14 | I4-v | 4 | `X`<br>`X`<br>`X`<br>`X` |
| 15 | O | 4 | `XX`<br>`XX` |
| 16 | L-0 | 4 | `X.`<br>`X.`<br>`XX` |
| 17 | L-90 | 4 | `XXX`<br>`X..` |
| 18 | L-180 | 4 | `XX`<br>`.X`<br>`.X` |
| 19 | L-270 | 4 | `..X`<br>`XXX` |
| 20 | J-0 | 4 | `.X`<br>`.X`<br>`XX` |
| 21 | J-90 | 4 | `X..`<br>`XXX` |
| 22 | J-180 | 4 | `XX`<br>`X.`<br>`X.` |
| 23 | J-270 | 4 | `XXX`<br>`..X` |
| 24 | T-0 | 4 | `XXX`<br>`.X.` |
| 25 | T-90 | 4 | `.X`<br>`XX`<br>`.X` |
| 26 | T-180 | 4 | `.X.`<br>`XXX` |
| 27 | T-270 | 4 | `X.`<br>`XX`<br>`X.` |
| 28 | S-h | 4 | `.XX`<br>`XX.` |
| 29 | S-v | 4 | `X.`<br>`XX`<br>`.X` |
| 30 | Z-h | 4 | `XX.`<br>`.XX` |
| 31 | Z-v | 4 | `.X`<br>`XX`<br>`X.` |
| 32 | I5-h | 5 | `XXXXX` |
| 33 | I5-v | 5 | `X`<br>`X`<br>`X`<br>`X`<br>`X` |
| 34 | V-a | 5 | `XXX`<br>`X..`<br>`X..` |
| 35 | V-b | 5 | `XXX`<br>`..X`<br>`..X` |
| 36 | V-c | 5 | `..X`<br>`..X`<br>`XXX` |
| 37 | V-d | 5 | `X..`<br>`X..`<br>`XXX` |
| 38 | BT-0 | 5 | `XXX`<br>`.X.`<br>`.X.` |
| 39 | BT-90 | 5 | `..X`<br>`XXX`<br>`..X` |
| 40 | BT-180 | 5 | `.X.`<br>`.X.`<br>`XXX` |
| 41 | BT-270 | 5 | `X..`<br>`XXX`<br>`X..` |
| 42 | U-0 | 5 | `X.X`<br>`XXX` |
| 43 | U-90 | 5 | `XX`<br>`X.`<br>`XX` |
| 44 | U-180 | 5 | `XXX`<br>`X.X` |
| 45 | U-270 | 5 | `XX`<br>`.X`<br>`XX` |
| 46 | plus | 5 | `.X.`<br>`XXX`<br>`.X.` |
<!-- END GENERATED PIECES -->

Families:

| family | cells | orientations |
|--------|------:|-------------:|
| monomino | 1 | 1 |
| domino (I2) | 2 | 2 |
| diagonal-2 | 2 | 2 |
| tromino line (I3) | 3 | 2 |
| tromino corner (small L) | 3 | 4 |
| diagonal-3 | 3 | 2 |
| tetromino line (I4) | 4 | 2 |
| tetromino square (O) | 4 | 1 |
| tetromino L | 4 | 4 |
| tetromino J | 4 | 4 |
| tetromino T | 4 | 4 |
| tetromino S | 4 | 2 |
| tetromino Z | 4 | 2 |
| pentomino line (I5) | 5 | 2 |
| pentomino big corner (V) | 5 | 4 |
| pentomino big T | 5 | 4 |
| pentomino U | 5 | 4 |
| pentomino plus (X) | 5 | 1 |
| **total** | | **47** |
