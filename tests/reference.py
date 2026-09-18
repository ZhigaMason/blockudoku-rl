"""Deliberately naive, loop-based implementation of docs/RULES.md.

Used as an oracle for the JAX env and the JS port. Keep it simple, not fast.
"""

from blockudoku.pieces import PIECES

N = 9


def cells_of(piece_id):
    return [(r, c) for r in range(5) for c in range(5) if PIECES[piece_id][r][c]]


def is_legal(board, hand, slot, r, c):
    if hand[slot] < 0:
        return False
    for dr, dc in cells_of(hand[slot]):
        rr, cc = r + dr, c + dc
        if rr >= N or cc >= N or board[rr][cc]:
            return False
    return True


def legal_actions(board, hand):
    return [s * 81 + r * 9 + c for s in range(3) for r in range(N) for c in range(N)
            if is_legal(board, hand, s, r, c)]


def apply_move(board, hand, action):
    """Returns (board, hand_after_removal, points, k). No refill (it is random)."""
    slot, cell = divmod(action, 81)
    r, c = divmod(cell, 9)
    assert is_legal(board, hand, slot, r, c)
    board = [row[:] for row in board]
    cells = cells_of(hand[slot])
    for dr, dc in cells:
        board[r + dr][c + dc] = True
    regions = []
    for i in range(N):
        regions.append([(i, j) for j in range(N)])
        regions.append([(j, i) for j in range(N)])
    for br in range(3):
        for bc in range(3):
            regions.append([(3 * br + i, 3 * bc + j) for i in range(3) for j in range(3)])
    full = [reg for reg in regions if all(board[a][b] for a, b in reg)]
    for reg in full:
        for a, b in reg:
            board[a][b] = False
    k = len(full)
    hand = list(hand)
    hand[slot] = -1
    return board, hand, len(cells) + 18 * k * (k + 1) // 2, k
