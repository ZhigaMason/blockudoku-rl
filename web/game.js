// Blockudoku game logic — a DOM-free port of src/blockudoku/env.py (spec: docs/RULES.md).
// Runs in the browser and in Node (tests/js/parity.mjs checks it against Python).

export const BOARD = 9;
export const SLOTS = 3;
export const CELLS = BOARD * BOARD; // 81
export const NUM_ACTIONS = SLOTS * CELLS; // 243
export const PIECE_SIZE = 5;
export const CLEAR_POINTS = 18;
export const EMPTY = -1;

export const encodeAction = (slot, r, c) => slot * CELLS + r * BOARD + c;
export function decodeAction(a) {
  const slot = Math.floor(a / CELLS);
  const cell = a % CELLS;
  return { slot, r: Math.floor(cell / BOARD), c: cell % BOARD };
}

/** pieces: array of [[dr, dc], ...] cell lists (from pieces.json). board: Uint8Array(81). */
export function canPlace(board, cells, r, c) {
  for (const [dr, dc] of cells) {
    const rr = r + dr, cc = c + dc;
    if (rr >= BOARD || cc >= BOARD || board[rr * BOARD + cc]) return false;
  }
  return true;
}

/** Uint8Array(243): 1 where action slot*81 + r*9 + c is legal. */
export function legalMask(board, hand, pieces) {
  const mask = new Uint8Array(NUM_ACTIONS);
  for (let s = 0; s < SLOTS; s++) {
    if (hand[s] === EMPTY) continue;
    const cells = pieces[hand[s]];
    for (let r = 0; r < BOARD; r++)
      for (let c = 0; c < BOARD; c++)
        if (canPlace(board, cells, r, c)) mask[encodeAction(s, r, c)] = 1;
  }
  return mask;
}

/** Full rows / columns / boxes of a board, as lists of cell indices. */
export function fullRegions(board) {
  const regions = [];
  const full = (idx) => idx.every((i) => board[i]);
  for (let i = 0; i < BOARD; i++) {
    const row = [], col = [];
    for (let j = 0; j < BOARD; j++) { row.push(i * BOARD + j); col.push(j * BOARD + i); }
    if (full(row)) regions.push(row);
    if (full(col)) regions.push(col);
  }
  for (let br = 0; br < 3; br++)
    for (let bc = 0; bc < 3; bc++) {
      const box = [];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) box.push((3 * br + i) * BOARD + 3 * bc + j);
      if (full(box)) regions.push(box);
    }
  return regions;
}

/**
 * Place + clear + score, emptying the used slot. Pure: returns new arrays.
 * The (random) refill of an empty hand is done by Game.step.
 * -> { board, hand, points, k, placed: [cell], cleared: [cell] }
 */
export function applyMove(board, hand, action, pieces) {
  const { slot, r, c } = decodeAction(action);
  const cells = pieces[hand[slot]];
  if (hand[slot] === EMPTY || !canPlace(board, cells, r, c)) throw new Error(`illegal action ${action}`);
  const next = Uint8Array.from(board);
  const placed = cells.map(([dr, dc]) => (r + dr) * BOARD + c + dc);
  for (const i of placed) next[i] = 1;
  const regions = fullRegions(next);
  const cleared = [...new Set(regions.flat())];
  for (const i of cleared) next[i] = 0;
  const k = regions.length;
  const newHand = hand.slice();
  newHand[slot] = EMPTY;
  return { board: next, hand: newHand, points: cells.length + (CLEAR_POINTS * k * (k + 1)) / 2, k, placed, cleared };
}

/** Network inputs (docs/RULES.md §6.2). */
export function encodeObservation(board, hand, pieces) {
  const planes = new Float32Array((1 + SLOTS) * CELLS);
  for (let i = 0; i < CELLS; i++) planes[i] = board[i];
  const mask = legalMask(board, hand, pieces);
  for (let a = 0; a < NUM_ACTIONS; a++) planes[CELLS + a] = mask[a];
  const feats = new Float32Array(SLOTS * PIECE_SIZE * PIECE_SIZE);
  for (let s = 0; s < SLOTS; s++) {
    if (hand[s] === EMPTY) continue;
    for (const [dr, dc] of pieces[hand[s]]) feats[s * 25 + dr * PIECE_SIZE + dc] = 1;
  }
  return { planes, pieces: feats, mask };
}

export class Game {
  /** catalogue: parsed pieces.json; rng: () => [0, 1). */
  constructor(catalogue, rng = Math.random) {
    this.pieces = catalogue.pieces.map((p) => p.cells);
    this.names = catalogue.pieces.map((p) => p.name);
    this.rng = rng;
    this.reset();
  }

  reset() {
    this.board = new Uint8Array(CELLS);
    this.hand = this.deal();
    this.score = 0;
    this.moves = 0;
    this.done = false;
  }

  /** Three independent, uniformly random piece ids (docs/RULES.md §3). */
  deal() {
    return Array.from({ length: SLOTS }, () => Math.floor(this.rng() * this.pieces.length));
  }

  legalMask() {
    return legalMask(this.board, this.hand, this.pieces);
  }

  isLegal(action) {
    const { slot, r, c } = decodeAction(action);
    return !this.done && this.hand[slot] !== EMPTY && canPlace(this.board, this.pieces[this.hand[slot]], r, c);
  }

  observation() {
    return encodeObservation(this.board, this.hand, this.pieces);
  }

  /** Apply a legal move (refilling the hand once all three are used); returns applyMove's result plus { refilled, done }. */
  step(action) {
    if (!this.isLegal(action)) throw new Error(`illegal action ${action}`);
    const res = applyMove(this.board, this.hand, action, this.pieces);
    this.board = res.board;
    this.hand = res.hand;
    const refilled = this.hand.every((h) => h === EMPTY);
    if (refilled) this.hand = this.deal();
    this.score += res.points;
    this.moves += 1;
    this.done = !this.legalMask().some((v) => v);
    return { ...res, refilled, done: this.done };
  }
}
