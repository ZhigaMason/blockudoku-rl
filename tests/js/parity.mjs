// node tests/js/parity.mjs fixture.json
// Checks web/game.js against positions + expected results produced by Python.
import { readFileSync } from 'node:fs';
import { legalMask, applyMove, encodeObservation } from '../../web/game.js';

const pieces = JSON.parse(readFileSync(new URL('../../web/pieces.json', import.meta.url))).pieces.map((p) => p.cells);
const cases = JSON.parse(readFileSync(process.argv[2]));
const eq = (a, b) => a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) < 1e-6);
let failures = 0;
cases.forEach((t, i) => {
  const board = Uint8Array.from(t.board);
  const fail = (what) => { failures++; console.error(`case ${i}: ${what} mismatch`); };
  if (!eq(Array.from(legalMask(board, t.hand, pieces)), t.mask)) fail('legal mask');
  const obs = encodeObservation(board, t.hand, pieces);
  if (!eq(Array.from(obs.planes), t.planes)) fail('planes');
  if (!eq(Array.from(obs.pieces), t.pieces)) fail('pieces');
  if (t.action >= 0) {
    const res = applyMove(board, t.hand, t.action, pieces);
    if (!eq(Array.from(res.board), t.next_board)) fail('next board');
    if (!eq(res.hand, t.next_hand)) fail('next hand');
    if (res.points !== t.points) fail(`points ${res.points} vs ${t.points}`);
  }
});
console.log(`${cases.length - failures}/${cases.length} ok`);
process.exit(failures ? 1 : 0);
