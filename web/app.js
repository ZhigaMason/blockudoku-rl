// UI: human play + agent (hint / single move / autoplay / Q heatmap).
import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import { Game, BOARD, CELLS, EMPTY, NUM_ACTIONS, encodeAction, decodeAction, canPlace, applyMove } from './game.js';
import { loadAgent } from './agent.js';

// GitHub Pages cannot send COOP/COEP headers, so SharedArrayBuffer threads are unavailable.
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

const $ = (id) => document.getElementById(id);
const catalogue = await (await fetch('pieces.json')).json();
const game = new Game(catalogue);

const ui = {
  selected: null, // slot index chosen by the human
  hover: null, // board cell index under the pointer
  hint: null, // { action } from the agent
  q: null, // last Q-values (Float32Array(243)) for the heatmap
  autoplay: false,
  busy: false,
  agent: null,
  best: Number(safeGet('blockudoku-best') ?? 0),
};

function safeGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function safeSet(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } }

// ---------- DOM construction ----------
const cells = [];
for (let i = 0; i < CELLS; i++) {
  const r = Math.floor(i / BOARD), c = i % BOARD;
  const el = document.createElement('div');
  el.className = 'cell';
  if ((Math.floor(r / 3) + Math.floor(c / 3)) % 2 === 1) el.classList.add('alt');
  if (c % 3 === 2) el.classList.add(c === BOARD - 1 ? 'edge-r' : 'box-r');
  if (r % 3 === 2) el.classList.add(r === BOARD - 1 ? 'edge-b' : 'box-b');
  el.setAttribute('role', 'gridcell');
  const heat = document.createElement('div');
  heat.className = 'heat';
  el.appendChild(heat);
  el.addEventListener('pointerenter', () => { ui.hover = i; render(); });
  el.addEventListener('click', () => onCellClick(i));
  $('board').appendChild(el);
  cells.push(el);
}
$('board').addEventListener('pointerleave', () => { ui.hover = null; render(); });

const slots = [];
for (let s = 0; s < 3; s++) {
  const el = document.createElement('button');
  el.className = 'slot';
  el.setAttribute('aria-label', `Piece ${s + 1}`);
  el.addEventListener('click', () => selectSlot(s));
  $('tray').appendChild(el);
  slots.push(el);
}

// ---------- helpers ----------
function pieceCells(slot) {
  return game.hand[slot] === EMPTY ? null : game.pieces[game.hand[slot]];
}

/** Offset from a piece's anchor (bbox top-left) to its grab point (bbox middle). */
function grabOffset(slot) {
  const pc = pieceCells(slot);
  const h = Math.max(...pc.map(([r]) => r)) + 1;
  const w = Math.max(...pc.map(([, c]) => c)) + 1;
  return { dr: Math.floor((h - 1) / 2), dc: Math.floor((w - 1) / 2) };
}

/** Anchor so the pointer sits on the middle of the piece's bounding box. */
function anchorFor(slot, cell) {
  const { dr, dc } = grabOffset(slot);
  return { r: Math.floor(cell / BOARD) - dr, c: (cell % BOARD) - dc };
}

function slotHasMove(slot) {
  const pc = pieceCells(slot);
  if (!pc) return false;
  for (let r = 0; r < BOARD; r++) for (let c = 0; c < BOARD; c++) if (canPlace(game.board, pc, r, c)) return true;
  return false;
}

function setStatus(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}

// ---------- rendering ----------
function render() {
  const preview = new Map(); // cell -> class
  if (ui.selected !== null && ui.hover !== null && pieceCells(ui.selected)) {
    const { r, c } = anchorFor(ui.selected, ui.hover);
    const pc = pieceCells(ui.selected);
    const inRange = r >= 0 && c >= 0;
    const ok = inRange && canPlace(game.board, pc, r, c);
    if (ok) {
      const res = applyMove(game.board, game.hand, encodeAction(ui.selected, r, c), game.pieces);
      for (const i of res.placed) preview.set(i, 'ghost');
      for (const i of res.cleared) preview.set(i, 'will-clear');
    } else {
      for (const [dr, dc] of pc) {
        const rr = r + dr, cc = c + dc;
        if (rr >= 0 && cc >= 0 && rr < BOARD && cc < BOARD) preview.set(rr * BOARD + cc, 'ghost-bad');
      }
    }
  }
  const hintCells = new Set();
  if (ui.hint && ui.hint.action >= 0) {
    const { slot, r, c } = decodeAction(ui.hint.action);
    for (const [dr, dc] of pieceCells(slot)) hintCells.add((r + dr) * BOARD + c + dc);
  }
  const heat = heatValues();
  cells.forEach((el, i) => {
    el.classList.toggle('filled', !!game.board[i]);
    for (const cls of ['ghost', 'ghost-bad', 'will-clear']) el.classList.toggle(cls, preview.get(i) === cls);
    el.classList.toggle('hint', hintCells.has(i));
    el.firstChild.style.background = heat && heat[i] > 0 ? `rgba(46, 170, 90, ${heat[i]})` : '';
  });

  slots.forEach((el, s) => {
    const pc = pieceCells(s);
    el.classList.toggle('empty', !pc);
    el.classList.toggle('stuck', !!pc && !slotHasMove(s));
    el.classList.toggle('selected', ui.selected === s);
    el.classList.toggle('hint', !!ui.hint && ui.hint.action >= 0 && decodeAction(ui.hint.action).slot === s);
    el.classList.toggle('heat-slot', heatSlot() === s);
    el.replaceChildren();
    if (!pc) return;
    const h = Math.max(...pc.map(([r]) => r)) + 1, w = Math.max(...pc.map(([, c]) => c)) + 1;
    const n = Math.max(h, w, 3);
    const mini = document.createElement('div');
    mini.className = 'mini';
    mini.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
    mini.style.width = `${(n / 5) * 100}%`;
    const on = new Set(pc.map(([r, c]) => r * n + c));
    const offR = Math.floor((n - h) / 2), offC = Math.floor((n - w) / 2);
    for (let k = 0; k < n * n; k++) {
      const cell = document.createElement('span');
      const r = Math.floor(k / n) - offR, c = (k % n) - offC;
      if (r >= 0 && c >= 0 && on.has(r * n + c)) cell.className = 'on';
      mini.appendChild(cell);
    }
    el.appendChild(mini);
  });

  $('score').textContent = game.score;
  $('moves').textContent = game.moves;
  $('best').textContent = ui.best;
  $('autoplay').setAttribute('aria-pressed', String(ui.autoplay));
  $('autoplay').textContent = ui.autoplay ? 'Stop' : 'Autoplay';
}

/** Slot shown by the heatmap: the selected one, else the hinted one, else the agent's best. */
function heatSlot() {
  if (!$('heatmap').checked || !ui.q) return null;
  if (ui.selected !== null) return ui.selected;
  let best = 0; // all-illegal Q is -Infinity, so a legal action always wins
  for (let a = 1; a < NUM_ACTIONS; a++) if (ui.q[a] > ui.q[best]) best = a;
  return decodeAction(ui.hint?.action ?? best).slot;
}

/**
 * Per-cell opacity from Q-values of the heat slot's moves. Each move is drawn on
 * the cell you would click to make it (the piece's grab point), so hovering the
 * brightest cell previews the agent's favourite placement of that piece.
 */
function heatValues() {
  const slot = heatSlot();
  if (slot === null) return null;
  const { dr, dc } = grabOffset(slot);
  const heat = new Array(CELLS).fill(0);
  const qs = ui.q.slice(slot * CELLS, (slot + 1) * CELLS);
  const finite = Array.from(qs).filter(Number.isFinite);
  if (!finite.length) return heat;
  const lo = Math.min(...finite), hi = Math.max(...finite);
  qs.forEach((v, anchor) => {
    if (!Number.isFinite(v)) return;
    const cell = (Math.floor(anchor / BOARD) + dr) * BOARD + (anchor % BOARD) + dc;
    heat[cell] = 0.1 + 0.65 * ((v - lo) / (hi - lo || 1)) ** 2;
  });
  return heat;
}

// ---------- actions ----------
function selectSlot(s) {
  if (ui.autoplay || !pieceCells(s)) return;
  ui.selected = ui.selected === s ? null : s;
  render();
}

function onCellClick(i) {
  if (ui.autoplay || ui.selected === null || !pieceCells(ui.selected)) return;
  const { r, c } = anchorFor(ui.selected, i);
  if (r < 0 || c < 0) return;
  const action = encodeAction(ui.selected, r, c);
  if (!game.isLegal(action)) return;
  play(action);
}

function play(action) {
  const res = game.step(action);
  ui.selected = null;
  ui.hint = null;
  ui.q = null;
  for (const i of res.cleared) {
    const el = cells[i];
    el.classList.remove('clearing');
    void el.offsetWidth; // restart animation
    el.classList.add('clearing');
    setTimeout(() => el.classList.remove('clearing'), 400);
  }
  if (res.k) setStatus(`Cleared ${res.k} region${res.k > 1 ? 's' : ''}: +${res.points}`);
  if (game.score > ui.best) { ui.best = game.score; safeSet('blockudoku-best', String(ui.best)); }
  render();
  if (game.done) gameOver();
  else refreshHeatmap();
}

function gameOver() {
  ui.autoplay = false;
  $('final-score').textContent = game.score;
  $('overlay').hidden = false;
  render();
}

function newGame() {
  game.reset();
  Object.assign(ui, { selected: null, hint: null, q: null });
  $('overlay').hidden = true;
  setStatus(ui.agent ? 'New game.' : 'New game — no model loaded, playing by hand.');
  render();
  refreshHeatmap();
}

/** Evaluate the current position; results for a position that has since changed are dropped. */
async function refreshQ() {
  if (!ui.agent || game.done) return null;
  const moves = game.moves, score = game.score;
  const res = await ui.agent.bestAction(game.board, game.hand, game.pieces);
  if (game.moves !== moves || game.score !== score) return null;
  ui.q = res.q;
  return res;
}

async function refreshHeatmap() {
  if (!$('heatmap').checked || !ui.agent) return;
  await refreshQ();
  render();
}

async function hint() {
  if (!ui.agent || game.done || ui.busy) return;
  ui.busy = true;
  try {
    ui.hint = await refreshQ();
    if (ui.hint) {
      const { slot } = decodeAction(ui.hint.action);
      setStatus(`Agent suggests piece ${slot + 1} (Q ≈ ${ui.hint.q[ui.hint.action].toFixed(2)}).`);
    }
    render();
  } finally { ui.busy = false; }
}

async function aiMove() {
  if (!ui.agent || game.done || ui.busy) return;
  ui.busy = true;
  try {
    const res = await refreshQ();
    if (res && res.action >= 0) play(res.action);
  } finally { ui.busy = false; }
}

/** Each autoplay move is first shown as a hint, then played after the speed delay. */
async function autoplayLoop() {
  while (ui.autoplay && !game.done) {
    if (!ui.hint && !ui.busy) {
      ui.busy = true;
      try { ui.hint = await refreshQ(); } finally { ui.busy = false; }
      render();
    }
    await new Promise((r) => setTimeout(r, 1000 - Number($('speed').value)));
    // play() and newGame() clear ui.hint, so a surviving hint is for the current position.
    if (ui.autoplay && ui.hint && ui.hint.action >= 0) play(ui.hint.action);
  }
  ui.autoplay = false;
  render();
}

function toggleAutoplay() {
  if (!ui.agent) return;
  if (game.done) newGame();
  ui.autoplay = !ui.autoplay;
  ui.selected = null;
  render();
  if (ui.autoplay) autoplayLoop();
}

// ---------- wiring ----------
$('new-game').addEventListener('click', () => { ui.autoplay = false; newGame(); });
$('again').addEventListener('click', newGame);
$('hint').addEventListener('click', hint);
$('ai-move').addEventListener('click', aiMove);
$('autoplay').addEventListener('click', toggleAutoplay);
$('heatmap').addEventListener('change', () => { render(); refreshHeatmap(); });
document.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (['1', '2', '3'].includes(e.key)) selectSlot(Number(e.key) - 1);
  else if (e.key === 'h' || e.key === 'H') hint();
  else if (e.key === 'a' || e.key === 'A') aiMove();
  else if (e.key === 'n' || e.key === 'N') { ui.autoplay = false; newGame(); }
  else if (e.key === ' ') { e.preventDefault(); toggleAutoplay(); }
  else if (e.key === 'Escape') { ui.selected = null; render(); }
});

render();
for (const el of document.querySelectorAll('.needs-model')) el.classList.add('disabled');
try {
  ui.agent = await loadAgent(ort, 'model/model.onnx');
  for (const el of document.querySelectorAll('.needs-model')) el.classList.remove('disabled');
  setStatus('Model loaded — runs entirely in your browser.');
  refreshHeatmap(); // the browser may have restored the checkbox as checked
} catch (err) {
  console.warn(err);
  for (const el of document.querySelectorAll('button.needs-model')) el.disabled = true;
  $('heatmap').disabled = true;
  setStatus('No model found (web/model/model.onnx) — play by hand. See README to train and export one.', true);
}
