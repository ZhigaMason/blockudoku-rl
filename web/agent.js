// Runs the exported Rainbow network with onnxruntime-web and picks moves.
// The `ort` module is injected so this file works in the browser (CDN import in
// app.js) and in Node tests (npm package), with identical inference code.

import { NUM_ACTIONS, SLOTS, BOARD, encodeObservation } from './game.js';

/** Create an agent from a model URL/path (or Uint8Array of the .onnx bytes). */
export async function loadAgent(ort, model, options = { executionProviders: ['wasm'] }) {
  const session = await ort.InferenceSession.create(model, options);

  /** Q-values for all 243 actions; illegal actions are set to -Infinity. */
  async function qValues(board, hand, pieces) {
    const obs = encodeObservation(board, hand, pieces);
    const out = await session.run({
      planes: new ort.Tensor('float32', obs.planes, [1, 1 + SLOTS, BOARD, BOARD]),
      pieces: new ort.Tensor('float32', obs.pieces, [1, obs.pieces.length]),
    });
    const q = Float32Array.from(out.q.data);
    for (let a = 0; a < NUM_ACTIONS; a++) if (!obs.mask[a]) q[a] = -Infinity;
    return q;
  }

  /** Greedy action (argmax over legal Q), or -1 if there is no legal move. */
  async function bestAction(board, hand, pieces) {
    const q = await qValues(board, hand, pieces);
    let best = -1;
    for (let a = 0; a < NUM_ACTIONS; a++) if (q[a] > -Infinity && (best < 0 || q[a] > q[best])) best = a;
    return { action: best, q };
  }

  return { session, qValues, bestAction };
}
