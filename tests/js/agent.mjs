// node tests/js/agent.mjs model.onnx fixture.json
// Runs the exported model through onnxruntime-web (same code path as the page)
// and compares Q-values and the greedy action with JAX.
import * as ort from 'onnxruntime-web';
import { readFileSync } from 'node:fs';
import { loadAgent } from '../../web/agent.js';

const [modelPath, fixture] = process.argv.slice(2);
const pieces = JSON.parse(readFileSync(new URL('../../web/pieces.json', import.meta.url))).pieces.map((p) => p.cells);
ort.env.wasm.numThreads = 1;
const agent = await loadAgent(ort, new Uint8Array(readFileSync(modelPath)));
let failures = 0;
for (const [i, t] of JSON.parse(readFileSync(fixture)).entries()) {
  const { action, q } = await agent.bestAction(Uint8Array.from(t.board), t.hand, pieces);
  const maxErr = Math.max(...t.q.map((v, a) => (t.mask[a] ? Math.abs(v - q[a]) : 0)));
  const illegalOk = t.mask.every((m, a) => m || q[a] === -Infinity);
  if (maxErr > 1e-3 || !illegalOk || action !== t.best) {
    failures++;
    console.error(`case ${i}: maxErr=${maxErr} illegalMasked=${illegalOk} action=${action} want=${t.best}`);
  }
}
console.log(failures ? `${failures} failures` : 'ok');
process.exit(failures ? 1 : 0);
