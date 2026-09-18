import jax
import jax.numpy as jnp
import numpy as np
import onnxruntime as ort
import pytest

from blockudoku import checkpoint, env
from blockudoku.config import load_config
from blockudoku.export_onnx import export
from blockudoku.network import RainbowNet, q_values, value_support


@pytest.mark.parametrize("preset", ["smoke", "small"])
def test_onnx_matches_jax(tmp_path, preset):
    cfg = load_config(preset)
    net = RainbowNet(cfg.net, key=jax.random.key(7))
    checkpoint.save(tmp_path / "run", net, cfg)
    out = export(tmp_path / "run", tmp_path / "web")

    states = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 8))
    boards = jax.random.bernoulli(jax.random.key(1), 0.4, (8, 9, 9))
    hands = states.hand.at[:, 1].set(jnp.array([-1, 3, 5, -1, 46, 0, 20, 33]))
    obs = jax.vmap(env.observe)(boards, hands)

    support = jnp.asarray(value_support(cfg.v_min, cfg.v_max, cfg.net.num_atoms, cfg.support_scale))
    want = np.asarray(q_values(jax.vmap(net)(obs), support))

    sess = ort.InferenceSession(str(out / "model.onnx"))
    (got,) = sess.run(["q"], {"planes": np.asarray(obs.planes), "pieces": np.asarray(obs.pieces)})
    assert got.shape == (8, 243)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
    assert (out / "model.yaml").exists()
