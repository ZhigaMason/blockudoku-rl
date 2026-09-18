import jax
import jax.numpy as jnp
import numpy as np

from blockudoku import env
from blockudoku.config import load_config
from blockudoku.network import (
    NUM_CELLS,
    RainbowNet,
    cast_floats,
    masked_argmax,
    q_values,
    rope_tables,
    symexp,
    symlog,
    value_support,
)

CFG = load_config("smoke")


def _obs():
    s = env.reset(jax.random.key(0))
    return env.observe(s.board, s.hand), env.action_mask(s.board, s.hand)


def test_shapes_and_noise():
    net = RainbowNet(CFG.net, key=jax.random.key(0))
    obs, _ = _obs()
    clean = net(obs)
    assert clean.shape == (243, CFG.net.num_atoms)
    assert jnp.allclose(clean, net(obs)), "noise-free path must be deterministic"
    n1, n2 = net(obs, key=jax.random.key(1)), net(obs, key=jax.random.key(2))
    assert not jnp.allclose(n1, n2), "noisy nets must depend on the key"


def test_batched_and_masked_argmax_is_legal():
    net = RainbowNet(CFG.net, key=jax.random.key(0))
    states = jax.vmap(env.reset)(jax.random.split(jax.random.key(3), 16))
    obs = jax.vmap(env.observe)(states.board, states.hand)
    support = jnp.asarray(value_support(CFG.v_min, CFG.v_max, CFG.net.num_atoms, CFG.support_scale))
    q = q_values(jax.vmap(net)(obs), support)
    mask = jax.vmap(env.action_mask)(states.board, states.hand)
    a = masked_argmax(q, mask)
    assert mask[jnp.arange(16), a].all()


def test_rope_is_relative_on_cells_and_off_for_other_tokens():
    T, hd = NUM_CELLS + 3 + 2, 16
    cos, sin, rot = rope_tables(T, hd, 100.0)
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=hd), rng.normal(size=hd)

    def rope(x, t):
        return x * cos[t] + (x @ rot) * sin[t]

    def cell(r, c):
        return r * 9 + c

    # same (dr, dc) offset -> same score, wherever it is on the board
    s1 = rope(q, cell(1, 2)) @ rope(k, cell(3, 7))
    s2 = rope(q, cell(4, 0)) @ rope(k, cell(6, 5))
    assert np.isclose(s1, s2, atol=1e-5)
    # row and column are separate axes: a different offset changes the score
    assert not np.isclose(s1, rope(q, cell(1, 2)) @ rope(k, cell(7, 3)))
    # rotation is norm-preserving; piece/register tokens are untouched
    assert np.isclose(np.linalg.norm(rope(q, cell(8, 8))), np.linalg.norm(q), atol=1e-5)
    assert np.allclose(rope(q, NUM_CELLS + 1), q)


def test_log_support():
    z = value_support(0.0, 5000.0, 101, "symlog")
    assert np.isclose(z[0], 0.0) and np.isclose(z[-1], 5000.0, rtol=1e-5)
    gaps = np.diff(z)
    assert (gaps > 0).all() and gaps[-1] > 100 * gaps[0], "atoms must be log-spaced"
    assert np.allclose(symexp(symlog(np.array([-3.0, 0.0, 7.5]))), [-3.0, 0.0, 7.5])
    assert np.allclose(value_support(0, 10, 11, "linear"), np.arange(11))


def test_bfloat16_compute_matches_float32():
    net = RainbowNet(load_config("small").net, key=jax.random.key(0))
    obs, _ = _obs()
    f32 = net(obs)
    bf16 = cast_floats(net, jnp.bfloat16)(obs)
    assert bf16.dtype == jnp.float32
    assert float(jnp.abs(bf16 - f32).max()) < 0.1
