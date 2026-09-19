import jax
import jax.numpy as jnp
import numpy as np

from blockudoku import replay


def _fill(T=16, E=2, rows=10, done_at=()):
    buf = replay.init(T, E)
    for t in range(rows):
        board = jnp.full((E, 9, 9), False).at[:, 0, 0].set(t % 2 == 1)
        hand = jnp.full((E, 3), t % 47, jnp.int32)
        done = jnp.array([t in done_at] * E)
        buf = replay.add(buf, board, hand, jnp.full((E,), t), jnp.full((E,), float(t)), done)
    return buf


def test_valid_rows_need_n_successors():
    buf = _fill(rows=10)
    valid = np.asarray(replay.valid_rows(buf, 3))
    assert valid[:7].all() and not valid[7:].any()


def test_wraparound_validity():
    buf = _fill(T=8, rows=11)  # ptr=3, rows 0..2 are newest
    valid = np.asarray(replay.valid_rows(buf, 2))
    # newest rows (1, 2) lack 2 successors; everything else is valid
    assert valid.tolist() == [True, False, False, True, True, True, True, True]


def test_nstep_returns_truncate_at_done():
    gamma = 0.5
    buf = _fill(rows=12, done_at=(4,))
    b = replay.sample(buf, jax.random.key(0), 256, 3, gamma, 0.5, jnp.float32(0.4))
    t = np.asarray(b.t)
    for i in range(len(t)):
        ti = int(t[i])
        steps = [ti, ti + 1, ti + 2]
        if 4 in steps:  # episode ends at step 4: stop summing after it, no bootstrap
            k = steps.index(4)
            want = sum(gamma**j * steps[j] for j in range(k + 1))
            assert float(b.discount[i]) == 0.0
        else:
            want = sum(gamma**j * s for j, s in enumerate(steps))
            assert np.isclose(float(b.discount[i]), gamma**3)
            assert int(b.next_hand[i, 0]) == (ti + 3) % 47
        assert np.isclose(float(b.returns[i]), want)
        assert int(b.action[i]) == ti and int(b.hand[i, 0]) == ti % 47


def test_sampling_follows_priorities():
    buf = _fill(rows=12)
    pr = jnp.zeros((16, 2)).at[3, 1].set(100.0).at[5, 0].set(1.0)
    buf = buf._replace(priority=pr)
    b = replay.sample(buf, jax.random.key(1), 1000, 3, 0.99, 1.0, jnp.float32(1.0))
    picks = list(zip(np.asarray(b.t).tolist(), np.asarray(b.e).tolist()))
    assert set(picks) <= {(3, 1), (5, 0)}
    frac = picks.count((3, 1)) / len(picks)
    assert 0.97 < frac < 1.0
    # the rarer item carries the larger importance weight
    w = dict(zip(picks, np.asarray(b.weight).tolist()))
    assert np.isclose(w[(5, 0)], 1.0) and w[(3, 1)] < 0.1


def test_priority_update_tracks_max():
    buf = _fill(rows=5)
    buf = replay.update_priorities(buf, jnp.array([0, 1]), jnp.array([0, 1]), jnp.array([0.5, 7.0]))
    assert float(buf.priority[1, 1]) == 7.0 and float(buf.max_priority) == 7.0
    buf = replay.add(buf, buf.board[0], buf.hand[0].astype(jnp.int32), buf.action[0], buf.reward[0], buf.done[0])
    assert (np.asarray(buf.priority[5]) == 7.0).all()


def test_importance_weights_stay_finite_for_zero_probability():
    # a zero-mass slot picked through float32 rounding used to give inf / inf = NaN
    prob = jnp.array([0.0, 1e-4, 4e-4, 1e-2])
    w = np.asarray(replay.importance_weights(prob, 1000, jnp.float32(0.5)))
    assert np.isfinite(w).all() and w[0] == 0.0
    expected = (1000 * np.asarray(prob[1:])) ** -0.5
    np.testing.assert_allclose(w[1:], expected / expected.max(), rtol=1e-6)
    # a batch of only zero-mass samples trains on nothing instead of NaN
    assert (np.asarray(replay.importance_weights(jnp.zeros(3), 1000, jnp.float32(1.0))) == 0).all()
