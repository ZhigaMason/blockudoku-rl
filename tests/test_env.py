import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference import apply_move, legal_actions

from blockudoku import env
from blockudoku.pieces import NUM_PIECES, PIECE_CELLS, PIECES


def random_position(rng, fill):
    board = rng.random((9, 9)) < fill
    hand = rng.integers(0, NUM_PIECES, size=3)
    hand[rng.random(3) < 0.25] = -1
    return board, hand


def test_catalogue():
    assert NUM_PIECES == 47
    assert PIECES.shape == (47, 5, 5)
    assert PIECES.any(axis=2)[:, 0].all() and PIECES.any(axis=1)[:, 0].all(), "not anchor-normalised"
    assert len({p.tobytes() for p in PIECES}) == 47, "duplicate piece"
    assert set(PIECE_CELLS.tolist()) == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("fill", [0.0, 0.3, 0.6, 0.85])
def test_legal_mask_matches_reference(fill):
    rng = np.random.default_rng(int(fill * 100))
    mask_fn = jax.jit(env.action_mask)
    for _ in range(40):
        board, hand = random_position(rng, fill)
        got = np.nonzero(np.asarray(mask_fn(jnp.asarray(board), jnp.asarray(hand, jnp.int32))))[0]
        want = legal_actions(board.tolist(), hand.tolist())
        assert got.tolist() == want


def test_step_matches_reference():
    rng = np.random.default_rng(0)
    step = jax.jit(env.step)
    checked = 0
    while checked < 300:
        board, hand = random_position(rng, rng.uniform(0.2, 0.8))
        legal = legal_actions(board.tolist(), hand.tolist())
        if not legal or (hand < 0).sum() == 2:  # skip positions that trigger a (random) refill
            continue
        a = int(rng.choice(legal))
        state = env.EnvState(jnp.asarray(board), jnp.asarray(hand, jnp.int32), jnp.int32(0),
                             jnp.int32(0), jnp.bool_(False), jax.random.key(0))
        nxt, points, done = step(state, a)
        want_board, want_hand, want_points, _ = apply_move(board.tolist(), hand.tolist(), a)
        assert np.asarray(nxt.board).tolist() == want_board
        assert np.asarray(nxt.hand).tolist() == want_hand
        assert int(points) == want_points == int(nxt.score)
        assert bool(done) == (not legal_actions(want_board, want_hand))
        checked += 1


def test_clear_row_col_box_simultaneously():
    board = np.zeros((9, 9), bool)
    board[4, :] = True  # row 4 except (4,4)
    board[:, 4] = True  # col 4
    board[3:6, 3:6] = True  # centre box
    board[4, 4] = False
    board[0, 0] = True  # survivor
    state = env.EnvState(jnp.asarray(board), jnp.asarray([0, 0, -1], jnp.int32), jnp.int32(0),
                         jnp.int32(0), jnp.bool_(False), jax.random.key(0))
    nxt, points, done = env.step(state, env.encode_action(0, 4, 4))
    assert int(points) == 1 + 18 * 6  # k = 3
    expected = np.zeros((9, 9), bool)
    expected[0, 0] = True
    assert (np.asarray(nxt.board) == expected).all()
    assert not bool(done)


def test_hand_refilled_only_after_all_three_placed():
    step = jax.jit(env.step)
    state = env.EnvState(jnp.zeros((9, 9), bool), jnp.asarray([5, 5, 5], jnp.int32),
                         jnp.int32(0), jnp.int32(0), jnp.bool_(False), jax.random.key(1))
    state, _, _ = step(state, env.encode_action(0, 0, 0))
    assert np.asarray(state.hand).tolist() == [-1, 5, 5]
    state, _, _ = step(state, env.encode_action(2, 1, 0))
    assert np.asarray(state.hand).tolist() == [-1, 5, -1]
    state, _, done = step(state, env.encode_action(1, 2, 0))
    assert (np.asarray(state.hand) >= 0).all() and not bool(done)


def test_deal_is_uniform():
    n = 47 * 2000
    ids = np.asarray(env.deal(jax.random.key(0), (n,)))
    counts = np.bincount(ids, minlength=NUM_PIECES)
    assert counts.shape == (NUM_PIECES,)
    chi2 = ((counts - 2000) ** 2 / 2000).sum()
    assert chi2 < 90, f"chi2={chi2:.1f} (46 dof, p~1e-4 threshold)"


def test_game_over_and_illegal():
    # Checkerboard: empty cells are never horizontally adjacent, so I2-h cannot fit.
    board = (np.add.outer(np.arange(9), np.arange(9)) % 2 == 1)
    state = env.EnvState(jnp.asarray(board), jnp.asarray([0, 1, -1], jnp.int32), jnp.int32(0),
                         jnp.int32(0), jnp.bool_(False), jax.random.key(0))
    _, points, done = env.step(state, env.encode_action(0, 0, 0))  # mono -> only I2-h left
    assert int(points) == 1 and bool(done)
    # an illegal action terminates with 0 points and leaves the board untouched
    bad, p, d = env.step(state, env.encode_action(1, 0, 0))
    assert bool(d) and int(p) == 0 and (np.asarray(bad.board) == board).all()


def test_vmapped_random_rollouts():
    n = 64
    keys = jax.random.split(jax.random.key(0), n)
    states = jax.vmap(env.reset)(keys)

    @jax.jit
    def play(states, key):
        def body(carry, k):
            s, total_done = carry
            mask = jax.vmap(env.action_mask)(s.board, s.hand)
            logits = jnp.where(mask, 0.0, -jnp.inf)
            a = jax.random.categorical(k, logits)
            s2, pts, done, final = jax.vmap(env.step_autoreset)(s, a)
            return (s2, total_done + done.sum()), (pts, done, final.score)
        return jax.lax.scan(body, (states, 0), jax.random.split(key, 300))

    (final_states, n_done), (_, dones, scores) = play(states, jax.random.key(1))
    assert int(n_done) > 0
    assert (np.asarray(scores)[np.asarray(dones)] > 0).all()
    assert np.asarray(final_states.hand).min() >= -1


def test_observation_layout():
    board = jnp.zeros((9, 9), bool).at[2, 3].set(True)
    hand = jnp.asarray([5, -1, 33], jnp.int32)  # I3-h, empty, I5-v
    obs = env.observe(board, hand)
    assert obs.planes.shape == (4, 9, 9) and obs.pieces.shape == (75,)
    assert obs.planes[0, 2, 3] == 1 and obs.planes[0].sum() == 1
    assert obs.planes[2].sum() == 0  # empty slot
    assert obs.planes[3, 5:].sum() == 0  # I5-v cannot be anchored in the bottom 4 rows
    assert (obs.planes[1:].reshape(243) == env.action_mask(board, hand)).all()
    assert obs.pieces[:25].reshape(5, 5)[0, :3].sum() == 3  # I3-h
    assert obs.pieces[25:50].sum() == 0
    assert obs.pieces[50:].reshape(5, 5)[:, 0].sum() == 5  # I5-v
