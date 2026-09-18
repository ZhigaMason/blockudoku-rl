"""Play full games with a policy and report scores.

    python -m blockudoku.evaluate runs/my-run            # trained agent (greedy, no noise)
    python -m blockudoku.evaluate --baseline random
    python -m blockudoku.evaluate --baseline greedy      # 1-ply max immediate points
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from blockudoku import env

Policy = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]  # (board, hand, key) -> action


def play(policy: Policy, key: jax.Array, num_episodes: int, max_moves: int) -> dict[str, np.ndarray]:
    """Run `num_episodes` games in parallel until all end (or `max_moves`)."""
    k_reset, k_play = jax.random.split(key)
    states = jax.vmap(env.reset)(jax.random.split(k_reset, num_episodes))

    @jax.jit
    def run(states, key):
        def cond(carry):
            s, _, i = carry
            return (~s.done).any() & (i < max_moves)

        def body(carry):
            s, key, i = carry
            key, k = jax.random.split(key)
            a = jax.vmap(policy)(s.board, s.hand, jax.random.split(k, num_episodes))
            nxt, _, _ = jax.vmap(env.step)(s, a)
            freeze = jax.vmap(lambda d, old, new: jax.tree.map(lambda x, y: jnp.where(d, x, y), old, new))
            s = freeze(s.done, s, nxt)  # finished games stay as they were
            return s, key, i + 1

        return jax.lax.while_loop(cond, body, (states, key, 0))[0]

    final = run(states, k_play)
    return {"score": np.asarray(final.score), "moves": np.asarray(final.num_moves),
            "finished": np.asarray(final.done)}


def random_policy(board, hand, key):
    mask = env.action_mask(board, hand)
    return jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf))


def greedy_points_policy(board, hand, key):
    """Maximise immediate points; ties broken by keeping the most empty cells, then randomly."""
    mask = env.action_mask(board, hand)

    def value(a):
        slot, r, c = env.decode_action(a)
        piece = jnp.maximum(hand[slot], 0)
        after, k = env.clear_regions(env.place(board, piece, r, c))
        return env.move_points(env.PIECE_CELLS_J[piece], k) * 100.0 - after.sum()

    v = jax.vmap(value)(jnp.arange(env.NUM_ACTIONS)) + jax.random.uniform(key, (env.NUM_ACTIONS,))
    return jnp.argmax(jnp.where(mask, v, -jnp.inf))


def agent_policy(run_dir: str) -> Policy:
    from blockudoku.checkpoint import load
    from blockudoku.network import value_support
    from blockudoku.rainbow import greedy_policy

    net, cfg = load(run_dir)
    pol = greedy_policy(net, jnp.asarray(value_support(cfg.v_min, cfg.v_max, cfg.net.num_atoms,
                                                       cfg.support_scale)))
    return lambda board, hand, key: pol(board, hand)


def summarize(result: dict[str, np.ndarray]) -> str:
    s = result["score"]
    return (f"episodes={len(s)} score mean={s.mean():.1f} median={np.median(s):.0f} "
            f"min={s.min()} max={s.max()} moves mean={result['moves'].mean():.1f} "
            f"unfinished={(~result['finished']).sum()}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?")
    ap.add_argument("--baseline", choices=["random", "greedy"])
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--max-moves", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    if bool(args.run_dir) == bool(args.baseline):
        ap.error("give exactly one of run_dir or --baseline")
    policy = {"random": random_policy, "greedy": greedy_points_policy}.get(args.baseline)
    if policy is None:
        policy = agent_policy(args.run_dir)
    print(summarize(play(policy, jax.random.key(args.seed), args.episodes, args.max_moves)))


if __name__ == "__main__":
    main()
