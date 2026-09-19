"""Prioritised n-step replay buffer as a pure JAX pytree.

Storage is a time-major ring of shape (T, E): row t holds the transition taken by
each of the E parallel envs at iteration t. Only compact game states are stored
(board + hand); observations and masks are recomputed at sample time. Because
envs auto-reset, row t+1 holds the successor of row t unless `done[t]`.

n-step targets are assembled at sample time from rows t .. t+n, so a row is
sampleable only once n newer rows exist and it has not been overwritten.

Sampling is proportional to priority**alpha via a two-level inverse CDF
(pick a row by row-sum, then an env within the row), which is O(T + B*E) and
keeps float32 cumsums short enough to stay accurate.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from blockudoku.env import BOARD_SIZE, NUM_SLOTS


class Replay(NamedTuple):
    board: jax.Array  # (T, E, 9, 9) bool
    hand: jax.Array  # (T, E, 3) int8
    action: jax.Array  # (T, E) int32
    reward: jax.Array  # (T, E) float32
    done: jax.Array  # (T, E) bool
    priority: jax.Array  # (T, E) float32, raw (alpha applied at sample time)
    ptr: jax.Array  # () int32, next row to write
    size: jax.Array  # () int32, rows written so far (<= T)
    max_priority: jax.Array  # () float32


class Batch(NamedTuple):
    board: jax.Array  # (B, 9, 9)
    hand: jax.Array  # (B, 3)
    action: jax.Array  # (B,)
    returns: jax.Array  # (B,) discounted n-step reward sum (truncated at episode end)
    discount: jax.Array  # (B,) gamma**n, or 0 if the episode ended within n steps
    next_board: jax.Array  # (B, 9, 9)
    next_hand: jax.Array  # (B, 3)
    weight: jax.Array  # (B,) importance-sampling weight, max-normalised
    t: jax.Array  # (B,) row index, for priority updates
    e: jax.Array  # (B,) env index


def init(capacity_steps: int, num_envs: int) -> Replay:
    T, E = capacity_steps, num_envs
    return Replay(
        board=jnp.zeros((T, E, BOARD_SIZE, BOARD_SIZE), bool),
        hand=jnp.zeros((T, E, NUM_SLOTS), jnp.int8),
        action=jnp.zeros((T, E), jnp.int32),
        reward=jnp.zeros((T, E), jnp.float32),
        done=jnp.zeros((T, E), bool),
        priority=jnp.zeros((T, E), jnp.float32),
        ptr=jnp.zeros((), jnp.int32),
        size=jnp.zeros((), jnp.int32),
        max_priority=jnp.ones((), jnp.float32),
    )


def add(buf: Replay, board, hand, action, reward, done) -> Replay:
    """Write one row (one transition per env). New rows get the max priority."""
    T = buf.board.shape[0]
    i = buf.ptr
    return buf._replace(
        board=buf.board.at[i].set(board),
        hand=buf.hand.at[i].set(hand.astype(jnp.int8)),
        action=buf.action.at[i].set(action),
        reward=buf.reward.at[i].set(reward),
        done=buf.done.at[i].set(done),
        priority=buf.priority.at[i].set(buf.max_priority),
        ptr=(i + 1) % T,
        size=jnp.minimum(buf.size + 1, T),
    )


def valid_rows(buf: Replay, n_step: int) -> jax.Array:
    """(T,) bool: rows whose n successors are already written."""
    T = buf.board.shape[0]
    t = jnp.arange(T)
    newer = (buf.ptr - 1 - t) % T  # rows written after t
    return (t < buf.size) & (newer >= n_step)


def num_transitions(buf: Replay, n_step: int) -> jax.Array:
    return valid_rows(buf, n_step).sum() * buf.board.shape[1]


def sample(buf: Replay, key, batch_size: int, n_step: int, gamma: float, alpha: float,
           beta: jax.Array) -> Batch:
    T, E = buf.priority.shape
    k_row, k_col = jax.random.split(key)
    p = jnp.where(valid_rows(buf, n_step)[:, None], buf.priority**alpha, 0.0)  # (T, E)
    row_mass = p.sum(axis=1)
    row_cdf = jnp.cumsum(row_mass)
    total = row_cdf[-1]

    # stratified sampling over rows, then proportional within the chosen row
    u = (jnp.arange(batch_size) + jax.random.uniform(k_row, (batch_size,))) / batch_size
    t = jnp.searchsorted(row_cdf, jnp.minimum(u, 1.0 - 1e-6) * total, side="right")
    t = jnp.minimum(t, T - 1)
    col_cdf = jnp.cumsum(p[t], axis=1)  # (B, E)
    v = jax.random.uniform(k_col, (batch_size,)) * col_cdf[:, -1] * (1.0 - 1e-6)
    e = jnp.minimum(jax.vmap(lambda c, x: jnp.searchsorted(c, x, side="right"))(col_cdf, v), E - 1)

    weight = importance_weights(p[t, e] / total, num_transitions(buf, n_step), beta)

    steps = (t[:, None] + jnp.arange(n_step)) % T  # (B, n)
    rewards = buf.reward[steps, e[:, None]]
    dones = buf.done[steps, e[:, None]].astype(jnp.float32)
    alive = jnp.cumprod(1.0 - dones, axis=1)  # alive after step k
    alive_before = jnp.concatenate([jnp.ones((batch_size, 1)), alive[:, :-1]], axis=1)
    returns = (rewards * alive_before * gamma ** jnp.arange(n_step)).sum(axis=1)
    discount = alive[:, -1] * gamma**n_step
    tn = (t + n_step) % T

    return Batch(
        board=buf.board[t, e], hand=buf.hand[t, e].astype(jnp.int32), action=buf.action[t, e],
        returns=returns, discount=discount,
        next_board=buf.board[tn, e], next_hand=buf.hand[tn, e].astype(jnp.int32),
        weight=weight, t=t, e=e,
    )


def importance_weights(prob, n_valid, beta) -> jax.Array:
    """Max-normalised IS weights (n_valid * prob)**-beta, always finite.

    float32 rounding in the CDF search can (very rarely) pick a zero-mass slot,
    e.g. one of the newest rows whose n-step successors are not written yet;
    (0)**-beta = inf would turn the whole batch's loss into NaN. The probability
    is clamped away from zero, and zero-mass samples get weight 0: they are
    dropped from the loss rather than trained on with a garbage target.
    """
    tiny = jnp.finfo(jnp.float32).tiny
    weight = (n_valid * jnp.maximum(prob, tiny)) ** (-beta)
    weight = jnp.where(prob > 0, weight, 0.0)
    return weight / jnp.maximum(weight.max(), tiny)


def update_priorities(buf: Replay, t, e, priorities) -> Replay:
    return buf._replace(
        priority=buf.priority.at[t, e].set(priorities),
        max_priority=jnp.maximum(buf.max_priority, priorities.max()),
    )
