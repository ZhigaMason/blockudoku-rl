"""Rainbow DQN (Hessel et al., 2018) in JAX + Equinox.

Components and where they live:
    distributional C51 ........ `categorical_projection`, `loss_fn`; atoms are
                                log-spaced (network.value_support, "symlog")
    double Q-learning ......... `loss_fn` (online net picks a*, target net evaluates)
    dueling heads ............. network.RainbowNet
    noisy nets ................ network.NoisyLinear (no epsilon-greedy)
    prioritised replay ........ replay.py (+ IS weights here)
    n-step returns ............ replay.sample

One iteration = every env takes a step, then `updates_per_iteration` gradient
updates. On GPU/TPU `run` fuses the iterations of a log window into one jitted
`lax.scan`; on CPU it loops in Python because XLA:CPU runs the model several
times slower inside while loops.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from blockudoku import env, replay
from blockudoku.config import Config
from blockudoku.network import RainbowNet, cast_floats, masked_argmax, q_values, value_support


class EpisodeStats(NamedTuple):
    count: jax.Array  # episodes finished in the current log window
    score_sum: jax.Array
    score_max: jax.Array
    moves_sum: jax.Array


class TrainState(NamedTuple):
    params: eqx.Module  # array leaves of the online net (float32 master copy)
    target_params: eqx.Module
    opt_state: optax.OptState
    buffer: replay.Replay
    envs: env.EnvState  # batched over num_envs
    key: jax.Array
    iteration: jax.Array
    num_updates: jax.Array
    loss_sum: jax.Array  # sum of losses over the current log window
    stats: EpisodeStats


def zero_stats() -> EpisodeStats:
    # distinct buffers: the training step donates its whole input state
    return EpisodeStats(*(jnp.zeros((), jnp.int32) for _ in EpisodeStats._fields))


def categorical_projection(target_probs, returns, discount, support):
    """Project the distribution of r + discount * Z onto a fixed, sorted support.

    target_probs (B, N); returns, discount (B,); support (N,) -> (B, N).
    Each shifted atom x is split between its two neighbouring atoms by linear
    interpolation, which preserves the mean on any (e.g. log-spaced) grid:
        w_i(x) = clip(min((x - z_{i-1}) / (z_i - z_{i-1}), (z_{i+1} - x) / (z_{i+1} - z_i)), 0, 1)
    For a uniform grid this is the classic C51 floor/ceil projection.
    """
    big = 1e9  # virtual atoms far outside both ends: the outer side never binds
    lo = jnp.concatenate([support[:1] - big, support[:-1]])  # z_{i-1}
    hi = jnp.concatenate([support[1:], support[-1:] + big])  # z_{i+1}
    x = jnp.clip(returns[:, None] + discount[:, None] * support[None, :], support[0], support[-1])
    x = x[:, None, :]  # (B, 1, Nj) against atoms i on axis 1
    rise = (x - lo[None, :, None]) / (support - lo)[None, :, None]
    fall = (hi[None, :, None] - x) / (hi - support)[None, :, None]
    weights = jnp.clip(jnp.minimum(rise, fall), 0.0, 1.0)  # (B, Ni, Nj)
    return (weights * target_probs[:, None, :]).sum(axis=-1)


class Rainbow(NamedTuple):
    cfg: Config
    static: eqx.Module
    optimizer: optax.GradientTransformation
    support: jax.Array
    init: Callable[[jax.Array], TrainState]
    run: Callable[[TrainState], TrainState]  # iterations_per_log iterations
    model: Callable[[eqx.Module], RainbowNet]  # params -> full module
    uses_scan: bool


def make_optimizer(cfg: Config) -> optax.GradientTransformation:
    schedule = optax.linear_schedule(0.0, cfg.learning_rate, max(cfg.lr_warmup_updates, 1))

    def decay_mask(params):  # weight-decay matrices only, not biases / LayerNorm gains
        return jax.tree.map(lambda p: p.ndim >= 2, params)

    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adamw(schedule, eps=cfg.adam_eps, weight_decay=cfg.weight_decay, mask=decay_mask),
    )


def make_rainbow(cfg: Config) -> Rainbow:
    E = cfg.num_envs
    support = jnp.asarray(value_support(cfg.v_min, cfg.v_max, cfg.net.num_atoms, cfg.support_scale))
    _, static = eqx.partition(RainbowNet(cfg.net, key=jax.random.key(0)), eqx.is_array)
    optimizer = make_optimizer(cfg)
    compute_dtype = jnp.dtype(cfg.compute_dtype)
    total_iterations = max(cfg.total_env_steps // E, 1)
    uses_scan = cfg.iteration_loop == "scan" or (
        cfg.iteration_loop == "auto" and jax.default_backend() != "cpu")

    def model(params) -> RainbowNet:
        return eqx.combine(params, static)

    def batched_logits(params, obs: env.Obs, key):
        """(B, 243, atoms) float32 logits; one noise sample shared by the batch."""
        net = model(cast_floats(params, compute_dtype))
        return jax.vmap(lambda o: net(o, key=key))(obs)

    def observe_batch(board, hand):
        return jax.vmap(env.observe)(board, hand)

    def loss_fn(params, target_params, batch: replay.Batch, key):
        k_online, k_next, k_target = jax.random.split(key, 3)
        B = batch.action.shape[0]
        obs = observe_batch(batch.board, batch.hand)
        next_obs = observe_batch(batch.next_board, batch.next_hand)
        next_mask = next_obs.planes[:, 1:].reshape(B, env.NUM_ACTIONS) > 0.5

        logits = batched_logits(params, obs, k_online)  # (B, A, N), the only pass we differentiate
        log_p = jax.nn.log_softmax(logits[jnp.arange(B), batch.action], axis=-1)

        # double DQN: online net selects, target net evaluates (no gradients through either)
        frozen = jax.lax.stop_gradient(params)
        next_q = q_values(batched_logits(frozen, next_obs, k_next), support)
        next_a = masked_argmax(next_q, next_mask)
        target_logits = batched_logits(target_params, next_obs, k_target)
        target_p = jax.nn.softmax(target_logits[jnp.arange(B), next_a], axis=-1)
        # a terminal next state has an empty mask; its discount is 0 so next_a is irrelevant
        m = categorical_projection(target_p, batch.returns, batch.discount, support)

        ce = -(m * log_p).sum(axis=-1)  # (B,)
        return (batch.weight * ce).mean(), ce

    def act(params, envs: env.EnvState, key):
        obs = observe_batch(envs.board, envs.hand)
        q = q_values(batched_logits(params, obs, key), support)
        mask = obs.planes[:, 1:].reshape(E, env.NUM_ACTIONS) > 0.5
        return masked_argmax(q, mask)

    def env_step(ts: TrainState, key) -> TrainState:
        actions = act(ts.params, ts.envs, key)
        envs, points, done, final = jax.vmap(env.step_autoreset)(ts.envs, actions)
        buffer = replay.add(ts.buffer, ts.envs.board, ts.envs.hand, actions,
                            points.astype(jnp.float32) * cfg.reward_scale, done)
        d = done.astype(jnp.int32)
        s = ts.stats
        stats = EpisodeStats(
            count=s.count + d.sum(),
            score_sum=s.score_sum + (final.score * d).sum(),
            score_max=jnp.maximum(s.score_max, (final.score * d).max()),
            moves_sum=s.moves_sum + (final.num_moves * d).sum(),
        )
        return ts._replace(envs=envs, buffer=buffer, stats=stats)

    def learn(ts: TrainState, key) -> TrainState:
        k_sample, k_loss = jax.random.split(key)
        progress = jnp.minimum(ts.iteration / total_iterations, 1.0)
        beta = cfg.priority_beta0 + (1.0 - cfg.priority_beta0) * progress
        batch = replay.sample(ts.buffer, k_sample, cfg.batch_size, cfg.n_step, cfg.gamma,
                              cfg.priority_alpha, beta)
        (loss, ce), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            ts.params, ts.target_params, batch, k_loss)
        updates, opt_state = optimizer.update(grads, ts.opt_state, ts.params)
        params = optax.apply_updates(ts.params, updates)
        buffer = replay.update_priorities(ts.buffer, batch.t, batch.e, ce + cfg.priority_eps)

        num_updates = ts.num_updates + 1
        sync = num_updates % cfg.target_update_period == 0
        target = jax.tree.map(lambda p, t: jnp.where(sync, p, t), params, ts.target_params)
        return ts._replace(params=params, target_params=target, opt_state=opt_state,
                           buffer=buffer, num_updates=num_updates, loss_sum=ts.loss_sum + loss)

    def learning_ready(iteration: int) -> bool:
        """Host-side mirror of replay.num_transitions after this iteration's env step."""
        rows = max(min(iteration + 1, cfg.buffer_steps) - cfg.n_step, 0)
        return rows * E >= max(cfg.learning_starts, cfg.batch_size)

    def make_step(with_learning: bool):
        def step(ts: TrainState) -> TrainState:
            key, k_act, k_learn = jax.random.split(ts.key, 3)
            ts = env_step(ts._replace(key=key), k_act)
            if with_learning:
                for k in jax.random.split(k_learn, cfg.updates_per_iteration):
                    ts = learn(ts, k)
            return ts._replace(iteration=ts.iteration + 1)

        return step

    def make_runner(with_learning: bool):
        step = make_step(with_learning)
        if not uses_scan:
            jitted = eqx.filter_jit(step, donate="all")

            def loop(ts, n: int):
                for _ in range(n):
                    ts = jitted(ts)
                return ts

            return loop

        def chunk(ts, n: int):  # n is static: at most a few distinct values get compiled
            return jax.lax.scan(lambda c, _: (step(c), None), ts, None, length=n)[0]

        return eqx.filter_jit(chunk, donate="all")

    runners = {False: make_runner(False), True: make_runner(True)}

    def run(ts: TrainState) -> TrainState:
        """Run cfg.iterations_per_log iterations; resets the window's stats and loss sum."""
        ts = ts._replace(stats=zero_stats(), loss_sum=jnp.zeros(()))
        start = int(ts.iteration)
        end = start + cfg.iterations_per_log
        first_learn = next((i for i in range(start, end) if learning_ready(i)), end)
        for lo, hi, learning in ((start, first_learn, False), (first_learn, end, True)):
            if hi > lo:
                ts = runners[learning](ts, hi - lo)
        return ts

    def init(key) -> TrainState:
        k_net, k_env, key = jax.random.split(key, 3)
        params, _ = eqx.partition(RainbowNet(cfg.net, key=k_net), eqx.is_array)
        return TrainState(
            params=params,
            target_params=jax.tree.map(jnp.copy, params),
            opt_state=optimizer.init(params),
            buffer=replay.init(cfg.buffer_steps, E),
            envs=jax.vmap(env.reset)(jax.random.split(k_env, E)),
            key=key,
            iteration=jnp.zeros((), jnp.int32),
            num_updates=jnp.zeros((), jnp.int32),
            loss_sum=jnp.zeros(()),
            stats=zero_stats(),
        )

    return Rainbow(cfg, static, optimizer, support, init, run, model, uses_scan)


def greedy_policy(net: RainbowNet, support: jax.Array):
    """Noise-free argmax policy: (board, hand) -> action. vmap-able."""

    def policy(board, hand):
        obs = env.observe(board, hand)
        q = q_values(net(obs), support)
        return masked_argmax(q, obs.planes[1:].reshape(env.NUM_ACTIONS) > 0.5)

    return policy
