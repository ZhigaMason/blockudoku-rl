import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blockudoku import checkpoint
from blockudoku.config import load_config
from blockudoku.network import value_support
from blockudoku.rainbow import categorical_projection, lr_schedule, make_rainbow, total_updates
from blockudoku.train import train, warm_start


def _uniform_c51_projection(p, returns, discount, support):
    """Classic floor/ceil C51 projection (uniform support only), as an oracle."""
    v_min, v_max = support[0], support[-1]
    dz = support[1] - support[0]
    tz = np.clip(returns[:, None] + discount[:, None] * support[None, :], v_min, v_max)
    b = (tz - v_min) / dz
    lo, hi = np.floor(b).astype(int), np.ceil(b).astype(int)
    m = np.zeros_like(p)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if lo[i, j] == hi[i, j]:
                m[i, lo[i, j]] += p[i, j]
            else:
                m[i, lo[i, j]] += p[i, j] * (hi[i, j] - b[i, j])
                m[i, hi[i, j]] += p[i, j] * (b[i, j] - lo[i, j])
    return m


def test_projection_matches_classic_c51_on_uniform_support():
    support = jnp.linspace(0.0, 10.0, 11)
    p = jax.nn.softmax(jax.random.normal(jax.random.key(0), (5, 11)))
    returns = jnp.array([0.0, 0.3, 1.7, 2.0, 0.5])
    discount = jnp.array([0.9, 0.9, 0.0, 0.5, 1.0])
    m = categorical_projection(p, returns, discount, support)
    want = _uniform_c51_projection(*(np.asarray(x, np.float64) for x in (p, returns, discount, support)))
    np.testing.assert_allclose(np.asarray(m), want, atol=1e-5)


def test_projection_on_log_support_preserves_mass_and_mean():
    support = jnp.asarray(value_support(0.0, 5000.0, 101, "symlog"))
    p = jax.nn.softmax(jax.random.normal(jax.random.key(1), (6, 101)) * 3)
    returns = jnp.array([0.0, 3.0, 18.5, 250.0, 7.0, 1.0])
    discount = jnp.array([0.99**3, 0.99**3, 0.0, 0.5, 0.0, 0.2])
    m = categorical_projection(p, returns, discount, support)
    assert np.allclose(m.sum(-1), 1.0, atol=1e-5)
    assert (np.asarray(m) >= 0).all()
    # nothing reaches v_max here, so linear interpolation keeps the mean exactly
    mean_in = returns + discount * (p * support).sum(-1)
    np.testing.assert_allclose((m * support).sum(-1), mean_in, rtol=1e-4, atol=1e-3)
    # terminal transition with return 18.5 -> all mass on the two atoms around it
    nz = np.nonzero(np.asarray(m[2]) > 1e-7)[0]
    assert len(nz) == 2 and support[nz[0]] <= 18.5 <= support[nz[1]]


@pytest.mark.parametrize("loop", ["python", "scan"])
def test_run_learns_and_updates_everything(loop):
    cfg = load_config("smoke", ["learning_starts=64", "iterations_per_log=40", "batch_size=16",
                                "lr_warmup_updates=1", f"iteration_loop={loop}"])
    rb = make_rainbow(cfg)
    ts0 = rb.init(jax.random.key(0))
    p0 = jax.tree.map(np.asarray, ts0.params)
    ts = rb.run(ts0)
    assert int(ts.iteration) == 40
    assert rb.uses_scan == (loop == "scan")
    assert int(ts.num_updates) > 0 and np.isfinite(float(ts.loss_sum))
    gmax, gmean = float(ts.grad_norm_max), float(ts.grad_norm_sum) / int(ts.num_updates)
    assert 0 < gmean <= gmax and np.isfinite(gmax)
    changed = jax.tree.map(lambda a, b: not np.allclose(a, np.asarray(b)), p0, ts.params)
    assert all(jax.tree.leaves(changed)), "some parameters received no gradient"
    assert int(ts.buffer.size) == 40


def test_bfloat16_training_step_is_finite():
    cfg = load_config("smoke", ["learning_starts=64", "iterations_per_log=20", "batch_size=16",
                                "compute_dtype=bfloat16"])
    rb = make_rainbow(cfg)
    ts = rb.run(rb.init(jax.random.key(0)))
    assert int(ts.num_updates) > 0 and np.isfinite(float(ts.loss_sum))
    assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(ts.params)), "master weights stay f32"


def test_lr_warms_up_then_decays_linearly_to_final_fraction():
    cfg = load_config("smoke", ["learning_rate=1.0e-3", "lr_warmup_updates=10", "lr_final_fraction=0.1"])
    sched, n = lr_schedule(cfg), total_updates(cfg)
    assert abs(n - (cfg.total_env_steps - cfg.learning_starts) / cfg.num_envs * cfg.updates_per_iteration) <= 2
    lr = np.array([float(sched(i)) for i in range(n + 20)])
    assert lr[0] == 0.0 and np.isclose(lr[10], 1e-3)
    assert (np.diff(lr[:11]) > 0).all() and (np.diff(lr[10:n + 1]) < 0).all()
    np.testing.assert_allclose(lr[(10 + n) // 2], 0.55e-3, rtol=1e-2)  # linear midpoint
    np.testing.assert_allclose(lr[n:], 1e-4, rtol=1e-6)  # held at the final value


def test_train_writes_checkpoint(tmp_path):
    cfg = load_config("smoke", ["total_env_steps=800", "eval_every_logs=1", "eval_episodes=4"])
    last = train(cfg, tmp_path, log=lambda *_: None)
    assert "eval_score_mean" in last
    assert {"grad_norm_mean", "grad_norm_max", "lr"} <= last.keys()
    _, cfg2 = checkpoint.load(tmp_path)
    assert cfg2 == cfg
    assert (tmp_path / "best.eqx").exists() and (tmp_path / "metrics.jsonl").exists()


def test_train_logs_to_wandb_offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here; offline mode needs no credentials
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.setenv("WANDB_SILENT", "true")
    cfg = load_config("smoke", ["total_env_steps=800", "eval_every_logs=1", "eval_episodes=4",
                                "wandb.enabled=true", "wandb.mode=offline"])
    train(cfg, tmp_path / "run", log=lambda *_: None)
    runs = list((tmp_path / "run" / "wandb").glob("offline-run-*"))
    assert len(runs) == 1 and any(runs[0].glob("*.wandb"))


def test_tracker_reads_credentials_from_dotenv(tmp_path, monkeypatch):
    from blockudoku.tracking import Tracker

    (tmp_path / ".env").write_text("WANDB_API_KEY=dummy-key-for-test\nWANDB_ENTITY=someone\n")
    (tmp_path / "sub").mkdir()
    monkeypatch.chdir(tmp_path / "sub")  # found by searching upward
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    for var in ("WANDB_API_KEY", "WANDB_ENTITY"):
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)  # registered with monkeypatch -> restored afterwards
    cfg = load_config("smoke", ["wandb.enabled=true", "wandb.mode=offline"])
    with Tracker(cfg, tmp_path / "run", log=lambda *_: None) as tracker:
        import os

        assert os.environ["WANDB_API_KEY"] == "dummy-key-for-test"
        assert tracker.run is not None


def test_time_limit_stops_early_with_final_eval(tmp_path):
    cfg = load_config("smoke", ["total_env_steps=100000", "eval_every_logs=1000"])
    last = train(cfg, tmp_path, log=lambda *_: None, time_limit_hours=1e-9)
    assert last["env_steps"] == cfg.iterations_per_log * cfg.num_envs  # one window only
    assert "eval_score_mean" in last and (tmp_path / "best.eqx").exists()


def test_warm_start_loads_online_and_target_nets(tmp_path):
    cfg = load_config("smoke", ["total_env_steps=800", "eval_every_logs=1", "eval_episodes=4"])
    train(cfg, tmp_path / "a", log=lambda *_: None)
    rb = make_rainbow(cfg)
    ts = rb.init(jax.random.key(1))
    for path in (tmp_path / "a", tmp_path / "a" / "best.eqx"):
        loaded = warm_start(rb, ts, path, log=lambda *_: None)
        want, _ = checkpoint.load(path if path.is_dir() else path.parent,
                                  "model.eqx" if path.is_dir() else path.name)
        for net in (rb.model(loaded.params), rb.model(loaded.target_params)):
            for a, b in zip(jax.tree.leaves(net), jax.tree.leaves(want)):
                np.testing.assert_array_equal(a, b)
    rb.run(loaded)  # donation-safe: online and target nets are distinct buffers
    last = train(cfg, tmp_path / "b", log=lambda *_: None, init_from=tmp_path / "a")
    assert "eval_score_mean" in last


def test_warm_start_rejects_mismatched_net(tmp_path):
    cfg = load_config("smoke", ["total_env_steps=800", "eval_every_logs=1", "eval_episodes=4"])
    train(cfg, tmp_path, log=lambda *_: None)
    other = load_config("smoke", ["v_max=123.0"])
    rb = make_rainbow(other)
    with pytest.raises(ValueError, match="v_max"):
        warm_start(rb, rb.init(jax.random.key(0)), tmp_path, log=lambda *_: None)
