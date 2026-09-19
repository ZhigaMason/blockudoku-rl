"""Typed schema for YAML training configs (values live in configs/*.yaml).

The dataclasses have no defaults on purpose: a config file must set every field,
so there is exactly one place a value can come from. A file may start with
`extends: <other.yaml>` (path relative to itself) and override a subset;
mappings are merged recursively.

    cfg = load_config("configs/small.yaml", overrides=["net.channels=96"])
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclasses.dataclass(frozen=True)
class NetConfig:
    # transformer trunk
    dim: int  # token width
    depth: int  # number of transformer blocks
    heads: int  # attention heads; head_dim = dim / heads must be divisible by 4 (2D RoPE)
    mlp_ratio: int  # MLP hidden = mlp_ratio * dim
    num_registers: int  # learned memory tokens appended to the sequence (feed the value head)
    rope_base: float  # RoPE frequency base (positions are only 0..8)
    # noisy dueling C51 heads
    adv_hidden: int
    value_hidden: int
    num_atoms: int
    noisy_sigma0: float


@dataclasses.dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    project: str
    entity: str  # "" -> WANDB_ENTITY from the environment / .env, else your default entity
    group: str  # "" -> no group
    tags: list  # list of str
    mode: str  # "online" | "offline" (offline runs can be uploaded later with `wandb sync`)
    log_checkpoints: bool  # upload best.eqx (+ config) as a model artifact on every new best


@dataclasses.dataclass(frozen=True)
class Config:
    seed: int
    # environment / acting
    num_envs: int
    total_env_steps: int
    reward_scale: float
    # prioritised n-step replay; capacity = buffer_steps * num_envs transitions
    buffer_steps: int
    learning_starts: int
    n_step: int
    gamma: float
    priority_alpha: float
    priority_beta0: float
    priority_eps: float
    # distributional support: `num_atoms` atoms from v_min to v_max, spaced
    # uniformly in symlog space ("symlog") or in value space ("linear")
    v_min: float
    v_max: float
    support_scale: str
    # optimisation
    batch_size: int
    updates_per_iteration: int
    learning_rate: float
    lr_warmup_updates: int
    lr_final_fraction: float  # after warm-up, lr decays linearly to this fraction by the last update
    weight_decay: float
    adam_eps: float
    max_grad_norm: float
    target_update_period: int
    compute_dtype: str  # "float32" | "bfloat16" (params and optimiser stay float32)
    iteration_loop: str  # "auto" (scan on GPU/TPU, python on CPU) | "scan" | "python"
    # bookkeeping
    iterations_per_log: int
    eval_every_logs: int
    eval_episodes: int
    eval_max_moves: int
    net: NetConfig
    wandb: WandbConfig

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_yaml())

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        d = dict(d)
        d["net"] = _build(NetConfig, d.get("net", {}), "net.")
        d["wandb"] = _build(WandbConfig, d.get("wandb", {}), "wandb.")
        return _build(cls, d, "")


CHOICES = {
    "support_scale": ("symlog", "linear"),
    "compute_dtype": ("float32", "bfloat16"),
    "iteration_loop": ("auto", "scan", "python"),
    "mode": ("online", "offline"),
}


def _build(cls, d: dict, prefix: str):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - set(fields))
    missing = sorted(set(fields) - set(d))
    if unknown or missing:
        raise ValueError(f"config error: unknown keys {[prefix + k for k in unknown]}, "
                         f"missing keys {[prefix + k for k in missing]}")
    kwargs = {}
    for name, value in d.items():
        typ = fields[name].type
        if typ in ("int", "float"):
            # YAML 1.1 reads "3e-4" (no dot) as a string, so accept numeric strings
            try:
                value = {"int": int, "float": float}[typ](value)
            except (TypeError, ValueError):
                raise ValueError(f"config error: {prefix}{name} must be {typ}, got {value!r}") from None
        elif typ == "bool" and not isinstance(value, bool):
            raise ValueError(f"config error: {prefix}{name} must be true/false, got {value!r}")
        elif typ == "str" and not isinstance(value, str):
            raise ValueError(f"config error: {prefix}{name} must be a string, got {value!r}")
        elif typ == "list" and not (isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise ValueError(f"config error: {prefix}{name} must be a list of strings, got {value!r}")
        if name in CHOICES and value not in CHOICES[name]:
            raise ValueError(f"config error: {prefix}{name} must be one of {CHOICES[name]}, got {value!r}")
        kwargs[name] = value
    return cls(**kwargs)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_yaml(path: str | Path) -> dict:
    """Load a YAML config, resolving `extends:` chains."""
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent is not None:
        data = _merge(read_yaml(path.parent / parent), data)
    return data


def apply_overrides(data: dict, overrides: list[str]) -> dict:
    """Apply `dotted.key=value` overrides; values are parsed as YAML scalars."""
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        *parents, leaf = key.split(".")
        patch: dict[str, Any] = {leaf: yaml.safe_load(raw)}
        for p in reversed(parents):
            patch = {p: patch}
        data = _merge(data, patch)
    return data


def load_config(path: str | Path, overrides: list[str] = ()) -> Config:
    """Resolve a name like "small" to configs/small.yaml; otherwise treat it as a path."""
    p = Path(path)
    if not p.suffix and not p.exists():
        p = CONFIG_DIR / f"{path}.yaml"
    return Config.from_dict(apply_overrides(read_yaml(p), list(overrides)))
