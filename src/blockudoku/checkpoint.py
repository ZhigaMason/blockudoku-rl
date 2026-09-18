"""Save / load a trained network. A run directory contains:

    config.yaml   full training Config (network shape + value support)
    model.eqx     equinox-serialised online network parameters
"""

from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax

from blockudoku.config import Config, load_config
from blockudoku.network import RainbowNet


def save(run_dir: str | Path, net: RainbowNet, cfg: Config, name: str = "model.eqx") -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(run_dir / "config.yaml")
    eqx.tree_serialise_leaves(run_dir / name, net)


def load(run_dir: str | Path, name: str = "model.eqx") -> tuple[RainbowNet, Config]:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    skeleton = RainbowNet(cfg.net, key=jax.random.key(0))
    return eqx.tree_deserialise_leaves(run_dir / name, skeleton), cfg
