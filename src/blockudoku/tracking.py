"""Optional Weights & Biases logging, configured by the `wandb:` section of the YAML config.

Credentials are read from a `.env` file (see `.env.example`), found by searching
upward from the working directory. Variables already set in the environment win.
Nothing here imports wandb unless `wandb.enabled` is true.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self

from blockudoku.config import Config


class Tracker:
    """No-op unless cfg.wandb.enabled. Use as a context manager around training."""

    def __init__(self, cfg: Config, run_dir: Path, log=print):
        self.cfg = cfg.wandb
        self.run = None
        if not self.cfg.enabled:
            return
        try:
            import wandb
            from dotenv import find_dotenv, load_dotenv
        except ImportError as e:
            raise RuntimeError("wandb.enabled is true but wandb is not installed: "
                               "run `uv sync --extra wandb` (or disable it in the config)") from e

        env_file = find_dotenv(usecwd=True)
        if env_file:
            load_dotenv(env_file, override=False)
        if self.cfg.mode == "online" and not os.environ.get("WANDB_API_KEY"):
            log("warning: WANDB_API_KEY is not set (no .env found or key missing); "
                "wandb will fall back to ~/.netrc if you ran `wandb login`")

        self._wandb = wandb
        self.run = wandb.init(
            project=self.cfg.project,
            entity=self.cfg.entity or None,
            group=self.cfg.group or None,
            tags=list(self.cfg.tags),
            mode=self.cfg.mode,
            name=run_dir.name,
            dir=str(run_dir),
            config=cfg.to_dict(),
        )
        # env_steps is the x-axis for everything
        self.run.define_metric("env_steps")
        self.run.define_metric("*", step_metric="env_steps")
        log(f"wandb: {self.run.url or f'offline run in {run_dir}'}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.run is not None:
            self.run.finish(exit_code=1 if exc_type else 0)

    def log(self, row: dict) -> None:
        if self.run is not None:
            self.run.log(row)

    def new_best(self, run_dir: Path, row: dict) -> None:
        """Record the best greedy eval and (optionally) upload the checkpoint."""
        if self.run is None:
            return
        self.run.summary["best_eval_score_mean"] = row["eval_score_mean"]
        self.run.summary["best_env_steps"] = row["env_steps"]
        if self.cfg.log_checkpoints:
            art = self._wandb.Artifact(f"{self.run.id}-model", type="model",
                                       metadata={k: v for k, v in row.items() if not isinstance(v, str)})
            art.add_file(str(run_dir / "best.eqx"))
            art.add_file(str(run_dir / "config.yaml"))
            self.run.log_artifact(art, aliases=["best", f"step-{row['env_steps']}"])
