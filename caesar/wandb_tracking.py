"""
Weights & Biases (W&B) experiment tracking for CAESAR co-evolutionary runs.

Usage
-----
    from caesar.wandb_tracking import WandbTracker

    tracker = WandbTracker(project="CAESAR", run_name="coevo_seed42")
    tracker.init(config={"episodes": 150, "seed": 42})

    for episode in range(episodes):
        # ... training step ...
        tracker.log({
            "episode": episode,
            "neutralization_rate": nr,
            "robustness_score": rs,
            "coevo_gap": gap,
            "attacker_fitness": f_att,
            "defender_fitness": f_def,
        })

    tracker.finish()

    # Or use as context manager:
    with WandbTracker("CAESAR") as tracker:
        tracker.init(config)
        ...
"""

from __future__ import annotations

from typing import Any

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class WandbTracker:
    """Thin W&B wrapper for CAESAR co-evolutionary training loops."""

    def __init__(
        self,
        project: str = "CAESAR",
        run_name: str | None = None,
        entity: str | None = None,
    ) -> None:
        if not _WANDB_AVAILABLE:
            raise ImportError("wandb not installed. Run: pip install wandb")
        self.project = project
        self.run_name = run_name
        self.entity = entity
        self._run = None

    def init(self, config: dict[str, Any] | None = None) -> None:
        self._run = wandb.init(
            project=self.project,
            name=self.run_name,
            entity=self.entity,
            config=config or {},
            reinit=True,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is None:
            raise RuntimeError("Call tracker.init() before logging.")
        wandb.log({k: v for k, v in metrics.items()
                   if isinstance(v, (int, float))}, step=step)

    def log_summary(self, metrics: dict[str, float]) -> None:
        """Write final summary metrics (shown prominently in W&B UI)."""
        for k, v in metrics.items():
            wandb.run.summary[k] = v

    def watch(self, model, log_freq: int = 50) -> None:
        """Track gradients and parameters of a PyTorch model."""
        wandb.watch(model, log="all", log_freq=log_freq)

    def finish(self) -> None:
        if self._run:
            wandb.finish()
            self._run = None

    def __enter__(self) -> "WandbTracker":
        return self

    def __exit__(self, *_) -> None:
        self.finish()


# ------------------------------------------------------------------
# Standalone helper — log a completed CAESAR result dict directly
# ------------------------------------------------------------------
def log_caesar_results(
    results: dict[str, Any],
    config: dict[str, Any],
    project: str = "CAESAR",
    run_name: str = "caesar_run",
) -> None:
    """One-shot logging of a finished CAESAR experiment."""
    tracker = WandbTracker(project=project, run_name=run_name)
    tracker.init(config=config)

    metrics = {
        "neutralization_rate": results.get("neutralization_rate", 0.0),
        "robustness_score":    results.get("robustness_score", 0.0),
        "coevo_gap":           results.get("coevo_gap", 0.0),
        "attacker_fitness":    results.get("attacker_fitness", 0.0),
        "defender_fitness":    results.get("defender_fitness", 0.0),
        "healing_success":     results.get("healing_success", 0.0),
    }
    tracker.log(metrics)
    tracker.log_summary(metrics)
    tracker.finish()
