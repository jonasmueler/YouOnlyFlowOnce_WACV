"""Optional Weights & Biases logging."""

from __future__ import annotations

import logging
from typing import Any

try:
    import wandb
except ImportError:  # wandb is optional
    wandb = None

logger = logging.getLogger(__name__)


class WandbLogger:
    """Thin wrapper that logs to Weights & Biases only when enabled.

    Args:
        enabled: Start a run; requires the ``wandb`` package.
        project: Project name.
        config: Run configuration stored with the run.
    """

    def __init__(self, enabled: bool, project: str, config: dict[str, Any]) -> None:
        self.enabled = enabled
        if enabled and wandb is None:
            raise SystemExit("--wandb requires the wandb package: pip install wandb")
        if enabled:
            wandb.init(project=project, config=config)
            logger.info("Weights & Biases run started in project %s", project)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a metrics dictionary.

        Args:
            metrics: Values to log.
            step: Optional global step.
        """
        if self.enabled:
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        """Close the run."""
        if self.enabled:
            wandb.finish()
