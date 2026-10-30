"""Run folders and JSON records."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RUN_PATTERN = re.compile(r"run_(\d+)")


def create_run_folder(base_dir: Path) -> Path:
    """Create the next ``run_NNN`` folder under ``base_dir``.

    Args:
        base_dir: Folder that collects the runs of one model and split.

    Returns:
        Path of the newly created run folder.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [
        int(m.group(1)) for m in (_RUN_PATTERN.match(p.name) for p in base_dir.iterdir()) if m
    ]
    run_folder = base_dir / f"run_{max(existing, default=0) + 1:03d}"
    run_folder.mkdir(parents=True, exist_ok=False)
    logger.info("Run folder %s", run_folder)
    return run_folder


def write_json(data: dict[str, Any], path: Path) -> Path:
    """Write a dictionary as indented JSON.

    Args:
        data: JSON-serialisable dictionary.
        path: Output file.

    Returns:
        The output path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def save_hyperparameters(hyperparams: dict[str, Any], run_folder: Path) -> Path:
    """Store the configuration of a run as ``hyperparameters.json``.

    Args:
        hyperparams: Configuration dictionary.
        run_folder: Run folder.

    Returns:
        Path of the written file.
    """
    record = {"created_at": datetime.now().isoformat(timespec="seconds"), **hyperparams}
    path = write_json(record, run_folder / "hyperparameters.json")
    logger.info("Hyperparameters saved to %s", path)
    return path


def timestamp() -> str:
    """Return a file-name friendly timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
