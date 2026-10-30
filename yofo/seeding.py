"""Seeding and determinism."""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

from yofo.config import DETERMINISTIC, GLOBAL_SEED

logger = logging.getLogger(__name__)


def set_seed(seed: int = GLOBAL_SEED, deterministic: bool = DETERMINISTIC) -> int:
    """Seed Python, NumPy and PyTorch, optionally in deterministic mode.

    Deterministic mode disables cuDNN autotuning, sets ``CUBLAS_WORKSPACE_CONFIG``
    and enables PyTorch's deterministic algorithms, so that repeated runs on the
    same hardware and software give identical numbers.

    Args:
        seed: Seed value.
        deterministic: Enable deterministic CUDA algorithms.

    Returns:
        The seed that was set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
        else:
            torch.backends.cudnn.benchmark = True
    logger.info("Seed %d (deterministic=%s)", seed, deterministic)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and Python inside a data-loader worker from the worker's torch seed.

    Args:
        worker_id: Worker index assigned by the data loader (unused, required by the API).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
