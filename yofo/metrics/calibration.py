"""Calibration metrics on sampled hypotheses: ECE, interval coverage, sharpness."""

from __future__ import annotations

import numpy as np
import torch


def cdf_at_ground_truth(samples: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Empirical CDF of the sampled hypotheses evaluated at the ground truth.

    Args:
        samples: Hypotheses ``(B, D, S)``, S samples per batch item and coordinate.
        gt: Ground truth ``(B, D)``.

    Returns:
        Fraction of samples at or below the ground truth, shape ``(B, D)``.
    """
    return (samples <= gt[:, :, None]).mean(axis=2)


def calibration_error_from_cdf(cdf: np.ndarray, num_bins: int = 100) -> float:
    """Root-mean-square distance between expected and observed CDF levels.

    Follows Kuleshov et al. (2018): for probability levels p the observed fraction of
    CDF values at or below p is compared with p itself.

    Args:
        cdf: Empirical CDF values ``(B, D)``.
        num_bins: Number of probability levels between 0.01 and 0.99.

    Returns:
        Calibration error.
    """
    expected = np.linspace(0.01, 0.99, num_bins)
    flat = cdf.flatten()
    observed = np.array([(flat <= p).mean() for p in expected])
    return float(np.sqrt(np.mean((observed - expected) ** 2)))


def expected_calibration_error(samples: np.ndarray, gt: np.ndarray) -> float:
    """Expected calibration error of sampled hypotheses.

    Args:
        samples: Hypotheses ``(B, D, S)``.
        gt: Ground truth ``(B, D)``.

    Returns:
        ECE: RMSE over 100 probability levels.
    """
    return calibration_error_from_cdf(cdf_at_ground_truth(samples, gt))


def interval_coverage(
    samples: np.ndarray,
    gt: np.ndarray,
    levels: tuple[float, ...] = (0.50, 0.90, 0.95),
) -> dict[float, float]:
    """Fraction of ground-truth coordinates inside the central credible intervals.

    Args:
        samples: Hypotheses ``(B, D, S)``.
        gt: Ground truth ``(B, D)``.
        levels: Credible-interval levels.

    Returns:
        Mapping from level to observed coverage.
    """
    num_samples = samples.shape[2]
    sorted_samples = np.sort(samples, axis=2)
    coverage: dict[float, float] = {}
    for level in levels:
        alpha = 1.0 - level
        lower_idx = int(alpha / 2 * num_samples)
        upper_idx = int((1.0 - alpha / 2) * num_samples) - 1
        lower = sorted_samples[:, :, lower_idx]
        upper = sorted_samples[:, :, upper_idx]
        coverage[level] = float(((gt >= lower) & (gt <= upper)).mean())
    return coverage


def sharpness(samples: torch.Tensor) -> torch.Tensor:
    """Mean predictive variance over batch items and coordinates.

    Args:
        samples: Hypotheses ``(B, D, S)``.

    Returns:
        Scalar mean variance.
    """
    return samples.var(dim=-1).mean()
