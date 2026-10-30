"""Metric computation and grouping shared by all evaluations."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from yofo.config import DatasetConfig, EvalConfig
from yofo.metrics import expected_calibration_error, interval_coverage, mpjpe, pa_mpjpe, sharpness

logger = logging.getLogger(__name__)


@dataclass
class EvalCollector:
    """Accumulates predictions, ground truth, hypotheses and metadata over batches.

    Attributes:
        preds: Point estimates ``(N, K, 3)`` per batch.
        gts: Ground truth ``(N, K, 3)`` per batch.
        samples: Hypotheses ``(N, K*3, S)`` per batch.
        groups: Per-sample group labels, for example ``{"subject": [...], "angle": [...]}``.
    """

    preds: list[torch.Tensor] = field(default_factory=list)
    gts: list[torch.Tensor] = field(default_factory=list)
    samples: list[torch.Tensor] = field(default_factory=list)
    groups: dict[str, list[str]] = field(default_factory=dict)

    def add(
        self, preds: torch.Tensor, gts: torch.Tensor, samples: torch.Tensor | None, groups: dict[str, list[str]]
    ) -> None:
        """Append one batch (tensors are moved to the CPU).

        Args:
            preds: Point estimates ``(B, K, 3)``.
            gts: Ground truth ``(B, K, 3)``.
            samples: Hypotheses ``(B, K*3, S)``, or ``None`` for a deterministic model
                (then only MPJPE and PA-MPJPE are computed).
            groups: Group labels of the batch items.
        """
        self.preds.append(preds.detach().cpu())
        self.gts.append(gts.detach().cpu())
        if samples is not None:
            self.samples.append(samples.detach().cpu())
        for key, values in groups.items():
            self.groups.setdefault(key, []).extend(values)

    def aggregate(self, eval_cfg: EvalConfig) -> dict[str, Any]:
        """Compute overall metrics and metrics per group value.

        Args:
            eval_cfg: Coverage levels.

        Returns:
            Dictionary with ``overall`` and one ``per_<group>`` dictionary per group key.
        """
        preds = torch.cat(self.preds)
        gts = torch.cat(self.gts)
        samples = torch.cat(self.samples) if self.samples else None
        results: dict[str, Any] = {"overall": compute_metrics(preds, gts, samples, eval_cfg.coverage_levels)}
        for key, labels in self.groups.items():
            labels_arr = np.array(labels)
            per_group: dict[str, Any] = {}
            for label in sorted(set(labels)):
                idx = np.where(labels_arr == label)[0]
                group_samples = samples[idx] if samples is not None else None
                per_group[label] = compute_metrics(preds[idx], gts[idx], group_samples, eval_cfg.coverage_levels)
            results[f"per_{key}"] = per_group
        return results


def compute_metrics(
    preds: torch.Tensor,
    gts: torch.Tensor,
    samples: torch.Tensor | None,
    coverage_levels: tuple[float, ...],
) -> dict[str, float]:
    """Accuracy and calibration metrics of a set of frames.

    Args:
        preds: Point estimates ``(N, K, 3)`` in centimetres.
        gts: Ground truth ``(N, K, 3)`` in centimetres.
        samples: Hypotheses ``(N, K*3, S)`` in centimetres, or ``None`` for a
            deterministic model.
        coverage_levels: Credible-interval levels.

    Returns:
        ``mpjpe``, ``pa_mpjpe`` and ``n``; with hypotheses also ``ece``, ``sharpness``
        (centimetres), ``sharpness_var`` (mean variance) and
        ``coverage_<level>``.
    """
    metrics: dict[str, float] = {
        "mpjpe": round(mpjpe(preds, gts).item(), 4),
        "pa_mpjpe": round(pa_mpjpe(preds, gts).item(), 4),
        "n": int(preds.size(0)),
    }
    if samples is None:
        return metrics
    gts_flat = gts.view(gts.size(0), -1).numpy()
    samples_np = samples.numpy()
    var = sharpness(samples).item()
    metrics["ece"] = round(expected_calibration_error(samples_np, gts_flat), 4)
    metrics["sharpness"] = round(math.sqrt(var), 4)
    metrics["sharpness_var"] = round(var, 4)
    for level, value in interval_coverage(samples_np, gts_flat, coverage_levels).items():
        metrics[f"coverage_{int(round(level * 100))}"] = round(value, 4)
    return metrics


def group_keys(dataset: DatasetConfig) -> tuple[str, ...]:
    """Metadata fields by which the evaluation of a dataset is broken down."""
    if dataset.name == "mmFI":
        return ("participant", "scene", "action")
    if dataset.name == "mmRadPose":
        return ("participant", "angle")
    return ("participant",)


def log_summary(results: dict[str, Any], dataset: DatasetConfig) -> None:
    """Log the overall metrics and the per-angle breakdown of mmRadPose.

    Args:
        results: Output of :meth:`EvalCollector.aggregate`.
        dataset: Dataset configuration.
    """
    overall = results["overall"]
    if "ece" not in overall:
        logger.info("Overall (%d frames): MPJPE %.3f  PA-MPJPE %.3f", overall["n"], overall["mpjpe"], overall["pa_mpjpe"])
    else:
        logger.info(
            "Overall (%d frames): MPJPE %.3f  PA-MPJPE %.3f  ECE %.4f  sharpness %.3f cm  "
            "coverage@50 %.3f  @90 %.3f  @95 %.3f",
            overall["n"],
            overall["mpjpe"],
            overall["pa_mpjpe"],
            overall["ece"],
            overall["sharpness"],
            overall["coverage_50"],
            overall["coverage_90"],
            overall["coverage_95"],
        )
    if dataset.name == "mmRadPose":
        for angle, metrics in sorted(results.get("per_angle", {}).items()):
            logger.info("  %s: MPJPE %.3f  PA-MPJPE %.3f", angle, metrics["mpjpe"], metrics["pa_mpjpe"])
