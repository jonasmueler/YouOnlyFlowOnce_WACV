"""Feature files written by ``extract_features.py`` and the dataset that reads them."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

FEATURE_KEYS = ("features", "mu_out", "cov", "gt_pose")
"""Tensor keys of a feature file: ``features (N, 512)``, ``mu_out (N, K, 3)``,
``cov (N, 3K, 3K)`` and ``gt_pose (N, K, 3)``; poses in centimetres."""


def history_indices(idx: int, past_frames: int) -> list[int]:
    """Indices of the samples that form the temporal history of a sample.

    The history consists of the ``past_frames`` preceding samples in file order,
    oldest first. Where a predecessor does not exist (start of the file) the sample
    ``idx + i`` is used in its place. The same rule is used by the flow and by the
    mmDiff baseline.

    Args:
        idx: Index of the sample.
        past_frames: Length of the history.

    Returns:
        ``past_frames`` indices.
    """
    indices = []
    for i in range(past_frames):
        past_idx = idx - past_frames + i
        if past_idx < 0:
            past_idx = idx + i
        indices.append(past_idx)
    return indices


@dataclass
class SampleMeta:
    """Metadata of one feature sample.

    Attributes:
        participant: Subject or participant ID.
        angle: Radar aspect angle (mmRadPose only).
        scene: Environment (MM-Fi only).
        action: Action (MM-Fi only).
    """

    participant: str
    angle: str | None = None
    scene: str | None = None
    action: str | None = None


class PrecomputedFeatureDataset(Dataset):
    """Backbone features with their ground truth and the past backbone predictions.

    The temporal history of a sample consists of the backbone mean poses of the
    ``past_frames`` preceding samples in file order. Near the start of the file, where
    no predecessor exists, the following samples are used instead.

    Args:
        data_path: Feature file (``train_features.pt`` or ``test_features.pt``).
        num_keypoints: Number of joints.
        past_frames: Length of the temporal history.
        return_meta: Also return :class:`SampleMeta`.
    """

    def __init__(
        self,
        data_path: Path,
        num_keypoints: int,
        past_frames: int = 6,
        return_meta: bool = False,
    ) -> None:
        self.num_keypoints = num_keypoints
        self.past_frames = past_frames
        self.return_meta = return_meta

        data = torch.load(data_path, map_location="cpu", weights_only=False)
        self.features: torch.Tensor = data["features"]
        self.mu_out: torch.Tensor = data["mu_out"]
        self.cov: torch.Tensor = data["cov"]
        self.gt_pose: torch.Tensor = data["gt_pose"]
        self.metadata: dict[str, Any] = data.get("metadata", {})
        self.participants: list[str] | None = data.get("participants") or data.get("subjects")
        self.angles: list[str] | None = data.get("angles")
        self.scenes: list[str] | None = data.get("scenes")
        self.actions: list[str] | None = data.get("actions")

        sizes = {self.features.size(0), self.mu_out.size(0), self.cov.size(0), self.gt_pose.size(0)}
        if len(sizes) != 1:
            raise ValueError(f"Inconsistent sample counts in {data_path}: {sizes}")
        logger.info(
            "Loaded %d samples from %s (features %s, poses %s)",
            len(self),
            data_path,
            tuple(self.features.shape[1:]),
            tuple(self.mu_out.shape[1:]),
        )

    @property
    def has_sample_meta(self) -> bool:
        """Whether per-sample participant information is available."""
        return self.participants is not None

    def __len__(self) -> int:
        return self.features.size(0)

    def history(self, idx: int) -> torch.Tensor:
        """Past backbone mean poses of a sample, ``(K, 3 * past_frames)``."""
        frames = [self.mu_out[past_idx] for past_idx in history_indices(idx, self.past_frames)]
        if not frames:
            return torch.zeros(self.num_keypoints, 0)
        return torch.cat(frames, dim=-1)

    def __getitem__(self, idx: int):
        """Return ``(features, mu_out, cov, gt_pose, x_history[, meta])`` for one sample."""
        base = (self.features[idx], self.mu_out[idx], self.cov[idx], self.gt_pose[idx], self.history(idx))
        if self.return_meta and self.has_sample_meta:
            meta = SampleMeta(
                participant=self.participants[idx],
                angle=self.angles[idx] if self.angles else None,
                scene=self.scenes[idx] if self.scenes else None,
                action=self.actions[idx] if self.actions else None,
            )
            return base + (meta,)
        return base


def collate_with_meta(batch: list[tuple]) -> tuple:
    """Stack feature samples and keep their metadata as a list.

    Args:
        batch: Items returned by :class:`PrecomputedFeatureDataset` with metadata.

    Returns:
        Five stacked tensors and the list of :class:`SampleMeta`.
    """
    features, mu_out, cov, gt_pose, history, metas = zip(*batch)
    return (
        torch.stack(features),
        torch.stack(mu_out),
        torch.stack(cov),
        torch.stack(gt_pose),
        torch.stack(history),
        list(metas),
    )
