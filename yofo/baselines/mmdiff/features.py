"""Feature files of the mmDiff backbone and the dataset that reads them."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from yofo.data.features import SampleMeta, history_indices

logger = logging.getLogger(__name__)

MMDIFF_FEATURE_KEYS = ("features", "coarse_pose", "gt_pose")
"""Tensor keys of an mmDiff feature file: ``features (N, K, 32)``, ``coarse_pose (N, K, 3)``
and ``gt_pose (N, K, 3)``; poses in metres, centred on joint 0, in dataset order."""


class MmDiffFeatureDataset(Dataset):
    """Per-joint backbone features with the coarse poses, ground truth and pose history.

    The history of a sample consists of the coarse poses of the ``past_frames``
    preceding samples in file order (the rule of :func:`yofo.data.features.history_indices`,
    shared with the flow).

    Args:
        data_path: ``mmdiff_train_features.pt`` or ``mmdiff_test_features.pt``.
        num_keypoints: Joints.
        past_frames: Length of the history.
        return_meta: Also return :class:`~yofo.data.features.SampleMeta`.
    """

    def __init__(self, data_path: Path, num_keypoints: int, past_frames: int, return_meta: bool = False) -> None:
        self.num_keypoints = num_keypoints
        self.past_frames = past_frames
        self.return_meta = return_meta
        data = torch.load(data_path, map_location="cpu", weights_only=False)
        self.features: torch.Tensor = data["features"]
        self.coarse_pose: torch.Tensor = data["coarse_pose"]
        self.gt_pose: torch.Tensor = data["gt_pose"]
        self.metadata: dict[str, Any] = data.get("metadata", {})
        self.participants: list[str] | None = data.get("participants")
        self.angles: list[str] | None = data.get("angles")
        self.scenes: list[str] | None = data.get("scenes")
        self.actions: list[str] | None = data.get("actions")
        sizes = {self.features.size(0), self.coarse_pose.size(0), self.gt_pose.size(0)}
        if len(sizes) != 1:
            raise ValueError(f"Inconsistent sample counts in {data_path}: {sizes}")
        if self.features.dim() != 3 or self.features.size(1) != num_keypoints or self.coarse_pose.shape[1:] != (num_keypoints, 3):
            raise ValueError(f"Unexpected tensor shapes in {data_path}: {tuple(self.features.shape)}, {tuple(self.coarse_pose.shape)}")
        logger.info("Loaded %d mmDiff samples from %s (features %s)", len(self), data_path, tuple(self.features.shape[1:]))

    @property
    def has_sample_meta(self) -> bool:
        """Whether per-sample participant information is available."""
        return self.participants is not None

    def __len__(self) -> int:
        return self.features.size(0)

    def history(self, idx: int) -> torch.Tensor:
        """Past coarse poses of a sample, ``(K, 3 * past_frames)``, oldest first."""
        return torch.cat([self.coarse_pose[i] for i in history_indices(idx, self.past_frames)], dim=-1)

    def meta(self, idx: int) -> SampleMeta:
        """Metadata of a sample."""
        return SampleMeta(
            participant=self.participants[idx],
            angle=self.angles[idx] if self.angles else None,
            scene=self.scenes[idx] if self.scenes else None,
            action=self.actions[idx] if self.actions else None,
        )

    def __getitem__(self, idx: int):
        """Return ``(features, coarse_pose, gt_pose, history[, meta])`` for one sample."""
        base = (self.features[idx], self.coarse_pose[idx], self.gt_pose[idx], self.history(idx))
        if self.return_meta and self.has_sample_meta:
            return base + (self.meta(idx),)
        return base


def collate_mmdiff(batch: list[tuple]) -> tuple:
    """Stack mmDiff feature samples; metadata, if present, is kept as a list.

    Args:
        batch: Items returned by :class:`MmDiffFeatureDataset`.

    Returns:
        Four stacked tensors, followed by the list of metadata when present.
    """
    tensors = tuple(torch.stack([item[i] for item in batch]) for i in range(4))
    if len(batch[0]) == 5:
        return tensors + ([item[4] for item in batch],)
    return tensors
