"""Construction of the raw-data datasets from a :class:`DatasetConfig`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from yofo.config import DatasetConfig
from yofo.data.mmfi import MMFiDataset
from yofo.data.mmradpose import RadarPoseSequenceDataset
from yofo.data.mri import MRIDataset


def build_dataset(
    dataset: DatasetConfig,
    split: str,
    part: str,
    data_root: Path,
    return_meta: bool = False,
) -> Dataset:
    """Create the train or test dataset of a split.

    Args:
        dataset: Dataset configuration.
        split: Split name (see ``DatasetConfig.splits``).
        part: ``"train"`` or ``"test"``.
        data_root: Dataset root folder.
        return_meta: Return per-window metadata with every sample.

    Returns:
        A dataset yielding ``(radar, pose)`` or ``(radar, pose, meta)``.
    """
    if part not in ("train", "test"):
        raise ValueError(f"part must be 'train' or 'test', got '{part}'.")
    if dataset.name == "mmFI":
        return MMFiDataset(
            root=data_root,
            split=part,
            split_strategy=split,
            protocol=dataset.mmfi_protocol,
            sequence_length=dataset.seq_len,
            frames_per_aggregate=dataset.frames_per_aggregate,
            max_points=dataset.max_points_per_frame,
            normalized=dataset.center_gt_to_root,
            seed=dataset.mmfi_seed,
            return_meta=return_meta,
        )
    if dataset.name == "mmRadPose":
        participants = dataset.train_participants if part == "train" else dataset.test_participants
        return RadarPoseSequenceDataset(
            root_dir=data_root,
            participants=participants,
            time_window=dataset.seq_len,
            frames_per_aggregate=dataset.frames_per_aggregate,
            max_points=dataset.max_points_per_frame,
            exclude_exercises=dataset.exclude_exercises,
            normalized=dataset.center_gt_to_root,
            return_meta=return_meta,
        )
    if dataset.name == "MRI":
        return MRIDataset(
            root_dir=data_root,
            split=part,
            sequence_length=dataset.seq_len,
            frames_per_aggregate=dataset.frames_per_aggregate,
            max_points_per_frame=dataset.max_points_per_frame,
            filter_extreme_poses=dataset.mri_filter_extreme_poses,
            max_pose_value=dataset.mri_max_pose_value,
            clean_gt=dataset.mri_clean_gt,
            filter_jitter=dataset.mri_filter_jitter,
            normalized=dataset.center_gt_to_root,
            return_meta=return_meta,
        )
    raise ValueError(f"Unknown dataset {dataset.name}")


def collate_sequences(batch: list[tuple]) -> tuple:
    """Stack ``(radar, pose)`` samples; metadata, if present, is kept as a list.

    Args:
        batch: Items returned by one of the raw-data datasets.

    Returns:
        Stacked radar and pose tensors, followed by the metadata list when present.
    """
    radar = torch.stack([item[0] for item in batch])
    pose = torch.stack([item[1] for item in batch])
    if len(batch[0]) == 3:
        return radar, pose, [item[2] for item in batch]
    return radar, pose


def meta_fields(dataset: DatasetConfig, meta: Any) -> dict[str, str]:
    """Flatten a dataset-specific metadata object into the keys of the feature files.

    Args:
        dataset: Dataset configuration.
        meta: Metadata object returned by the dataset.

    Returns:
        Dictionary with ``participant`` and, where available, ``angle``, ``exercise``,
        ``scene`` and ``action``.
    """
    if dataset.name == "mmFI":
        return {"participant": meta.subject, "scene": meta.scene, "action": meta.action}
    if dataset.name == "mmRadPose":
        return {"participant": meta.participant, "angle": meta.angle, "exercise": meta.exercise}
    return {"participant": meta.subject}
