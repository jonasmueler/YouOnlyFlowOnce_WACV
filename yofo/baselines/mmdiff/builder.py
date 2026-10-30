"""Construction of the mmDiff baseline for a dataset."""

from __future__ import annotations

import torch

from yofo.config import DatasetConfig, MmDiffConfig
from yofo.models.skeleton import skeleton_adjacency, skeleton_edges
from yofo.third_party.mmdiff import GCNdiff, PointTransformerReg


def limb_pairs(dataset: DatasetConfig) -> torch.Tensor:
    """Joint pairs whose distances are the limb lengths of a dataset's skeleton.

    Args:
        dataset: Dataset configuration.

    Returns:
        Pairs ``(L, 2)``: 16 for the COCO skeleton, 25 for mmRadPose.
    """
    return skeleton_edges(dataset.skeleton)


def limb_lengths(poses: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Euclidean lengths of the limbs of a batch of poses.

    Args:
        poses: Poses ``(B, K, 3)``.
        pairs: Joint pairs ``(L, 2)``.

    Returns:
        Lengths ``(B, L)``.
    """
    return torch.norm(poses[:, pairs[:, 0]] - poses[:, pairs[:, 1]], dim=-1)


def build_backbone(dataset: DatasetConfig, cfg: MmDiffConfig) -> PointTransformerReg:
    """Create the PointTransformer backbone of mmDiff for a dataset.

    Args:
        dataset: Dataset configuration (point features and joints).
        cfg: mmDiff hyperparameters.

    Returns:
        The backbone.
    """
    return PointTransformerReg(
        input_dim=dataset.input_dim,
        num_keypoints=dataset.num_keypoints,
        nblocks=cfg.nblocks,
        nneighbor=cfg.nneighbor,
        transformer_dim=cfg.transformer_dim,
        feature_dim=cfg.feature_dim,
    )


def build_gcndiff(dataset: DatasetConfig, cfg: MmDiffConfig) -> GCNdiff:
    """Create the GCNdiff denoiser of mmDiff for a dataset.

    Args:
        dataset: Dataset configuration (skeleton and joints).
        cfg: mmDiff hyperparameters.

    Returns:
        The denoiser.
    """
    return GCNdiff(
        adj=skeleton_adjacency(dataset.skeleton, dataset.num_keypoints),
        num_keypoints=dataset.num_keypoints,
        past_frames=cfg.past_frames,
        global_feat_size=cfg.feature_dim,
        num_limbs=int(limb_pairs(dataset).size(0)),
        hid_dim=cfg.hid_dim,
        num_layers=cfg.num_layers,
        n_head=cfg.n_head,
        dropout=cfg.dropout,
    )
