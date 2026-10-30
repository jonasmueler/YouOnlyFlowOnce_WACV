"""MPJPE and Procrustes-aligned MPJPE."""

from __future__ import annotations

import torch


def procrustes_align(target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Align ``source`` to ``target`` with a similarity transform (Procrustes analysis).

    Args:
        target: Reference joints ``(J, 3)``.
        source: Joints to align ``(J, 3)``.

    Returns:
        Aligned copy of ``source`` with shape ``(J, 3)``.
    """
    mu_target = target.mean(dim=0, keepdim=True)
    mu_source = source.mean(dim=0, keepdim=True)
    target0 = target - mu_target
    source0 = source - mu_source

    norm_target = torch.sqrt((target0**2).sum())
    norm_source = torch.sqrt((source0**2).sum())
    target0 = target0 / norm_target
    source0 = source0 / norm_source

    a = target0.T @ source0
    u, s, vt = torch.linalg.svd(a, full_matrices=False)
    rotation = vt.T @ u.T
    scale = s.sum()
    return norm_target * scale * (source0 @ rotation) + mu_target


def _as_joints(preds: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reshape flat poses ``(B, J*3)`` to ``(B, J, 3)``; pass ``(B, J, 3)`` through."""
    if preds.dim() == 2:
        num_joints = preds.size(1) // 3
        return preds.view(preds.size(0), num_joints, 3), targets.view(targets.size(0), num_joints, 3)
    if preds.dim() != 3:
        raise ValueError(f"Expected poses of shape (B, J, 3) or (B, J*3), got {tuple(preds.shape)}.")
    return preds, targets


def mpjpe(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean per-joint position error.

    Args:
        preds: Predicted joints ``(B, J, 3)`` or ``(B, J*3)``.
        targets: Ground-truth joints with the same shape.

    Returns:
        Scalar mean Euclidean joint error in the unit of the inputs.
    """
    preds, targets = _as_joints(preds, targets)
    return torch.norm(preds - targets, dim=-1).mean()


def pa_mpjpe(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Procrustes-aligned mean per-joint position error.

    Each predicted pose is aligned to its ground truth with a similarity transform
    before the joint error is measured.

    Args:
        preds: Predicted joints ``(B, J, 3)`` or ``(B, J*3)``.
        targets: Ground-truth joints with the same shape.

    Returns:
        Scalar mean aligned joint error in the unit of the inputs.
    """
    preds, targets = _as_joints(preds, targets)
    aligned = torch.stack([procrustes_align(t, p) for p, t in zip(preds, targets)])
    return torch.norm(aligned - targets, dim=-1).mean()
