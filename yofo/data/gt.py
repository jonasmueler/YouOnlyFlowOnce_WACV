"""Ground-truth handling shared by training, extraction and evaluation."""

from __future__ import annotations

import torch


def last_frame_pose(y: torch.Tensor, scale_to_cm: float) -> torch.Tensor:
    """Select the pose of the last frame and flatten it to centimetres.

    Args:
        y: Ground-truth sequence ``(B, T, J, 3)``.
        scale_to_cm: Factor that converts the loader unit to centimetres.

    Returns:
        Flattened poses ``(B, J*3)`` in centimetres.
    """
    return (y[:, -1] * scale_to_cm).flatten(start_dim=-2, end_dim=-1)
