"""Noise schedules and indexing helpers of the diffusion process (from DiffPose)."""

from __future__ import annotations

import math

import torch


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Linear beta schedule scaled to the number of timesteps.

    Args:
        timesteps: Number of diffusion steps.

    Returns:
        Betas ``(timesteps,)`` in float64.
    """
    scale = 1000 / timesteps
    return torch.linspace(scale * 0.0001, scale * 0.02, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule of Nichol and Dhariwal (2021).

    Args:
        timesteps: Number of diffusion steps.
        s: Offset that keeps the first betas away from zero.

    Returns:
        Betas ``(timesteps,)`` in float64, clipped to ``[0, 0.999]``.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    """Gather per-sample schedule values and reshape them for broadcasting.

    Args:
        a: Schedule tensor ``(timesteps,)``.
        t: Timesteps ``(B,)``.
        x_shape: Shape of the tensor the values are broadcast against.

    Returns:
        Values ``(B, 1, ..., 1)`` with as many dimensions as ``x_shape``.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
