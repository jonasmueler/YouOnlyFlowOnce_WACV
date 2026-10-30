"""Noise schedule, timestep sampling and the DDIM sampler of mmDiff.

Adapted from https://github.com/FanJunqiao/mmDiff (MIT License), functions
``get_beta_schedule`` and ``compute_alpha`` in ``utils.py`` and ``generalized_steps``
and ``phase2_train`` in ``mmDiff.py``. The arithmetic is that of the original.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def linear_betas(timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    """Linear noise schedule.

    Args:
        timesteps: Number of diffusion steps.
        beta_start: First beta.
        beta_end: Last beta.

    Returns:
        Betas ``(timesteps,)`` as float32.
    """
    return torch.from_numpy(np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)).float()


def alpha_bar(betas: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Cumulative product ``alpha_bar_t`` with ``alpha_bar_{-1} = 1``.

    Args:
        betas: Betas ``(T,)``.
        t: Timesteps ``(B,)`` in ``[-1, T - 1]``.

    Returns:
        ``alpha_bar_t`` as ``(B, 1, 1)``.
    """
    beta = torch.cat([torch.zeros(1, device=betas.device, dtype=betas.dtype), betas], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)


def sample_timesteps_antithetic(n: int, timesteps: int, device: torch.device) -> torch.Tensor:
    """Draw timesteps in antithetic pairs ``(t, T - 1 - t)`` as in the original training.

    Args:
        n: Batch size.
        timesteps: Number of diffusion steps ``T``.
        device: Device of the result.

    Returns:
        Timesteps ``(n,)``.
    """
    t = torch.randint(low=0, high=timesteps, size=(n // 2 + 1,), device=device)
    return torch.cat([t, timesteps - t - 1], dim=0)[:n]


def ddim_sequence(horizon: int, steps: int) -> list[int]:
    """Timesteps visited by the sampler, uniformly skipped over a horizon.

    Args:
        horizon: Number of timesteps of the inference schedule (the sampler starts at
            ``horizon - 1``).
        steps: Requested number of steps; the skip is ``horizon // steps``.

    Returns:
        Ascending timesteps, for example ``[0, 12, 24]`` for ``(25, 2)``.
    """
    skip = horizon // steps
    return list(range(0, horizon, skip))


def generalized_steps(
    model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_start: torch.Tensor,
    seq: list[int],
    betas: torch.Tensor,
    eta: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DDIM sampling from a given start pose over a sequence of timesteps.

    The start pose is used as the state at the last timestep of ``seq`` without added
    noise, exactly as the original refines the coarse pose. At every step the noise
    prediction gives the estimate of the clean pose, which is re-noised to the next
    timestep with a stochastic part scaled by ``eta``. The final step is noise-free.

    Args:
        model_fn: Noise predictor ``(x_t, t) -> eps`` with ``x_t`` and ``eps`` of shape
            ``(B, K, 3)`` and ``t`` of shape ``(B,)``.
        x_start: Start poses ``(B, K, 3)``.
        seq: Ascending timesteps, see :func:`ddim_sequence`.
        betas: Betas ``(T,)``.
        eta: Stochasticity of the DDIM update (0 deterministic, 1 DDPM-like).
        generator: Random generator of the injected noise.

    Returns:
        Final poses ``(B, K, 3)``.
    """
    n = x_start.size(0)
    device = x_start.device
    seq_next = [-1] + list(seq[:-1])
    x = x_start
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        next_t = torch.full((n,), j, device=device, dtype=torch.long)
        at = alpha_bar(betas, t)
        at_next = alpha_bar(betas, next_t)
        et = model_fn(x, t.float())
        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
        c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1**2).sqrt()
        noise = torch.randn(x.shape, generator=generator, device=device, dtype=x.dtype)
        x = at_next.sqrt() * x0_t + c1 * noise + c2 * et
    return x
