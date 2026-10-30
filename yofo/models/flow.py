"""Conditional RealNVP flow with a standard Laplace base distribution (NFPG)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

SQRT2 = math.sqrt(2.0)
LOG2 = math.log(2.0)


class AffineCoupling(nn.Module):
    """One affine coupling layer with a binary mask.

    Half of the coordinates (selected by ``mask``) pass through unchanged and, together
    with the conditioning vector, parameterise the scale and shift of the other half.

    Args:
        in_dim: Pose dimension.
        hidden: Hidden width of the scale-and-shift network.
        cond_dim: Dimension of the conditioning vector (0 for none).
        mask: Binary mask ``(in_dim,)`` of pass-through coordinates.
        dropout: Dropout inside the scale-and-shift network.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        cond_dim: int,
        mask: torch.Tensor,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        self.register_buffer("mask", mask)
        st_input_dim = in_dim + cond_dim if cond_dim > 0 else in_dim
        self.st_net = nn.Sequential(
            nn.Linear(st_input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2 * in_dim),
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the coupling transform or its inverse.

        Args:
            x: Inputs ``(B, in_dim)``.
            cond: Conditioning ``(B, cond_dim)`` or ``None``.
            reverse: Apply the inverse transform.

        Returns:
            Tuple of outputs ``(B, in_dim)`` and log-determinants ``(B,)``.
        """
        x_masked = self.mask * x
        st_input = torch.cat([x_masked, cond], dim=-1) if cond is not None else x_masked
        s, t = torch.chunk(self.st_net(st_input), 2, dim=-1)
        s = torch.tanh(s)
        mbar = 1.0 - self.mask
        if not reverse:
            y = x_masked + mbar * (x * torch.exp(s) + t)
            logdet = torch.sum(mbar * s, dim=-1)
        else:
            y = x_masked + mbar * ((x - t) * torch.exp(-s))
            logdet = -torch.sum(mbar * s, dim=-1)
        return y, logdet


class RealNVPFlow(nn.Module):
    """Stack of affine coupling layers with alternating masks.

    Args:
        in_dim: Pose dimension (joints times three).
        num_layers: Number of coupling layers.
        hidden: Hidden width of the coupling networks.
        cond_dim: Dimension of the conditioning vector.
        dropout: Dropout inside the coupling networks.
    """

    def __init__(
        self,
        in_dim: int,
        num_layers: int = 8,
        hidden: int = 256,
        cond_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim
        base_mask = torch.tensor([1.0 if i % 2 == 0 else 0.0 for i in range(in_dim)])
        alt_mask = 1.0 - base_mask
        masks = [base_mask if i % 2 == 0 else alt_mask for i in range(num_layers)]
        self.layers = nn.ModuleList(
            [AffineCoupling(in_dim, hidden, cond_dim, mask, dropout) for mask in masks]
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Map poses to the base space (forward) or base samples to poses (reverse).

        Args:
            x: Inputs ``(B, in_dim)``.
            cond: Conditioning ``(B, cond_dim)``.
            reverse: Apply the inverse flow.

        Returns:
            Tuple of outputs ``(B, in_dim)`` and summed log-determinants ``(B,)``.
        """
        if x.shape[1] != self.in_dim:
            raise ValueError(f"Expected inputs of dimension {self.in_dim}, got {x.shape[1]}.")
        if cond is not None and (cond.shape[1] != self.cond_dim or cond.shape[0] != x.shape[0]):
            raise ValueError(
                f"Conditioning of shape {tuple(cond.shape)} does not match inputs {tuple(x.shape)}."
            )
        logdet_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        layers = reversed(self.layers) if reverse else self.layers
        y = x
        for layer in layers:
            y, logdet = layer(y, cond=cond, reverse=reverse)
            logdet_total = logdet_total + logdet
        return y, logdet_total


def flow_nll(flow: RealNVPFlow, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
    """Negative log-likelihood of poses under the flow with a Laplace(0, 1/sqrt(2)) base.

    Args:
        flow: The normalizing flow.
        x: Ground-truth poses ``(B, D)``.
        cond: Conditioning ``(B, cond_dim)``.

    Returns:
        Mean negative log-likelihood over the batch.
    """
    _, dim = x.shape
    z, logdet = flow(x, cond=cond, reverse=False)
    logpz = -dim * 0.5 * LOG2 - SQRT2 * torch.sum(torch.abs(z), dim=-1)
    return -(logpz + logdet).mean()


def sample_laplace(shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Draw samples from Laplace(0, 1/sqrt(2)), which has unit variance.

    Args:
        shape: Output shape.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Samples of the requested shape.
    """
    u = torch.rand(shape, device=device, dtype=dtype)
    u = torch.clamp(u, min=1e-7, max=1.0 - 1e-7)
    arg = 1.0 - 2.0 * torch.abs(u - 0.5)
    return -torch.sign(u - 0.5) * torch.log(arg) / SQRT2


def sample_poses(flow: RealNVPFlow, n_samples: int, cond: torch.Tensor) -> torch.Tensor:
    """Draw pose hypotheses by pushing base samples through the inverse flow.

    Args:
        flow: The normalizing flow.
        n_samples: Hypotheses per conditioning vector.
        cond: Conditioning ``(B, cond_dim)``.

    Returns:
        Hypotheses ``(B, n_samples, D)``.
    """
    batch = cond.shape[0]
    z = sample_laplace((batch, n_samples, flow.in_dim), device=cond.device, dtype=cond.dtype)
    cond_expanded = cond.unsqueeze(1).repeat(1, n_samples, 1).reshape(-1, cond.shape[1])
    x, _ = flow(z.reshape(-1, flow.in_dim), cond=cond_expanded, reverse=True)
    return x.reshape(batch, n_samples, flow.in_dim)
