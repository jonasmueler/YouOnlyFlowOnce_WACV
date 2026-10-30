"""Losses of the MG-Prior (phase 1)."""

from __future__ import annotations

import math

import torch


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence between the latent Gaussian and a standard normal.

    Args:
        mu: Latent means ``(B, D)``.
        logvar: Latent log-variances ``(B, D)``.

    Returns:
        Mean KL divergence over all entries.
    """
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def gaussian_nll_from_cov(
    gt: torch.Tensor,
    mu: torch.Tensor,
    cov: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multivariate Gaussian negative log-likelihood with a variance penalty.

    Args:
        gt: Ground-truth poses ``(B, K)``.
        mu: Predicted means ``(B, K)``.
        cov: Predicted covariance matrices ``(B, K, K)``.
        gamma: Weight of the penalty on the mean predicted variance.

    Returns:
        Tuple of the loss (mean NLL plus penalty) and the mean penalty.
    """
    _, k = gt.shape
    eye = torch.eye(k, device=cov.device, dtype=cov.dtype)
    cov = cov + 1e-6 * eye

    chol = torch.linalg.cholesky(cov)
    diff = (gt - mu).unsqueeze(-1)
    y = torch.linalg.solve_triangular(chol, diff, upper=False)
    maha = y.square().sum(dim=(-2, -1))
    logdet = 2.0 * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)

    nll = 0.5 * (maha + logdet + k * math.log(2.0 * math.pi))
    penalty = gamma * cov.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    return nll.mean() + penalty.mean(), penalty.mean()
