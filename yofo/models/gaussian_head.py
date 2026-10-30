"""Gaussian prediction head with GCN refinement (the head of the MG-Prior)."""

from __future__ import annotations

import torch
import torch.nn as nn

from yofo.models.chebconv import ChebConv
from yofo.models.skeleton import skeleton_adjacency


class GCNGaussianHead(nn.Module):
    """Latent Gaussian head whose decoded pose samples are refined along the skeleton.

    The backbone feature is mapped to the mean and log-variance of a latent Gaussian.
    Latent samples are decoded to poses, each pose is refined by a two-layer Chebyshev
    graph convolution over the skeleton with a residual connection, and the sample set
    is returned together with its mean and per-coordinate variance.

    Args:
        d_model: Input feature dimension.
        d_latent: Latent dimension.
        output_dim: Flattened pose dimension (joints times three).
        num_keypoints: Number of joints.
        skeleton: Name of the skeleton edge set.
        gcn_hidden: Hidden width of the GCN refinement.
        gcn_k: Chebyshev polynomial order.
        default_num_samples: Latent samples drawn when ``n_samples`` is not given.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_latent: int = 256,
        output_dim: int = 78,
        num_keypoints: int = 26,
        skeleton: str = "mmradpose",
        gcn_hidden: int = 128,
        gcn_k: int = 2,
        default_num_samples: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.output_dim = output_dim
        self.num_keypoints = num_keypoints
        self.default_num_samples = default_num_samples

        self.register_buffer("adj", skeleton_adjacency(skeleton, num_keypoints))
        self.gconv1 = ChebConv(in_c=3, out_c=gcn_hidden, K=gcn_k)
        self.gconv2 = ChebConv(in_c=gcn_hidden, out_c=3, K=gcn_k)
        self.gcn_relu = nn.ReLU()

        self.mu = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_latent),
        )
        self.logvar = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_latent),
        )
        self.varianceScaling = nn.Parameter(torch.tensor(0.1))
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, 256),
            nn.Linear(256, output_dim),
        )

    def _apply_gcn(self, poses: torch.Tensor) -> torch.Tensor:
        """Refine decoded poses along the skeleton.

        Args:
            poses: Decoded poses ``(B, E, K*3)``.

        Returns:
            Refined poses ``(B, E, K*3)``.
        """
        batch, num_samples, _ = poses.shape
        poses_kp = poses.view(batch * num_samples, self.num_keypoints, 3)
        delta = self.gconv1(poses_kp, self.adj)
        delta = self.gcn_relu(delta)
        delta = self.gconv2(delta, self.adj)
        return (poses_kp + delta).view(batch, num_samples, self.num_keypoints * 3)

    def get_latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Latent mean and log-variance of a feature batch.

        Args:
            x: Features ``(B, d_model)``.

        Returns:
            Tuple of mean and log-variance, each ``(B, d_latent)``.
        """
        return self.mu(x), self.logvar(x)

    def sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Draw one reparameterised latent sample per batch item.

        Args:
            mu: Latent means ``(B, d_latent)``.
            logvar: Latent log-variances ``(B, d_latent)``.

        Returns:
            Latent sample ``(B, d_latent)``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std * self.varianceScaling

    def forward(
        self, x: torch.Tensor, n_samples: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode a set of pose hypotheses from a feature batch.

        Args:
            x: Features ``(B, d_model)``.
            n_samples: Latent samples to draw; defaults to ``default_num_samples``.

        Returns:
            Tuple ``(preds, mu, logvar, mu_out, var_out)`` with the hypotheses
            ``(B, K*3, E)``, the latent parameters ``(B, d_latent)`` each, the
            sample mean ``(B, K*3)`` and the per-coordinate sample variance ``(B, K*3)``.
        """
        if n_samples is None:
            n_samples = self.default_num_samples
        mu, logvar = self.get_latent(x)
        samples_latent = torch.stack([self.sample(mu, logvar) for _ in range(n_samples)], dim=1)
        decoded = self._apply_gcn(self.decoder(samples_latent))
        preds = decoded.permute(0, 2, 1)
        return preds, mu, logvar, preds.mean(dim=2), preds.var(dim=2)
