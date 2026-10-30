"""MG-Prior: spatiotemporal transformer backbone with the GCN Gaussian head."""

from __future__ import annotations

import torch
import torch.nn as nn

from yofo.config import BackboneConfig, DatasetConfig
from yofo.models.gaussian_head import GCNGaussianHead
from yofo.models.spatiotemporal_transformer import SpatiotemporalTransformer


class MGPrior(nn.Module):
    """Multi-hypothesis Gaussian prior over poses (phase 1 of MH-NFPG).

    Args:
        dataset: Dataset configuration (input shape, joints, skeleton).
        cfg: Backbone and head hyperparameters.
    """

    def __init__(self, dataset: DatasetConfig, cfg: BackboneConfig) -> None:
        super().__init__()
        self.num_keypoints = dataset.num_keypoints
        self.cov_eps = cfg.cov_eps
        self.backbone = SpatiotemporalTransformer(
            input_dim=dataset.input_dim,
            max_points=dataset.max_points_per_frame,
            num_frames=dataset.seq_len,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            output_dim=cfg.feat_dim,
        )
        self.head = GCNGaussianHead(
            d_model=cfg.feat_dim,
            d_latent=cfg.d_latent,
            output_dim=dataset.output_dim,
            num_keypoints=dataset.num_keypoints,
            skeleton=dataset.skeleton,
            gcn_hidden=cfg.gcn_hidden,
            gcn_k=cfg.gcn_k,
            default_num_samples=cfg.n_samples_extract,
        )
        self._backbone_features: torch.Tensor | None = None

    def compute_gaussian_stats(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample mean and full sample covariance of pose hypotheses.

        Args:
            samples: Hypotheses ``(B, D, S)``.

        Returns:
            Tuple of mean ``(B, D)`` and covariance ``(B, D, D)`` with jitter on the diagonal.
        """
        _, dim, num_samples = samples.shape
        mu = samples.mean(dim=2)
        centred = samples - mu.unsqueeze(-1)
        cov = torch.matmul(centred, centred.transpose(1, 2)) / (num_samples - 1)
        cov = cov + self.cov_eps * torch.eye(dim, device=samples.device, dtype=samples.dtype).unsqueeze(0)
        return mu, cov

    @property
    def backbone_features(self) -> torch.Tensor:
        """Backbone features of the most recent forward pass, ``(B, feat_dim)``."""
        if self._backbone_features is None:
            raise RuntimeError("Run a forward pass before reading the backbone features.")
        return self._backbone_features

    def forward(
        self, x: torch.Tensor, n_samples: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict pose hypotheses and their Gaussian summary.

        Args:
            x: Radar point clouds ``(B, T, N, F)``.
            n_samples: Latent samples to draw; defaults to the head's default.

        Returns:
            Tuple ``(preds, mu, logvar, mu_out, cov)``: hypotheses ``(B, K*3, S)``,
            latent parameters ``(B, d_latent)`` each, sample mean ``(B, K*3)`` and full
            sample covariance ``(B, K*3, K*3)``.
        """
        features = self.backbone(x)
        preds, mu, logvar, _, _ = self.head(features, n_samples=n_samples)
        mu_out, cov = self.compute_gaussian_stats(preds)
        self._backbone_features = features
        return preds, mu, logvar, mu_out, cov
