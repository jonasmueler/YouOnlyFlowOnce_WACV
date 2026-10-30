"""Conditioning of DiffPose from MG-Prior samples and backbone features."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding of scalar coordinates.

    Args:
        dim: Embedding dimension.
        base_freq: Base frequency.
    """

    def __init__(self, dim: int, base_freq: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.base_freq = base_freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed values.

        Args:
            x: Values of any shape ``(*,)``.

        Returns:
            Embeddings ``(*, dim)``.
        """
        half_dim = self.dim // 2
        emb = math.log(self.base_freq) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


def _gaussian(mu_flat: torch.Tensor, cov: torch.Tensor) -> MultivariateNormal:
    """Multivariate normal with a jitter fallback for non positive-definite covariances."""
    try:
        return MultivariateNormal(mu_flat, covariance_matrix=cov)
    except ValueError:
        cov_fixed = cov + 1e-4 * torch.eye(cov.shape[-1], device=cov.device)
        return MultivariateNormal(mu_flat, covariance_matrix=cov_fixed)


class GaussianSamplingPoseformer(nn.Module):
    """DiffPose conditioning from samples of the MG-Prior and the backbone feature.

    For every hypothesis the mean pose and ``num_samples - 1`` samples of the MG-Prior's
    Gaussian are embedded per joint, weighted by their likelihood, summed, combined with
    a joint identity embedding and passed through a joint-wise transformer. The result
    is concatenated with a projection of the backbone feature.

    Args:
        num_joints: Number of joints.
        num_samples: Poses per hypothesis including the mean.
        condition_dim: Output width per joint.
        embedding_dim: Width of the coordinate embedding.
        n_heads: Attention heads of the joint-wise transformer.
        n_layers: Layers of the joint-wise transformer.
        feedforward_dim: MLP width of the transformer.
        backbone_feature_dim: Dimension of the backbone feature.
        dropout_prob: Probability of dropping a joint's embedding during training.
    """

    def __init__(
        self,
        num_joints: int = 26,
        num_samples: int = 32,
        condition_dim: int = 128,
        embedding_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        feedforward_dim: int = 512,
        backbone_feature_dim: int = 512,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.num_samples = num_samples - 1
        self.condition_dim = condition_dim
        self.embedding_dim = embedding_dim
        self.backbone_feature_dim = backbone_feature_dim
        self.dropout_prob = dropout_prob

        self.pos_encoder = SinusoidalPosEmb(embedding_dim)
        self.pos_proj = nn.Linear(2 * embedding_dim, 2 * embedding_dim)
        self.joint_embedding = nn.Embedding(num_joints, embedding_dim * 2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=2 * embedding_dim, nhead=n_heads, dim_feedforward=feedforward_dim, batch_first=True
            ),
            num_layers=n_layers,
        )
        self.pos_out = nn.Linear(2 * embedding_dim * num_joints, condition_dim * num_joints)
        self.backbone_proj = nn.Sequential(
            nn.Linear(backbone_feature_dim, condition_dim * num_joints),
            nn.LayerNorm(condition_dim * num_joints),
            nn.SiLU(),
        )

    @property
    def output_dim(self) -> int:
        """Dimension of one conditioning vector."""
        return 2 * self.condition_dim * self.num_joints

    def _sample_from_gaussian(
        self, mu: torch.Tensor, cov: torch.Tensor, num_samples: int, max_samples_per_batch: int = 64
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw poses from the MG-Prior Gaussian together with their log-densities.

        Args:
            mu: Means ``(B, J, 3)``.
            cov: Covariances ``(B, 3J, 3J)``.
            num_samples: Samples per batch item.
            max_samples_per_batch: Samples drawn per chunk.

        Returns:
            Samples ``(B, num_samples, J, 3)`` and log-densities ``(B, num_samples)``.
        """
        batch = mu.shape[0]
        dist = _gaussian(mu.view(batch, -1), cov)
        samples_list = []
        log_prob_list = []
        remaining = num_samples
        while remaining > 0:
            chunk = min(remaining, max_samples_per_batch)
            chunk_samples = dist.rsample((chunk,))
            samples_list.append(chunk_samples)
            log_prob_list.append(dist.log_prob(chunk_samples))
            remaining -= chunk
        samples = torch.cat(samples_list, dim=0).permute(1, 0, 2).view(batch, num_samples, self.num_joints, 3)
        log_probs = torch.cat(log_prob_list, dim=0).permute(1, 0)
        return samples, log_probs

    def _embed_poses(self, poses: torch.Tensor) -> torch.Tensor:
        """Embed every coordinate sinusoidally and sum over x, y and z.

        Args:
            poses: Poses ``(B, S, J, 3)``.

        Returns:
            Embeddings ``(B, S, J, 2 * embedding_dim)`` with zero-padded second half.
        """
        batch, num_samples, num_joints, _ = poses.shape
        embedded = self.pos_encoder(poses.reshape(-1))
        embedded = embedded.view(batch, num_samples, num_joints, 3, self.embedding_dim).sum(dim=3)
        return torch.cat([embedded, torch.zeros_like(embedded)], dim=-1)

    def forward(
        self, mu: torch.Tensor, cov: torch.Tensor, num_hypotheses: int, backbone_features: torch.Tensor
    ) -> torch.Tensor:
        """Build one conditioning vector per hypothesis.

        Args:
            mu: MG-Prior means ``(B, J, 3)`` in metres.
            cov: MG-Prior covariances ``(B, 3J, 3J)`` in square metres.
            num_hypotheses: Hypotheses per batch item.
            backbone_features: Backbone features ``(B, backbone_feature_dim)``.

        Returns:
            Conditioning ``(B, num_hypotheses, output_dim)``.
        """
        batch = mu.shape[0]
        device = mu.device
        total_random = num_hypotheses * self.num_samples
        if self.num_samples > 0:
            random_samples, random_log_probs = self._sample_from_gaussian(mu, cov, total_random)
        mu_flat = mu.view(batch, -1)
        mean_log_prob = _gaussian(mu_flat, cov).log_prob(mu_flat)
        backbone_cond = self.backbone_proj(backbone_features)
        joint_emb = self.joint_embedding(torch.arange(self.num_joints, device=device)).unsqueeze(0).expand(batch, -1, -1)

        conditioning = []
        for h in range(num_hypotheses):
            mean_sample = mu.unsqueeze(1)
            mean_prob = mean_log_prob.unsqueeze(1)
            if self.num_samples > 0:
                start = h * self.num_samples
                samples = torch.cat([mean_sample, random_samples[:, start : start + self.num_samples]], dim=1)
                log_probs = torch.cat([mean_prob, random_log_probs[:, start : start + self.num_samples]], dim=1)
            else:
                samples, log_probs = mean_sample, mean_prob
            probs = torch.softmax(log_probs, dim=-1)
            embedded = self._embed_poses(samples)
            if self.training and self.dropout_prob > 0:
                mask = torch.rand(batch, self.num_joints, device=device) > self.dropout_prob
                embedded = embedded * mask[:, None, :, None].float()
            embedded = self.pos_proj(embedded)
            embedded = (embedded * probs[:, :, None, None]).sum(dim=1)
            embedded = self.transformer(embedded + joint_emb)
            out = self.pos_out(embedded.reshape(batch, -1))
            conditioning.append(torch.cat([backbone_cond, out], dim=-1))
        return torch.stack(conditioning, dim=1)
