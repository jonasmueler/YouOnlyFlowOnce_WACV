"""PointTransformer backbone of mmDiff (phase 1).

Adapted from https://github.com/FanJunqiao/mmDiff (MIT License), files
``pysensing/mmwave/PC/model/hpe/pointTrans/mmwave_point_transformer_feat.py`` and
``pysensing/mmwave/PC/model/hpe/pointTrans/transformer.py``. The layer structure, the
parameter names and the forward computation are those of the original so that the
released weights load unchanged; ``einops`` calls were replaced by plain reshapes and
the number of joints is an argument.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by index.

    Args:
        points: Point features ``(B, N, C)``.
        idx: Indices ``(B, S)`` or ``(B, S, K)``.

    Returns:
        Gathered features ``(B, S, C)`` or ``(B, S, K, C)``.
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between every pair of points.

    Args:
        src: Points ``(B, N, C)``.
        dst: Points ``(B, M, C)``.

    Returns:
        Distances ``(B, N, M)``.
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


class TransformerBlock(nn.Module):
    """Point transformer block with vector attention over the k nearest neighbours.

    Args:
        d_points: Input and output feature width.
        d_model: Attention width.
        k: Neighbours per point (the point itself included).
    """

    def __init__(self, d_points: int, d_model: int, k: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(nn.Linear(3, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.fc_gamma = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the block.

        Args:
            xyz: Point coordinates ``(B, N, 3)``.
            features: Point features ``(B, N, d_points)``.

        Returns:
            Tuple of updated features ``(B, N, d_points)`` and attention ``(B, N, k, d_model)``.
        """
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, : self.k]
        knn_xyz = index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)

        res = torch.einsum("bmnf,bmnf->bmf", attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn


class TransitionDown(nn.Module):
    """Two pointwise convolutions on the coordinates concatenated with the features.

    No points are subsampled; the name is kept from the original.

    Args:
        in_channel: Input width (3 coordinates plus features).
        internal_channel: Width after the first convolution.
        out_channel: Output width.
    """

    def __init__(self, in_channel: int, internal_channel: int, out_channel: int) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channel, internal_channel, 1), nn.BatchNorm1d(internal_channel), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(internal_channel, out_channel, 1), nn.BatchNorm1d(out_channel), nn.ReLU())

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Apply the convolutions.

        Args:
            xyz: Point coordinates ``(B, N, 3)``.
            features: Point features ``(B, N, D)``.

        Returns:
            Features ``(B, N, out_channel)``.
        """
        new_features = torch.cat([xyz, features], dim=-1).permute(0, 2, 1)
        new_features = self.conv2(self.conv1(new_features))
        return new_features.permute(0, 2, 1)


class PointTransformerBackbone(nn.Module):
    """Stack of point transformer blocks over all points of all frames.

    Args:
        nblocks: Number of transformer blocks (the first one at width 32).
        nneighbor: Neighbours per point.
        input_dim: Features per point.
        transformer_dim: Attention width.
    """

    def __init__(self, nblocks: int, nneighbor: int, input_dim: int, transformer_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 32))
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        append_blocks = nblocks - 1
        for i in range(append_blocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(channel // 2 + 3, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.append_blocks = append_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-point features.

        Args:
            x: Points ``(B, N, C)`` whose first three values are x, y, z.

        Returns:
            Features ``(B, N, 32 * 2 ** (nblocks - 1))``.
        """
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x))[0]
        for i in range(self.append_blocks):
            points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
        return points


class Residual(nn.Module):
    """Residual wrapper ``x + fn(x)``.

    Args:
        fn: Wrapped module.
    """

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the wrapped module and add the input."""
        return self.fn(x) + x


class PreNorm(nn.Module):
    """Layer normalisation applied before a module.

    Args:
        dim: Feature width.
        fn: Wrapped module.
    """

    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise and apply the wrapped module."""
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    """Two-layer MLP with GELU.

    Args:
        dim: Feature width.
        hidden_dim: Hidden width.
        dropout: Dropout probability.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP."""
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention with a GELU after the output projection.

    Args:
        dim: Feature width.
        heads: Attention heads.
        dim_head: Width per head.
        dropout: Dropout after the output projection.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.GELU(), nn.Dropout(dropout)) if project_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Tokens ``(B, N, dim)``.

        Returns:
            Tokens ``(B, N, dim)``.
        """
        batch, n, _ = x.shape
        k, q, v = self.to_k(x), self.to_q(x), self.to_v(x)
        k, q, v = (t.view(batch, n, self.heads, -1).transpose(1, 2) for t in (k, q, v))
        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(batch, n, -1)
        return self.to_out(out)


class PoseTransformer(nn.Module):
    """Pre-norm transformer over the joint tokens and the point features.

    Args:
        dim: Feature width.
        depth: Number of layers.
        heads: Attention heads.
        dim_head: Width per head.
        mlp_dim: Hidden width of the feed-forward MLP.
        dropout: Dropout of the feed-forward MLP.
    """

    def __init__(self, dim: int, depth: int, heads: int, dim_head: int, mlp_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=0.0))),
                        Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all layers."""
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


def joint_positional_embeddings(sequence_length: int, d: int) -> torch.Tensor:
    """Interleaved sinusoidal initialisation of the joint tokens (as in the original).

    Args:
        sequence_length: Number of joints.
        d: Token width.

    Returns:
        Embeddings ``(1, sequence_length, d)``.
    """
    i = np.arange(sequence_length, dtype=np.float64)[:, None]
    j = np.arange(d, dtype=np.float64)[None, :]
    even = np.sin(i / (10000 ** (j / d)))
    odd = np.cos(i / (10000 ** ((j - 1) / d)))
    result = np.where(j % 2 == 0, even, odd)
    return torch.tensor(result[None], dtype=torch.float32)


class PointTransformerReg(nn.Module):
    """PointTransformer regression head of mmDiff: coarse pose and per-joint features.

    The points of all input frames are flattened into one point set. Learnable joint
    tokens are concatenated in front of the point features, a transformer mixes them,
    and the joint tokens are projected to a 32-dimensional feature (``feat``, the
    conditioning of the diffusion model) and to the coarse pose (``pts``).

    Args:
        input_dim: Features per point.
        num_keypoints: Joints.
        nblocks: Point transformer blocks.
        nneighbor: Neighbours per point.
        transformer_dim: Attention width of the point transformer blocks.
        feature_dim: Width of the per-joint feature.
    """

    def __init__(
        self,
        input_dim: int = 5,
        num_keypoints: int = 17,
        nblocks: int = 5,
        nneighbor: int = 16,
        transformer_dim: int = 128,
        feature_dim: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = PointTransformerBackbone(nblocks, nneighbor, input_dim, transformer_dim)
        self.nblocks = nblocks
        self.n_p = num_keypoints
        dim, depth, heads, dim_head, mlp_dim, dropout = 512, 5, 4, 128, 256, 0.0
        self.joint_posembeds_vector = nn.Parameter(joint_positional_embeddings(self.n_p, dim))
        self.transformer = PoseTransformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.fc2 = nn.Sequential(nn.Linear(dim, 256), nn.Dropout(dropout), nn.ReLU(), nn.Linear(256, feature_dim))
        self.fc3 = nn.Sequential(nn.ReLU(), nn.Linear(feature_dim, 64), nn.Dropout(dropout), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the coarse pose and the per-joint features.

        Args:
            x: Radar points ``(B, T, N, C)`` or ``(B, N, C)``.

        Returns:
            Tuple of coarse poses ``(B, K, 3)`` and features ``(B, K, feature_dim)``.
        """
        if x.dim() == 4:
            b, t, n, c = x.shape
            x = x.reshape(b, t * n, c)
        b = x.shape[0]
        points = self.backbone(x)
        joint_embedding = self.joint_posembeds_vector.expand(b, -1, -1)
        embedding = torch.cat([joint_embedding, points], dim=1)
        output = self.transformer(embedding)[:, : self.n_p, :]
        feat = self.fc2(output)
        pts = self.fc3(feat)
        return pts, feat
