"""Graph attention and graph convolution layers of mmDiff's denoiser.

Adapted from https://github.com/FanJunqiao/mmDiff (MIT License), files
``pysensing/mmwave/PC/model/hpe/mmDiff/GraFormer.py``, ``ChebConv.py`` and
``mmDiff.py``. Layer structure, parameter names and computations are those of the
original; the Chebyshev convolution is shared with :mod:`yofo.models.chebconv`.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from yofo.models.chebconv import ChebConv


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Deep-copy a module ``n`` times (the copies share their initial weights)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding (sine block followed by cosine block).

    Args:
        timesteps: Timesteps ``(B,)``.
        embedding_dim: Embedding width.

    Returns:
        Embeddings ``(B, embedding_dim)``.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


def silu(x: torch.Tensor) -> torch.Tensor:
    """Swish activation ``x * sigmoid(x)``."""
    return x * torch.sigmoid(x)


class LayerNorm(nn.Module):
    """Layer normalisation with the unbiased standard deviation and ``eps`` on the std.

    Args:
        features: Feature width.
        eps: Added to the standard deviation.
    """

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise over the last dimension."""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """Pre-norm residual connection ``x + dropout(sublayer(norm(x)))``.

    Args:
        size: Feature width.
        dropout: Dropout probability.
    """

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """Apply a sublayer with the residual connection."""
        return x + self.dropout(sublayer(self.norm(x)))


def attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout: nn.Dropout | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention with dropout on the probabilities.

    Args:
        query: Queries ``(B, h, N, d_k)``.
        key: Keys ``(B, h, N, d_k)``.
        value: Values ``(B, h, N, d_k)``.
        dropout: Dropout applied to the attention probabilities.

    Returns:
        Tuple of the attended values and the probabilities.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """Multi-head attention with four linear maps (query, key, value, output).

    Args:
        h: Heads.
        d_model: Feature width.
        dropout: Dropout on the attention probabilities.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn: torch.Tensor | None = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Apply attention over the joints.

        Args:
            query: Tokens ``(B, N, d_model)``.
            key: Tokens ``(B, N, d_model)``.
            value: Tokens ``(B, N, d_model)``.

        Returns:
            Tokens ``(B, N, d_model)``.
        """
        nbatches = query.size(0)
        q, k, v = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(q, k, v, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class LAMGconv(nn.Module):
    """Graph convolution with a symmetrically normalised learnable adjacency.

    Args:
        in_features: Input width.
        out_features: Output width.
        activation: Activation after the linear map, or ``None``.
    """

    def __init__(self, in_features: int, out_features: int, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    @staticmethod
    def laplacian_batch(a_hat: torch.Tensor) -> torch.Tensor:
        """Normalise a batch of adjacency matrices ``D^-1/2 A D^-1/2`` with ``D`` from the column sums."""
        batch, n = a_hat.shape[:2]
        d_hat = (torch.sum(a_hat, 1) + 1e-5) ** (-0.5)
        return d_hat.view(batch, n, 1) * a_hat * d_hat.view(batch, 1, n)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Apply the convolution.

        Args:
            x: Node features ``(B, N, in_features)``.
            a: Adjacency ``(N, N)``.

        Returns:
            Node features ``(B, N, out_features)``.
        """
        batch = x.size(0)
        a_hat = a.unsqueeze(0).repeat(batch, 1, 1)
        x = self.fc(torch.bmm(self.laplacian_batch(a_hat), x))
        if self.activation is not None:
            x = self.activation(x)
        return x


class GraphNet(nn.Module):
    """Two LAM graph convolutions with a learnable adjacency initialised to the identity.

    Args:
        in_features: Input width.
        out_features: Output width.
        n_pts: Nodes.
    """

    def __init__(self, in_features: int = 2, out_features: int = 2, n_pts: int = 21) -> None:
        super().__init__()
        self.A_hat = nn.Parameter(torch.eye(n_pts).float(), requires_grad=True)
        self.gconv1 = LAMGconv(in_features, in_features * 2, nn.ReLU(inplace=True))
        self.gconv2 = LAMGconv(in_features * 2, out_features, activation=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both convolutions."""
        return self.gconv2(self.gconv1(x, self.A_hat), self.A_hat)


class GraAttenLayer(nn.Module):
    """Self-attention followed by a graph network, each with a pre-norm residual.

    Args:
        size: Feature width.
        self_attn: Attention module.
        feed_forward: Graph network used as the feed-forward sublayer.
        dropout: Dropout of the residual connections.
    """

    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the layer.

        Args:
            x: Tokens ``(B, N, size)``.

        Returns:
            Tokens ``(B, N, size)``.
        """
        x = self.sublayer[0](x, lambda y: self.self_attn(y, y, y))
        return self.sublayer[1](x, self.feed_forward)


class GraphConv(nn.Module):
    """Chebyshev graph convolution followed by ReLU, dropout and ReLU.

    Args:
        input_dim: Input width.
        output_dim: Output width.
        p_dropout: Dropout probability, or ``None`` for no dropout.
    """

    def __init__(self, input_dim: int, output_dim: int, p_dropout: float | None = None) -> None:
        super().__init__()
        self.gconv = ChebConv(input_dim, output_dim, K=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p_dropout) if p_dropout is not None else None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply the convolution.

        Args:
            x: Node features ``(B, N, input_dim)``.
            adj: Adjacency ``(N, N)``.

        Returns:
            Node features ``(B, N, output_dim)``.
        """
        x = self.gconv(x, adj)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        return self.relu(x)


class ResChebGCDiff(nn.Module):
    """Residual block of two graph convolutions with the time embedding and condition added between them.

    Args:
        input_dim: Input width.
        output_dim: Output width.
        emd_dim: Width of the time embedding.
        hid_dim: Hidden width.
        p_dropout: Dropout probability of the graph convolutions.
    """

    def __init__(self, input_dim: int, output_dim: int, emd_dim: int, hid_dim: int, p_dropout: float) -> None:
        super().__init__()
        self.gconv1 = GraphConv(input_dim, hid_dim, p_dropout)
        self.gconv2 = GraphConv(hid_dim, output_dim, p_dropout)
        self.temb_proj = nn.Linear(emd_dim, hid_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, temb: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply the block.

        Args:
            x: Node features ``(B, N, input_dim)``.
            adj: Adjacency ``(N, N)``.
            temb: Time embedding ``(B, emd_dim)``.
            condition: Condition ``(B, N, hid_dim)``.

        Returns:
            Node features ``(B, N, output_dim)``.
        """
        residual = x
        out = self.gconv1(x, adj)
        out = out + self.temb_proj(silu(temb))[:, None, :]
        out = out + condition
        out = self.gconv2(out, adj)
        return residual + out
