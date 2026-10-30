"""Spatiotemporal transformer backbone over radar point-cloud sequences."""

from __future__ import annotations

import torch
import torch.nn as nn

from yofo.models.transformer_block import TransformerBlock


class SpatiotemporalTransformer(nn.Module):
    """Full self-attention over all points of all frames in a sequence.

    The ``T`` frames with ``N`` points each are flattened into one sequence of
    ``T * N`` tokens. Every point is embedded together with its normalised frame
    index, a learnable temporal embedding per frame is added, and a class token
    summarises the sequence.

    Args:
        input_dim: Features per radar point.
        max_points: Points per frame (after padding).
        num_frames: Frames per sequence.
        d_model: Transformer width.
        num_heads: Attention heads.
        num_layers: Transformer blocks.
        dim_feedforward: MLP width inside each block.
        dropout: Dropout probability.
        output_dim: Dimension of the returned feature.
    """

    def __init__(
        self,
        input_dim: int = 5,
        max_points: int = 322,
        num_frames: int = 5,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.max_points = max_points
        self.num_frames = num_frames
        self.d_model = d_model
        self.output_dim = output_dim

        self.point_embed = nn.Sequential(
            nn.Linear(input_dim + 1, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.temporal_embed = nn.Embedding(num_frames, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dim_feedforward, dropout) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a point-cloud sequence into one feature vector.

        Args:
            x: Point clouds ``(B, T, N, C)`` or ``(B, N, C)`` for a single frame.

        Returns:
            Features ``(B, output_dim)`` read from the class token.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        batch, num_frames, num_points, _ = x.shape

        frame_coords = torch.linspace(0.0, 1.0, num_frames, device=x.device)
        frame_coords = frame_coords.view(1, num_frames, 1, 1).expand(batch, num_frames, num_points, 1)
        x = torch.cat([x, frame_coords], dim=-1)

        x_embed = self.point_embed(x.view(batch, num_frames * num_points, -1))

        frame_indices = torch.arange(num_frames, device=x.device)
        frame_indices = frame_indices.view(1, num_frames, 1).expand(batch, num_frames, num_points)
        x_embed = x_embed + self.temporal_embed(frame_indices.reshape(batch, num_frames * num_points))

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x_embed = torch.cat([cls_tokens, x_embed], dim=1)
        for block in self.blocks:
            x_embed = block(x_embed)
        return self.out(x_embed[:, 0])
