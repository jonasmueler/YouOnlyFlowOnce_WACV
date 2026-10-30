"""Pre-norm transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Self-attention followed by an MLP, both with residual connections.

    The block applies pre-norm self-attention with a residual connection, an MLP
    with a residual connection, and a final layer norm.

    Args:
        d_model: Token dimension.
        num_heads: Attention heads.
        dim_feedforward: Hidden width of the MLP.
        dropout: Dropout probability in attention and MLP.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a token sequence.

        Args:
            x: Tokens ``(B, L, d_model)``.

        Returns:
            Tokens ``(B, L, d_model)``.
        """
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(x)
        return self.norm2(x)
