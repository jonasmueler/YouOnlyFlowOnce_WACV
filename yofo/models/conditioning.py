"""Graph conditioning module of the flow."""

from __future__ import annotations

import torch
import torch.nn as nn

from yofo.models.chebconv import ChebConv


class FlowConditioningModule(nn.Module):
    """Build the conditioning vector of the flow from backbone features and past poses.

    The backbone feature is projected to one vector per joint and passed through a
    Chebyshev graph convolution (global embedding). The past backbone mean poses are
    encoded by a graph convolution each, stacked in time and reduced with a temporal
    convolution and max-pooling (temporal embedding). Both embeddings are summed,
    flattened and mapped to the conditioning dimension.

    Args:
        adj: Normalised skeleton adjacency ``(K, K)``.
        num_keypoints: Number of joints.
        feat_dim: Backbone feature dimension.
        hid_dim: Width of the GCN outputs.
        global_feat_dim: Per-joint width after projecting the backbone feature.
        past_frames: Past backbone poses used as temporal context.
        cond_dim: Output dimension.
        dropout: Dropout inside the output MLP.
        temporal_fidelity: Full-width temporal convolution instead of the 256-channel bottleneck.
    """

    def __init__(
        self,
        adj: torch.Tensor,
        num_keypoints: int,
        feat_dim: int = 512,
        hid_dim: int = 96,
        global_feat_dim: int = 64,
        past_frames: int = 6,
        cond_dim: int = 256,
        dropout: float = 0.1,
        temporal_fidelity: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("adj", adj, persistent=False)
        self.num_keypoints = num_keypoints
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.global_feat_dim = global_feat_dim
        self.past_frames = past_frames
        self.cond_dim = cond_dim

        self.backbone_to_keypoint = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(),
            nn.Linear(feat_dim * 2, num_keypoints * global_feat_dim),
        )
        self.gconv_global = ChebConv(in_c=global_feat_dim, out_c=hid_dim, K=2)
        self.gconv_temporal = ChebConv(in_c=3, out_c=hid_dim, K=2)
        self.temporal_hid_dim = hid_dim * num_keypoints

        if past_frames > 0:
            mid = self.temporal_hid_dim if temporal_fidelity else 256
            self.temporal_conv: nn.Module | None = nn.Sequential(
                nn.Conv1d(self.temporal_hid_dim, mid, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(mid, self.temporal_hid_dim, kernel_size=3, padding=1),
                nn.MaxPool1d(kernel_size=past_frames),
            )
        else:
            self.temporal_conv = None

        self.output_mlp = nn.Sequential(
            nn.Linear(num_keypoints * hid_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, cond_dim),
        )

    def forward(self, feat_cond: torch.Tensor, x_history: torch.Tensor) -> torch.Tensor:
        """Compute the conditioning vector.

        Args:
            feat_cond: Backbone features ``(B, feat_dim)``.
            x_history: Past backbone mean poses ``(B, K, 3 * past_frames)``.

        Returns:
            Conditioning vector ``(B, cond_dim)``.
        """
        batch = feat_cond.size(0)
        adj = self.adj

        cemd = self.backbone_to_keypoint(feat_cond).view(batch, self.num_keypoints, self.global_feat_dim)
        condition_embedding = self.gconv_global(cemd, adj)

        if self.temporal_conv is not None:
            past = [
                self.gconv_temporal(x_history[:, :, i * 3 : (i + 1) * 3], adj) for i in range(self.past_frames)
            ]
            past_emb = torch.stack(past, dim=1).view(batch, self.past_frames, -1).permute(0, 2, 1)
            past_emb = self.temporal_conv(past_emb).squeeze(-1).view(batch, self.num_keypoints, self.hid_dim)
            condition_embedding = condition_embedding + past_emb

        return self.output_mlp(condition_embedding.view(batch, -1))
