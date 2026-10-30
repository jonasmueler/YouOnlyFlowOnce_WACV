"""GCNdiff, the graph-convolutional denoiser of mmDiff (phase 2).

Adapted from https://github.com/FanJunqiao/mmDiff (MIT License), class ``GCNdiff`` in
``pysensing/mmwave/PC/model/hpe/mmDiff/mmDiff.py`` in the configuration the authors
use for radar point clouds of the MM-Fi kind: global conditioning on the per-joint
backbone features, temporal conditioning on the past coarse poses, limb-length
conditioning predicted from the pose history (``limb_flag=2``) and no local radar
conditioning (``local_flag=2`` builds modules the original never runs; they are left
out here). The number of joints, past frames and limbs are arguments; module names
match the original so that its released weights load.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from yofo.models.chebconv import ChebConv
from yofo.third_party.mmdiff.graformer import (
    GraAttenLayer,
    GraphNet,
    MultiHeadedAttention,
    ResChebGCDiff,
    silu,
    timestep_embedding,
)

UPSTREAM_UNUSED_PREFIXES = ("gconv_input.", "gconv_input3.", "local_emb.", "local_attention.")
"""Parameters of the original checkpoints that belong to modules the original never runs."""


class GCNdiff(nn.Module):
    """Denoiser of mmDiff conditioned on backbone features and past coarse poses.

    Args:
        adj: Normalised skeleton adjacency ``(K, K)``.
        num_keypoints: Joints ``K``.
        past_frames: Past coarse poses in the temporal context.
        global_feat_size: Width of the per-joint backbone feature.
        num_limbs: Limb lengths predicted by the limb head.
        hid_dim: Width of the graph layers.
        num_layers: Graph attention layers and residual graph convolution blocks.
        n_head: Attention heads.
        dropout: Dropout of the attention residual connections.
    """

    def __init__(
        self,
        adj: torch.Tensor,
        num_keypoints: int,
        past_frames: int,
        global_feat_size: int = 32,
        num_limbs: int = 16,
        hid_dim: int = 96,
        num_layers: int = 5,
        n_head: int = 4,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.register_buffer("adj", adj.clone().float(), persistent=False)
        self.hid_dim = hid_dim
        self.emd_dim = hid_dim * 4
        self.n_layers = num_layers
        self.feature_size = global_feat_size
        self.past_frames = past_frames
        self.n_pts = num_keypoints
        self.limb_size = num_limbs

        self.gconv_input2 = ChebConv(in_c=3, out_c=hid_dim, K=2)

        attn = MultiHeadedAttention(n_head, hid_dim)
        gcn = GraphNet(in_features=hid_dim, out_features=hid_dim, n_pts=num_keypoints)
        gconv_layers = []
        attention_layers = []
        for _ in range(num_layers):
            gconv_layers.append(
                ResChebGCDiff(input_dim=hid_dim, output_dim=hid_dim, emd_dim=self.emd_dim, hid_dim=hid_dim, p_dropout=0.1)
            )
            attention_layers.append(GraAttenLayer(hid_dim, copy.deepcopy(attn), copy.deepcopy(gcn), dropout))
        self.gconv_layers = nn.ModuleList(gconv_layers)
        self.atten_layers = nn.ModuleList(attention_layers)
        self.gconv_output = ChebConv(in_c=hid_dim, out_c=3 * (past_frames + 1), K=2)

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([nn.Linear(hid_dim, self.emd_dim), nn.Linear(self.emd_dim, self.emd_dim)])

        self.gconv_input1 = ChebConv(in_c=global_feat_size, out_c=hid_dim, K=2)

        self.gconv_input_t = ChebConv(in_c=3, out_c=hid_dim, K=2)
        self.temperal_hid_dim = hid_dim * num_keypoints
        self.temperal_conv = nn.Sequential(
            nn.Conv1d(self.temperal_hid_dim, self.temperal_hid_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.temperal_hid_dim, self.temperal_hid_dim, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=past_frames),
        )

        limb_in = (past_frames + 1) * num_keypoints * 3
        self.limb_linear = nn.Sequential(
            nn.LayerNorm(limb_in), nn.Linear(limb_in, 1024), nn.ReLU(), nn.Linear(1024, num_limbs)
        )
        self.limb_emb_layer = nn.Module()
        self.limb_emb_layer.dense = nn.ModuleList([nn.Linear(num_limbs, self.emd_dim), nn.Linear(self.emd_dim, self.emd_dim)])

    def global_embedding(self, cemd: torch.Tensor) -> torch.Tensor:
        """Global conditioning from the per-joint backbone features.

        Args:
            cemd: Features ``(B, K, global_feat_size)``.

        Returns:
            Embedding ``(B, K, hid_dim)``.
        """
        return self.gconv_input1(cemd, self.adj)

    def temporal_embedding(self, history: torch.Tensor) -> torch.Tensor:
        """Temporal conditioning from the past coarse poses.

        Args:
            history: Past poses ``(B, K, 3 * past_frames)``, oldest first.

        Returns:
            Embedding ``(B, K, hid_dim)``.
        """
        batch = history.size(0)
        x_past = torch.stack(
            [self.gconv_input_t(history[:, :, i * 3 : (i + 1) * 3], self.adj) for i in range(self.past_frames)], dim=1
        )
        past_emb = x_past.reshape(batch, self.past_frames, self.n_pts * self.hid_dim).permute(0, 2, 1)
        return self.temperal_conv(past_emb).reshape(batch, self.n_pts, self.hid_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cemd: torch.Tensor, condition: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict the noise of the current pose and the limb lengths.

        Args:
            x: Pose history and current noisy pose ``(B, K, 3 * (past_frames + 1))``; the
                first ``past_frames`` slots are the past coarse poses (oldest first), the
                last slot is the noisy pose ``x_t``.
            t: Timesteps ``(B,)``.
            cemd: Per-joint backbone features ``(B, K, global_feat_size)``.
            condition: Precomputed sum of the global and temporal embeddings
                ``(B, K, hid_dim)``; computed from ``cemd`` and ``x`` when ``None``.
                The result is identical either way.

        Returns:
            Tuple of the noise prediction ``(B, K, 3)`` and the limb lengths ``(B, num_limbs)``.
        """
        batch = x.size(0)
        temb = timestep_embedding(t, self.hid_dim)
        temb = self.temb.dense[0](temb)
        temb = silu(temb)
        temb = self.temb.dense[1](temb)

        if condition is None:
            condition = self.global_embedding(cemd) + self.temporal_embedding(x[:, :, : 3 * self.past_frames])

        limb_len_pred = self.limb_linear(x.reshape(batch, (self.past_frames + 1) * self.n_pts * 3))
        limb_emb = self.limb_emb_layer.dense[0](limb_len_pred)
        limb_emb = silu(limb_emb)
        limb_emb = self.limb_emb_layer.dense[1](limb_emb)

        condition_embedding = condition
        temb = temb + limb_emb

        out = self.gconv_input2(x[:, :, -3:], self.adj) + condition_embedding
        for i in range(self.n_layers):
            out = self.atten_layers[i](out)
            out = self.gconv_layers[i](out, self.adj, temb, condition_embedding)
        out = self.gconv_output(out, self.adj)
        return out[:, :, -3:], limb_len_pred


def load_upstream_state(model: GCNdiff, state: dict[str, torch.Tensor]) -> None:
    """Load a state dict of the original implementation, ignoring its unused modules.

    Every other key must match exactly (strict loading).

    Args:
        model: Denoiser built with the original's joint count, past frames and limbs.
        state: State dict of the original ``GCNdiff``.
    """
    filtered = {k: v for k, v in state.items() if not k.startswith(UPSTREAM_UNUSED_PREFIXES)}
    model.load_state_dict(filtered, strict=True)
