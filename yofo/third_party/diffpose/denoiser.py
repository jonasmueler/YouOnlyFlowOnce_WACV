"""MLP denoiser of DiffPose."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two linear layers with leaky ReLU, optional batch norm and a residual connection.

    Args:
        num_neurons: Width of the block.
        use_batchnorm: Apply batch normalisation after each layer.
    """

    def __init__(self, num_neurons: int = 1024, use_batchnorm: bool = False) -> None:
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.l1 = nn.Linear(num_neurons, num_neurons)
        self.bn1 = nn.BatchNorm1d(num_neurons)
        self.l2 = nn.Linear(num_neurons, num_neurons)
        self.bn2 = nn.BatchNorm1d(num_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform a batch of vectors.

        Args:
            x: Inputs ``(B, num_neurons)``.

        Returns:
            Outputs ``(B, num_neurons)``.
        """
        inp = x
        x = F.leaky_relu(self.l1(x))
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.leaky_relu(self.l2(x))
        if self.use_batchnorm:
            x = self.bn2(x)
        return x + inp


class DiffPoseDenoiser(nn.Module):
    """Predict the noise of a noisy pose from the pose, the timestep and the conditioning.

    Args:
        num_joints: Number of joints.
        cond_dim: Dimension of the conditioning vector.
        t_dim: Dimension of the timestep input (a scalar).
        hidden_dim: Width of the residual blocks.
        use_batchnorm: Batch normalisation inside the residual blocks.
    """

    def __init__(
        self,
        num_joints: int = 26,
        cond_dim: int = 3328,
        t_dim: int = 1,
        hidden_dim: int = 1024,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.cond_dim = cond_dim
        self.t_dim = t_dim
        input_dim = num_joints * 3 + cond_dim + t_dim
        self.upscale = nn.Linear(input_dim, hidden_dim)
        self.res_block1 = ResBlock(hidden_dim, use_batchnorm)
        self.res_block2 = ResBlock(hidden_dim, use_batchnorm)
        self.pose_out = nn.Linear(hidden_dim, num_joints * 3)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Denoise one step.

        Args:
            x: Noisy poses ``(B, num_joints * 3)``.
            t: Timesteps ``(B,)``.
            cond: Conditioning ``(B, cond_dim)``.

        Returns:
            Predicted noise ``(B, num_joints * 3)``.
        """
        x = self.upscale(torch.cat([x, t.unsqueeze(-1).to(x.dtype), cond], dim=-1))
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.pose_out(x)
