"""Gaussian diffusion over poses conditioned on the MG-Prior (from DiffPose)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from yofo.third_party.diffpose.schedules import cosine_beta_schedule, extract, linear_beta_schedule

CM_PER_M = 100.0


class MultiHypothesisDiffusion(nn.Module):
    """DDPM over flattened poses whose denoiser is conditioned per hypothesis.

    Poses are converted from centimetres to metres and multiplied by
    ``scaling_3d_pose`` before diffusion; samples are mapped back to centimetres.

    Args:
        denoise_fn: Denoiser network.
        condition_emb: Conditioning module (MG-Prior samples and backbone feature).
        timesteps: Diffusion steps.
        loss_type: ``"l1"`` or ``"l2"``.
        objective: ``"pred_noise"`` or ``"pred_x0"``.
        beta_schedule: ``"cosine"`` or ``"linear"``.
        scaling_3d_pose: Scale applied to poses in metres.
        clip_denoised: Clip the denoised estimate to three times the scale.
        noise_scale: Scale of the Gaussian noise.
        num_joints: Number of joints.
    """

    def __init__(
        self,
        denoise_fn: nn.Module,
        condition_emb: nn.Module,
        timesteps: int = 25,
        loss_type: str = "l2",
        objective: str = "pred_noise",
        beta_schedule: str = "cosine",
        scaling_3d_pose: float = 10.0,
        clip_denoised: bool = True,
        noise_scale: float = 1.0,
        num_joints: int = 26,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.denoise_fn = denoise_fn
        self.condition_emb = condition_emb
        self.objective = objective
        self.scaling_3d_pose = scaling_3d_pose
        self.clip_denoised = clip_denoised
        self.noise_scale = noise_scale
        self.loss_type = loss_type
        self.num_timesteps = int(timesteps)

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        buffers = {
            "betas": betas,
            "alphas_cumprod": alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
            "log_one_minus_alphas_cumprod": torch.log(1.0 - alphas_cumprod),
            "sqrt_recip_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod),
            "sqrt_recipm1_alphas_cumprod": torch.sqrt(1.0 / alphas_cumprod - 1),
            "posterior_variance": posterior_variance,
            "posterior_log_variance_clipped": torch.log(posterior_variance.clamp(min=1e-20)),
            "posterior_mean_coef1": betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            "posterior_mean_coef2": (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        }
        for name, value in buffers.items():
            self.register_buffer(name, value.to(torch.float32))

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Estimate the clean pose from a noisy pose and the predicted noise."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean, variance and log-variance of ``q(x_{t-1} | x_t, x_0)``."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return (
            posterior_mean,
            extract(self.posterior_variance, t, x_t.shape),
            extract(self.posterior_log_variance_clipped, t, x_t.shape),
        )

    def p_mean_variance(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Model estimate of the mean and variance of ``p(x_{t-1} | x_t)``."""
        model_output = self.denoise_fn(x, t, cond)
        if self.objective == "pred_noise":
            x_start = self.predict_start_from_noise(x, t=t, noise=model_output)
        elif self.objective == "pred_x0":
            x_start = model_output
        else:
            raise ValueError(f"Unknown objective: {self.objective}")
        if self.clip_denoised:
            clip_val = 3.0 * self.scaling_3d_pose
            x_start = x_start.clamp(-clip_val, clip_val)
        return self.q_posterior(x_start=x_start, x_t=x, t=t)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """One reverse diffusion step."""
        batch = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = torch.randn_like(x) * self.noise_scale
        nonzero_mask = (1 - (t == 0).float()).reshape(batch, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        """Run the full reverse process from Gaussian noise.

        Args:
            cond: Conditioning ``(N, cond_dim)``.
            shape: Shape of the sampled poses ``(N, num_joints * 3)``.

        Returns:
            Samples in the scaled diffusion space.
        """
        device = self.betas.device
        x = torch.randn(shape, device=device) * self.noise_scale
        for i in reversed(range(self.num_timesteps)):
            x = self.p_sample(x, torch.full((shape[0],), i, device=device, dtype=torch.long), cond)
        return x

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion to timestep ``t``."""
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Denoising loss at random timesteps."""
        noise = torch.randn_like(x_start) * self.noise_scale
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.denoise_fn(x_t, t, cond)
        target = noise if self.objective == "pred_noise" else x_start
        loss_fn = F.l1_loss if self.loss_type == "l1" else F.mse_loss
        loss = loss_fn(model_output, target, reduction="none").reshape(x_start.shape[0], -1).mean(dim=1)
        return loss.mean()

    def forward(
        self, gt_pose: torch.Tensor, mu: torch.Tensor, cov: torch.Tensor, backbone_features: torch.Tensor
    ) -> torch.Tensor:
        """Training loss of a batch.

        Args:
            gt_pose: Ground truth ``(B, J, 3)`` in centimetres.
            mu: MG-Prior means ``(B, J, 3)`` in centimetres.
            cov: MG-Prior covariances ``(B, 3J, 3J)`` in square centimetres.
            backbone_features: Backbone features ``(B, feat_dim)``.

        Returns:
            Scalar loss.
        """
        batch = gt_pose.shape[0]
        cond = self.condition_emb(mu / CM_PER_M, cov / CM_PER_M**2, 1, backbone_features).squeeze(1)
        t = torch.randint(0, self.num_timesteps, (batch,), device=gt_pose.device).long()
        gt_flat = (gt_pose / CM_PER_M).view(batch, -1) * self.scaling_3d_pose
        return self.p_losses(gt_flat, t, cond)

    @torch.no_grad()
    def sample(
        self,
        mu: torch.Tensor,
        cov: torch.Tensor,
        backbone_features: torch.Tensor,
        n_hypotheses: int,
        max_hypotheses_per_batch: int = 32,
    ) -> torch.Tensor:
        """Draw pose hypotheses.

        Args:
            mu: MG-Prior means ``(B, J, 3)`` in centimetres.
            cov: MG-Prior covariances ``(B, 3J, 3J)`` in square centimetres.
            backbone_features: Backbone features ``(B, feat_dim)``.
            n_hypotheses: Hypotheses per batch item.
            max_hypotheses_per_batch: Hypotheses generated per chunk.

        Returns:
            Hypotheses ``(B, n_hypotheses, J, 3)`` in centimetres.
        """
        batch = mu.shape[0]
        mu_m, cov_m = mu / CM_PER_M, cov / CM_PER_M**2
        chunks = []
        remaining = n_hypotheses
        while remaining > 0:
            count = min(remaining, max_hypotheses_per_batch)
            cond = self.condition_emb(mu_m, cov_m, count, backbone_features).reshape(batch * count, -1)
            samples = self.p_sample_loop(cond, (batch * count, self.num_joints * 3))
            samples = samples / self.scaling_3d_pose * CM_PER_M
            chunks.append(samples.view(batch, count, self.num_joints, 3))
            remaining -= count
        return torch.cat(chunks, dim=1)
