"""Construction of the MG-Prior + DiffPose model from the configuration."""

from __future__ import annotations

from yofo.config import DatasetConfig, DiffPoseConfig
from yofo.third_party.diffpose import DiffPoseDenoiser, GaussianSamplingPoseformer, MultiHypothesisDiffusion


def build_diffpose(dataset: DatasetConfig, cfg: DiffPoseConfig) -> MultiHypothesisDiffusion:
    """Instantiate the diffusion model, its denoiser and its conditioning module.

    Args:
        dataset: Dataset configuration (number of joints).
        cfg: DiffPose hyperparameters.

    Returns:
        The model on the CPU.
    """
    condition_emb = GaussianSamplingPoseformer(
        num_joints=dataset.num_keypoints,
        num_samples=cfg.num_samples,
        condition_dim=cfg.condition_dim,
        embedding_dim=cfg.embedding_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        feedforward_dim=cfg.feedforward_dim,
        backbone_feature_dim=cfg.backbone_feature_dim,
    )
    denoiser = DiffPoseDenoiser(
        num_joints=dataset.num_keypoints,
        cond_dim=condition_emb.output_dim,
        hidden_dim=cfg.hidden_dim,
        use_batchnorm=cfg.use_batchnorm,
    )
    return MultiHypothesisDiffusion(
        denoise_fn=denoiser,
        condition_emb=condition_emb,
        timesteps=cfg.timesteps,
        loss_type=cfg.loss_type,
        objective=cfg.objective,
        beta_schedule=cfg.beta_schedule,
        scaling_3d_pose=cfg.scaling_3d_pose,
        clip_denoised=cfg.clip_denoised,
        noise_scale=cfg.noise_scale,
        num_joints=dataset.num_keypoints,
    )
