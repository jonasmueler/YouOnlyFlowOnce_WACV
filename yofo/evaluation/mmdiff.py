"""Evaluation of the mmDiff baseline: PointTransformer, mmDiff (eta=0) and mmDiff (eta=1)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yofo.baselines.mmdiff import MmDiffFeatureDataset, build_gcndiff, collate_mmdiff
from yofo.config import DatasetConfig, EvalConfig, MmDiffConfig
from yofo.evaluation.common import EvalCollector, group_keys
from yofo.evaluation.flow import sample_meta_groups
from yofo.third_party.mmdiff import GCNdiff, ddim_sequence, generalized_steps, linear_betas

logger = logging.getLogger(__name__)

CM_PER_M = 100.0
SECTIONS = ("pointtransformer", "mmdiff_eta0", "mmdiff_eta1")
"""Result sections: the coarse pose of the backbone, the deterministic refinement and the
stochastic refinement with ``n_hypotheses`` hypotheses."""


def load_gcndiff(dataset: DatasetConfig, cfg: MmDiffConfig, checkpoint: Path, device: torch.device) -> GCNdiff:
    """Build the denoiser and load the final weights of a run.

    Args:
        dataset: Dataset configuration.
        cfg: mmDiff hyperparameters.
        checkpoint: ``final_model.pt`` of a diffusion run.
        device: Device.

    Returns:
        The denoiser in evaluation mode.
    """
    model = build_gcndiff(dataset, cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    logger.info("GCNdiff loaded from %s (epoch %s)", checkpoint, state.get("epoch"))
    return model


def centre_history(history: torch.Tensor, num_keypoints: int, past_frames: int) -> torch.Tensor:
    """Centre every pose slot of a history on its joint 0 (the original's test-time centring).

    Args:
        history: Past poses ``(B, K, 3 * past_frames)``.
        num_keypoints: Joints.
        past_frames: Slots.

    Returns:
        Centred history with the same shape.
    """
    slots = history.view(history.size(0), num_keypoints, past_frames, 3)
    slots = slots - slots[:, :1, :, :]
    return slots.reshape(history.size(0), num_keypoints, 3 * past_frames)


def sample_hypotheses(
    model: GCNdiff,
    features: torch.Tensor,
    history: torch.Tensor,
    start: torch.Tensor,
    seq: list[int],
    betas: torch.Tensor,
    eta: float,
    n_hypotheses: int,
    max_rows: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Refine the coarse poses of a batch with the DDIM sampler, several hypotheses per window.

    The global and temporal conditioning is computed once per window and shared by its
    hypotheses and by all denoising steps (the denoiser output is identical to the plain
    forward pass). The hypotheses of a window are placed in the batch dimension, so one
    denoiser call handles ``windows x hypotheses`` rows; the windows of the batch are
    split into chunks when that exceeds ``max_rows``.

    Args:
        model: Denoiser.
        features: Per-joint features ``(B, K, F)``.
        history: Centred past poses ``(B, K, 3 * past_frames)``.
        start: Centred coarse poses ``(B, K, 3)``.
        seq: Timesteps of the sampler.
        betas: Betas ``(T,)``.
        eta: Stochasticity of the sampler.
        n_hypotheses: Hypotheses per window.
        max_rows: Rows per denoiser call.
        generator: Random generator of the injected noise.

    Returns:
        Hypotheses ``(B, n_hypotheses, K, 3)``.
    """
    batch = start.size(0)
    condition = model.global_embedding(features) + model.temporal_embedding(history)
    windows_per_chunk = max(1, max_rows // n_hypotheses)
    outputs = []
    for begin in range(0, batch, windows_per_chunk):
        end = min(begin + windows_per_chunk, batch)
        count = end - begin

        def expand(tensor: torch.Tensor) -> torch.Tensor:
            """Repeat the windows of the chunk ``n_hypotheses`` times along the batch dimension."""
            return tensor[begin:end].unsqueeze(1).expand(-1, n_hypotheses, -1, -1).reshape(count * n_hypotheses, *tensor.shape[1:])

        feat, hist, cond, x0 = expand(features), expand(history), expand(condition), expand(start)

        def model_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Noise prediction for the current chunk with the shared conditioning."""
            return model(torch.cat([hist, x_t], dim=-1), t, feat, condition=cond)[0]

        out = generalized_steps(model_fn, x0, seq, betas, eta, generator)
        outputs.append(out.view(count, n_hypotheses, *start.shape[1:]))
    return torch.cat(outputs, dim=0)


def evaluate_mmdiff(
    dataset: DatasetConfig,
    checkpoint: Path,
    feature_folder: Path,
    cfg: MmDiffConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Evaluate the three mmDiff rows on the test windows of a split.

    ``pointtransformer`` is the coarse pose of the backbone; ``mmdiff_eta0`` refines it
    with the deterministic sampler of the authors (timesteps 24, 12, 0); ``mmdiff_eta1``
    runs the stochastic sampler over every timestep of the horizon with
    ``eval_cfg.n_samples`` hypotheses and reports the mean pose and the calibration
    metrics. All three use identical windows; metrics are in centimetres.

    Args:
        dataset: Dataset configuration.
        checkpoint: ``final_model.pt`` of a diffusion run.
        feature_folder: Folder with ``mmdiff_test_features.pt``.
        cfg: mmDiff hyperparameters.
        eval_cfg: Evaluation protocol (hypotheses, batch size, coverage levels).
        device: Device.
        seed: Seed of the sampling noise.

    Returns:
        One result dictionary per section plus the sampler description.
    """
    model = load_gcndiff(dataset, cfg, checkpoint, device)
    betas = linear_betas(cfg.timesteps, cfg.beta_start, cfg.beta_end).to(device)
    seq_eta0 = ddim_sequence(cfg.test_horizon, cfg.eta0_steps)
    seq_eta1 = ddim_sequence(cfg.test_horizon, cfg.eta1_steps)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    data = MmDiffFeatureDataset(feature_folder / "mmdiff_test_features.pt", dataset.num_keypoints, cfg.past_frames, return_meta=True)
    if not data.has_sample_meta:
        raise ValueError("The feature file has no per-sample metadata; re-run the extraction.")
    loader = DataLoader(data, batch_size=eval_cfg.batch_size, shuffle=False, num_workers=eval_cfg.num_workers, collate_fn=collate_mmdiff)
    logger.info(
        "Evaluating mmDiff on %d windows: eta=0 timesteps %s, eta=1 timesteps %d..0 with %d hypotheses",
        len(data),
        list(reversed(seq_eta0)),
        seq_eta1[-1],
        eval_cfg.n_samples,
    )

    keys = group_keys(dataset)
    collectors = {name: EvalCollector() for name in SECTIONS}
    with torch.no_grad():
        for features, coarse, gt_pose, history, metas in tqdm(loader, desc="evaluate mmdiff"):
            features = features.to(device).float()
            coarse = coarse.to(device).float()
            gt_pose = gt_pose.to(device).float()
            history = centre_history(history.to(device).float(), dataset.num_keypoints, cfg.past_frames)
            start = coarse - coarse[:, :1, :]
            groups = sample_meta_groups(metas, keys)
            gt_cm = gt_pose * CM_PER_M

            collectors["pointtransformer"].add(start * CM_PER_M, gt_cm, None, groups)

            det = sample_hypotheses(model, features, history, start, seq_eta0, betas, 0.0, 1, cfg.max_rows_per_forward, None)
            collectors["mmdiff_eta0"].add(det[:, 0] * CM_PER_M, gt_cm, None, groups)

            hyp = sample_hypotheses(
                model, features, history, start, seq_eta1, betas, 1.0, eval_cfg.n_samples, cfg.max_rows_per_forward, generator
            )
            hyp_cm = hyp * CM_PER_M
            samples = hyp_cm.reshape(hyp_cm.size(0), eval_cfg.n_samples, -1).permute(0, 2, 1)
            collectors["mmdiff_eta1"].add(hyp_cm.mean(dim=1), gt_cm, samples, groups)

    results: dict[str, Any] = {name: collector.aggregate(eval_cfg) for name, collector in collectors.items()}
    results["sampler"] = {
        "eta0_timesteps": list(reversed(seq_eta0)),
        "eta1_timesteps": list(reversed(seq_eta1)),
        "n_hypotheses": eval_cfg.n_samples,
        "start": "coarse pose at the last timestep, no noise added",
        "noise_seed": seed,
    }
    return results
