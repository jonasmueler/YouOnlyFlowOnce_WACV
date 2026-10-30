"""Evaluation of the MG-Prior + DiffPose baseline on extracted features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from yofo.baselines.diffpose import build_diffpose
from yofo.config import DatasetConfig, DiffPoseConfig, EvalConfig
from yofo.evaluation.common import EvalCollector, group_keys
from yofo.evaluation.flow import feature_loader, sample_meta_groups
from yofo.third_party.diffpose import MultiHypothesisDiffusion

logger = logging.getLogger(__name__)


def load_diffpose_checkpoint(model: MultiHypothesisDiffusion, path: Path) -> None:
    """Load the EMA weights of a training run (or the raw weights if no EMA was kept).

    Args:
        model: Model built with the training configuration.
        path: ``final_model.pt`` of a run.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "ema_shadow" in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint["ema_shadow"], strict=False)
        buffers = {name for name, _ in model.named_buffers()}
        unexpected_params = [m for m in missing if m not in buffers]
        if unexpected_params or unexpected:
            raise RuntimeError(f"EMA weights do not match the model: missing {unexpected_params}, unexpected {unexpected}")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("DiffPose checkpoint loaded from %s", path)


def evaluate_diffpose(
    dataset: DatasetConfig,
    checkpoint: Path,
    feature_folder: Path,
    cfg: DiffPoseConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a DiffPose run with ``eval_cfg.n_samples`` hypotheses per frame.

    Args:
        dataset: Dataset configuration.
        checkpoint: ``final_model.pt`` of a training run.
        feature_folder: Folder with ``test_features.pt``.
        cfg: DiffPose hyperparameters.
        eval_cfg: Evaluation protocol.
        device: Device.

    Returns:
        Overall and per-group metrics.
    """
    model = build_diffpose(dataset, cfg).to(device)
    load_diffpose_checkpoint(model, checkpoint)
    model.eval()

    loader = feature_loader(feature_folder, dataset, cfg.past_frames, eval_cfg)
    logger.info("Evaluating MG-Prior + DiffPose on %d frames with %d hypotheses", len(loader.dataset), eval_cfg.n_samples)

    keys = group_keys(dataset)
    collector = EvalCollector()
    with torch.no_grad():
        for features, mu, cov, gt_pose, _, metas in tqdm(loader, desc="evaluate"):
            features = features.to(device).float()
            mu = mu.to(device).float()
            cov = cov.to(device).float()
            samples = model.sample(mu, cov, features, eval_cfg.n_samples, cfg.max_hypotheses_per_batch)
            preds = samples.mean(dim=1)
            batch, n_hyp = samples.shape[:2]
            samples_flat = samples.view(batch, n_hyp, -1).permute(0, 2, 1)
            collector.add(preds, gt_pose, samples_flat, sample_meta_groups(metas, keys))
    return collector.aggregate(eval_cfg)
