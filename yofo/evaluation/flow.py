"""Evaluation of MH-NFPG (MG-Prior + NFPG) on extracted features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yofo.checkpoints import load_flow_checkpoint
from yofo.config import DatasetConfig, EvalConfig, FlowConfig
from yofo.data.features import PrecomputedFeatureDataset, collate_with_meta
from yofo.evaluation.common import EvalCollector, group_keys
from yofo.models.flow import sample_poses
from yofo.seeding import worker_init_fn
from yofo.training.flow import build_flow

logger = logging.getLogger(__name__)


def feature_loader(
    feature_folder: Path, dataset: DatasetConfig, past_frames: int, eval_cfg: EvalConfig
) -> DataLoader:
    """Data loader over ``test_features.pt`` with metadata.

    Args:
        feature_folder: Folder with the feature files.
        dataset: Dataset configuration.
        past_frames: Length of the temporal history.
        eval_cfg: Batch size and workers.

    Returns:
        Data loader yielding features, backbone poses, covariances, ground truth,
        history and metadata.
    """
    test_set = PrecomputedFeatureDataset(
        feature_folder / "test_features.pt", dataset.num_keypoints, past_frames, return_meta=True
    )
    if not test_set.has_sample_meta:
        raise ValueError(f"{feature_folder}/test_features.pt has no per-sample metadata.")
    return DataLoader(
        test_set,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=eval_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_with_meta,
        worker_init_fn=worker_init_fn,
    )


def sample_meta_groups(metas: list, keys: tuple[str, ...]) -> dict[str, list[str]]:
    """Group labels of a batch of :class:`SampleMeta` objects."""
    return {key: [getattr(m, key) for m in metas] for key in keys}


def evaluate_flow(
    dataset: DatasetConfig,
    checkpoint: Path,
    feature_folder: Path,
    flow_cfg: FlowConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a flow checkpoint with ``eval_cfg.n_samples`` hypotheses per frame.

    Args:
        dataset: Dataset configuration.
        checkpoint: Flow checkpoint (``final_model.pt`` of a run).
        feature_folder: Folder with ``test_features.pt``.
        flow_cfg: Flow hyperparameters.
        eval_cfg: Evaluation protocol.
        device: Device.

    Returns:
        Overall and per-group metrics.
    """
    flow, cond_module = build_flow(dataset, flow_cfg)
    flow = flow.to(device)
    cond_module = cond_module.to(device)
    load_flow_checkpoint(flow, cond_module, checkpoint)
    flow.eval()
    cond_module.eval()

    loader = feature_loader(feature_folder, dataset, flow_cfg.past_frames, eval_cfg)
    logger.info("Evaluating MH-NFPG on %d frames with %d hypotheses", len(loader.dataset), eval_cfg.n_samples)

    keys = group_keys(dataset)
    collector = EvalCollector()
    with torch.no_grad():
        for features, _, _, gt_pose, x_history, metas in tqdm(loader, desc="evaluate"):
            features = features.to(device).float()
            x_history = x_history.to(device).float()
            cond = cond_module(features, x_history)
            samples = sample_poses(flow, eval_cfg.n_samples, cond)
            preds = samples.mean(dim=1).view(-1, dataset.num_keypoints, 3)
            collector.add(preds, gt_pose, samples.permute(0, 2, 1), sample_meta_groups(metas, keys))
    return collector.aggregate(eval_cfg)
