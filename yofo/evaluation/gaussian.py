"""Evaluation of the MG-Prior alone on raw radar sequences."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yofo.checkpoints import load_backbone_checkpoint
from yofo.config import BackboneConfig, DatasetConfig, EvalConfig, Paths
from yofo.data.factory import build_dataset, collate_sequences, meta_fields
from yofo.data.gt import last_frame_pose
from yofo.evaluation.common import EvalCollector, group_keys
from yofo.models.mg_prior import MGPrior
from yofo.seeding import worker_init_fn

logger = logging.getLogger(__name__)


def evaluate_gaussian(
    dataset: DatasetConfig,
    split: str,
    checkpoint: Path,
    backbone_cfg: BackboneConfig,
    eval_cfg: EvalConfig,
    paths: Paths,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate an MG-Prior checkpoint with ``eval_cfg.n_samples`` hypotheses per frame.

    Args:
        dataset: Dataset configuration.
        split: Split name.
        checkpoint: MG-Prior checkpoint.
        backbone_cfg: Architecture hyperparameters.
        eval_cfg: Evaluation protocol.
        paths: Root folders.
        device: Device.

    Returns:
        Overall and per-group metrics.
    """
    model = MGPrior(dataset, backbone_cfg).to(device)
    load_backbone_checkpoint(model, checkpoint)
    model.eval()

    test_set = build_dataset(dataset, split, "test", paths.data_root, return_meta=True)
    loader = DataLoader(
        test_set,
        batch_size=eval_cfg.batch_size,
        shuffle=False,
        num_workers=eval_cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_sequences,
        worker_init_fn=worker_init_fn,
    )
    logger.info("Evaluating MG-Prior on %d frames with %d hypotheses", len(test_set), eval_cfg.n_samples)

    keys = group_keys(dataset)
    collector = EvalCollector()
    with torch.no_grad():
        for radar, gt, metas in tqdm(loader, desc="evaluate"):
            radar = radar.to(device).float()
            gt_flat = last_frame_pose(gt.to(device).float(), dataset.gt_scale_to_cm)
            samples, _, _, mu_out, _ = model(radar, n_samples=eval_cfg.n_samples)
            preds = mu_out.view(mu_out.size(0), dataset.num_keypoints, 3)
            gts = gt_flat.view(gt_flat.size(0), dataset.num_keypoints, 3)
            fields = [meta_fields(dataset, m) for m in metas]
            collector.add(preds, gts, samples, {k: [f[k] for f in fields] for k in keys})
    return collector.aggregate(eval_cfg)
