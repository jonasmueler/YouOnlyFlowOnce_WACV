"""Training of the MG-Prior + DiffPose baseline on extracted features."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from yofo.baselines.diffpose import build_diffpose
from yofo.config import DatasetConfig, DiffPoseConfig, Paths, run_dir, to_dict
from yofo.data.features import PrecomputedFeatureDataset
from yofo.runs import create_run_folder, save_hyperparameters
from yofo.seeding import worker_init_fn
from yofo.training.ema import EMAHelper
from yofo.training.wandb_logging import WandbLogger

logger = logging.getLogger(__name__)


def train_diffpose(
    dataset: DatasetConfig,
    split: str,
    cfg: DiffPoseConfig,
    paths: Paths,
    feature_folder: Path,
    device: torch.device,
    seed: int,
    wandb_logger: WandbLogger,
) -> Path:
    """Train the diffusion baseline and save the final weights with their EMA.

    Args:
        dataset: Dataset configuration.
        split: Split name.
        cfg: DiffPose hyperparameters.
        paths: Root folders.
        feature_folder: Folder with ``train_features.pt``.
        device: Device.
        seed: Random seed (recorded in the run folder).
        wandb_logger: Optional logger.

    Returns:
        The run folder holding ``final_model.pt`` and ``hyperparameters.json``.
    """
    run_folder = create_run_folder(run_dir(paths, dataset, split, "diffusion_multi_hyp"))
    save_hyperparameters(
        {"model": "diffusion_multi_hyp", "split": split, "seed": seed, "feature_dir": str(feature_folder), **to_dict(dataset, cfg)},
        run_folder,
    )

    train_set = PrecomputedFeatureDataset(feature_folder / "train_features.pt", dataset.num_keypoints, cfg.past_frames)
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    model = build_diffpose(dataset, cfg).to(device)
    logger.info("Parameters: %d", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-8)
    ema = EMAHelper(mu=cfg.ema_rate) if cfg.ema else None
    if ema is not None:
        ema.register(model)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        pbar = tqdm(loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for features, mu, cov, gt_pose, _ in pbar:
            features = features.to(device).float()
            mu = mu.to(device).float()
            cov = cov.to(device).float()
            gt_pose = gt_pose.to(device).float()
            optimizer.zero_grad()
            loss = model(gt_pose, mu, cov, backbone_features=features)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total += loss.item()
            count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wandb_logger.log({"loss": loss.item()})
        logger.info("Epoch %d/%d: loss %.4f", epoch, cfg.epochs, total / max(count, 1))
        wandb_logger.log({"epoch/loss": total / max(count, 1)})

    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": cfg.epochs - 1,
    }
    if ema is not None:
        state["ema_shadow"] = ema.state_dict()
    torch.save(state, run_folder / "final_model.pt")
    logger.info("Checkpoint saved to %s", run_folder / "final_model.pt")
    return run_folder
