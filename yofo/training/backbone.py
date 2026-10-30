"""Phase 1: training of the MG-Prior on raw radar sequences."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from yofo.checkpoints import save_backbone_checkpoint
from yofo.config import BackboneConfig, DatasetConfig, Paths, run_dir, to_dict
from yofo.data.factory import build_dataset, collate_sequences
from yofo.data.gt import last_frame_pose
from yofo.losses import gaussian_nll_from_cov, kl_loss
from yofo.models.mg_prior import MGPrior
from yofo.runs import create_run_folder, save_hyperparameters
from yofo.seeding import set_seed, worker_init_fn
from yofo.training.wandb_logging import WandbLogger

logger = logging.getLogger(__name__)


def gaussian_loss(
    model: MGPrior,
    radar: torch.Tensor,
    gt: torch.Tensor,
    cfg: BackboneConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL-regularised Gaussian negative log-likelihood of a batch.

    Args:
        model: MG-Prior.
        radar: Radar sequences ``(B, T, N, F)``.
        gt: Flattened ground-truth poses ``(B, K*3)`` in centimetres.
        cfg: Training hyperparameters.

    Returns:
        Tuple of total loss, weighted KL term, NLL term and variance penalty.
    """
    _, mu, logvar, mu_out, cov = model(radar, n_samples=cfg.n_samples_train)
    kl = kl_loss(mu, logvar) * cfg.beta
    nll, penalty = gaussian_nll_from_cov(gt, mu_out, cov, cfg.gamma)
    return nll + kl, kl, nll, penalty


def train_epoch(
    model: MGPrior,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    dataset: DatasetConfig,
    cfg: BackboneConfig,
    device: torch.device,
    wandb_logger: WandbLogger,
) -> float:
    """Run one training epoch.

    Args:
        model: MG-Prior.
        loader: Training data loader.
        optimizer: Optimizer.
        dataset: Dataset configuration (ground-truth unit).
        cfg: Training hyperparameters.
        device: Device.
        wandb_logger: Optional logger.

    Returns:
        Mean total loss over the epoch.
    """
    model.train()
    total = 0.0
    count = 0
    for radar, gt in tqdm(loader, desc="train"):
        radar = radar.to(device).float()
        gt = last_frame_pose(gt.to(device).float(), dataset.gt_scale_to_cm)
        optimizer.zero_grad()
        loss, kl, nll, penalty = gaussian_loss(model, radar, gt, cfg)
        wandb_logger.log({"loss": loss.item(), "KLL": kl.item(), "nLL": nll.item(), "pen": penalty.item()})
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
        optimizer.step()
        total += loss.item()
        count += 1
    return total / max(count, 1)


def train_backbone(
    dataset: DatasetConfig,
    split: str,
    cfg: BackboneConfig,
    paths: Paths,
    device: torch.device,
    seed: int,
    deterministic: bool,
    wandb_logger: WandbLogger,
) -> Path:
    """Train the MG-Prior for the configured number of epochs and save the final state.

    The order of random operations matches the original experiments: the model is
    initialised after the first seeding, the seed is set again before the optimizer is
    created, and the data loader shuffles after that.

    Args:
        dataset: Dataset configuration.
        split: Split name.
        cfg: Training hyperparameters.
        paths: Root folders.
        device: Device.
        seed: Random seed.
        deterministic: Deterministic CUDA algorithms.
        wandb_logger: Optional logger.

    Returns:
        The run folder holding ``final_model.pt`` and ``hyperparameters.json``.
    """
    model = MGPrior(dataset, cfg).to(device)
    logger.info("Parameters: %d", sum(p.numel() for p in model.parameters()))

    train_set = build_dataset(dataset, split, "train", paths.data_root)
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size_for(dataset),
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_sequences,
        worker_init_fn=worker_init_fn,
    )
    logger.info("Training windows: %d", len(train_set))

    set_seed(seed, deterministic)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    scheduler = MultiplicativeLR(optimizer, lambda _epoch: cfg.lr_decay)

    epochs = cfg.epochs_for(dataset)
    run_folder = create_run_folder(run_dir(paths, dataset, split, "gaussian"))
    save_hyperparameters(
        {"model": "gaussian", "split": split, "epochs": epochs, "seed": seed, **to_dict(dataset, cfg)},
        run_folder,
    )
    for epoch in range(epochs):
        mean_loss = train_epoch(model, loader, optimizer, dataset, cfg, device, wandb_logger)
        scheduler.step()
        logger.info("Epoch %d/%d: loss %.4f", epoch + 1, epochs, mean_loss)
    save_backbone_checkpoint(model, optimizer, epochs - 1, run_folder / "final_model.pt")
    return run_folder
