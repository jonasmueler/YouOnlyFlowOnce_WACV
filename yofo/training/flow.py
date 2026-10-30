"""Phase 2: training of the conditional flow (NFPG) on extracted features."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from yofo.checkpoints import load_flow_checkpoint, save_flow_checkpoint
from yofo.config import DatasetConfig, FlowConfig, Paths, run_dir, to_dict
from yofo.data.features import PrecomputedFeatureDataset
from yofo.models.conditioning import FlowConditioningModule
from yofo.models.flow import RealNVPFlow, flow_nll
from yofo.models.skeleton import skeleton_adjacency
from yofo.runs import create_run_folder, save_hyperparameters
from yofo.seeding import worker_init_fn
from yofo.training.wandb_logging import WandbLogger

logger = logging.getLogger(__name__)


def build_flow(dataset: DatasetConfig, cfg: FlowConfig) -> tuple[RealNVPFlow, FlowConditioningModule]:
    """Instantiate the flow and its conditioning module for a dataset.

    Args:
        dataset: Dataset configuration (joints and skeleton).
        cfg: Flow hyperparameters.

    Returns:
        Tuple of the flow and the conditioning module (on the CPU).
    """
    flow = RealNVPFlow(
        in_dim=dataset.output_dim,
        num_layers=cfg.flow_num_layers,
        hidden=cfg.flow_hidden,
        cond_dim=cfg.cond_dim,
        dropout=cfg.dropout,
    )
    cond_module = FlowConditioningModule(
        adj=skeleton_adjacency(dataset.skeleton, dataset.num_keypoints),
        num_keypoints=dataset.num_keypoints,
        feat_dim=cfg.feat_dim,
        hid_dim=cfg.hid_dim,
        global_feat_dim=cfg.global_feat_dim,
        past_frames=cfg.past_frames,
        cond_dim=cfg.cond_dim,
        dropout=cfg.dropout,
        temporal_fidelity=cfg.temporal_fidelity,
    )
    return flow, cond_module


def train_epoch(
    flow: RealNVPFlow,
    cond_module: FlowConditioningModule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: FlowConfig,
    device: torch.device,
    epoch: int,
    wandb_logger: WandbLogger,
) -> float:
    """Run one training epoch.

    Args:
        flow: RealNVP flow.
        cond_module: Conditioning module.
        loader: Training data loader.
        optimizer: Optimizer.
        cfg: Flow hyperparameters.
        device: Device.
        epoch: Epoch number (for the progress bar).
        wandb_logger: Optional logger.

    Returns:
        Mean negative log-likelihood over the epoch.
    """
    flow.train()
    cond_module.train()
    total = 0.0
    count = 0
    params = list(flow.parameters()) + list(cond_module.parameters())
    pbar = tqdm(loader, desc=f"epoch {epoch}")
    for features, _, _, gt_pose, x_history in pbar:
        features = features.to(device).float()
        gt_pose = gt_pose.to(device).float()
        x_history = x_history.to(device).float()
        optimizer.zero_grad()
        cond = cond_module(features, x_history)
        loss = flow_nll(flow, gt_pose.view(gt_pose.size(0), -1), cond=cond)
        loss.backward()
        nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        total += loss.item()
        count += 1
        pbar.set_postfix(nll=f"{loss.item():.4f}")
        wandb_logger.log({"flow_nll": loss.item()})
    return total / max(count, 1)


def train_flow(
    dataset: DatasetConfig,
    split: str,
    cfg: FlowConfig,
    paths: Paths,
    feature_folder: Path,
    device: torch.device,
    seed: int,
    resume_from: Path | None,
    wandb_logger: WandbLogger,
) -> Path:
    """Train the flow for the configured number of epochs and save the final state.

    Args:
        dataset: Dataset configuration.
        split: Split name.
        cfg: Flow hyperparameters.
        paths: Root folders.
        feature_folder: Folder with ``train_features.pt``.
        device: Device.
        seed: Random seed (recorded in the run folder).
        resume_from: Checkpoint to continue from, or ``None``.
        wandb_logger: Optional logger.

    Returns:
        The run folder holding ``final_model.pt`` and ``hyperparameters.json``.
    """
    epochs = cfg.epochs_for(dataset)
    run_folder = create_run_folder(run_dir(paths, dataset, split, "norm_flow_graph"))
    save_hyperparameters(
        {
            "model": "norm_flow_graph",
            "split": split,
            "epochs": epochs,
            "seed": seed,
            "feature_dir": str(feature_folder),
            **to_dict(dataset, cfg),
        },
        run_folder,
    )

    train_set = PrecomputedFeatureDataset(feature_folder / "train_features.pt", dataset.num_keypoints, cfg.past_frames)
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )

    flow, cond_module = build_flow(dataset, cfg)
    flow = flow.to(device)
    cond_module = cond_module.to(device)
    logger.info(
        "Flow parameters: %d, conditioning parameters: %d",
        sum(p.numel() for p in flow.parameters()),
        sum(p.numel() for p in cond_module.parameters()),
    )
    optimizer = torch.optim.AdamW(
        list(flow.parameters()) + list(cond_module.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    start_epoch = 0
    if resume_from is not None:
        start_epoch = load_flow_checkpoint(flow, cond_module, resume_from, optimizer)

    for epoch in range(start_epoch + 1, epochs + 1):
        mean_nll = train_epoch(flow, cond_module, loader, optimizer, cfg, device, epoch, wandb_logger)
        logger.info("Epoch %d/%d: nll %.4f", epoch, epochs, mean_nll)
        wandb_logger.log({"epoch/flow_nll": mean_nll})
    save_flow_checkpoint(flow, cond_module, optimizer, epochs, run_folder / "final_model.pt")
    return run_folder
