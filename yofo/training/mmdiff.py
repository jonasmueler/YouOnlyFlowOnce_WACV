"""Training of the mmDiff baseline: PointTransformer backbone, feature extraction, GCNdiff.

The three phases follow the authors' code (see ``docs/BASELINES.md``): the backbone is
trained with a mean squared error on the root-centred pose of the last frame in
metres, its coarse poses and per-joint features are dumped in dataset order, and the
denoiser is trained on those dumps with antithetic timesteps, the noise-prediction
loss summed over joints and the limb-length loss. Both phases keep the weights of
the final epoch and write a checkpoint after every epoch so that a run can resume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from yofo.baselines.mmdiff import MmDiffFeatureDataset, build_backbone, build_gcndiff, collate_mmdiff, limb_lengths, limb_pairs
from yofo.config import DatasetConfig, MmDiffConfig, Paths, run_dir, to_dict
from yofo.data.factory import build_dataset, collate_sequences, meta_fields
from yofo.data.gt import last_frame_pose
from yofo.metrics import mpjpe
from yofo.runs import create_run_folder, save_hyperparameters
from yofo.seeding import set_seed, worker_init_fn
from yofo.third_party.mmdiff import PointTransformerReg, alpha_bar, linear_betas, sample_timesteps_antithetic
from yofo.training.wandb_logging import WandbLogger

logger = logging.getLogger(__name__)

BACKBONE_MODEL_NAME = "mmdiff_backbone"
DIFFUSION_MODEL_NAME = "mmdiff"
CM_PER_M = 100.0


def _limit(dataset: Dataset, limit_windows: int | None) -> Dataset:
    """Restrict a dataset to its first windows (smoke tests only)."""
    if limit_windows is None or limit_windows >= len(dataset):
        return dataset
    logger.warning("Using only the first %d of %d windows (smoke test)", limit_windows, len(dataset))
    return Subset(dataset, list(range(limit_windows)))


def _epoch_checkpoint_path(run_folder: Path, epoch: int) -> Path:
    """Path of the checkpoint written after an epoch (zero-based)."""
    return run_folder / f"epoch_{epoch:03d}.pt"


def _save_epoch_checkpoint(run_folder: Path, epoch: int, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """Write the resumable checkpoint of an epoch, including the random-number states."""
    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "rng_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    torch.save(state, _epoch_checkpoint_path(run_folder, epoch))


def _latest_epoch_checkpoint(run_folder: Path) -> Path | None:
    """Return the newest ``epoch_NNN.pt`` of a run folder, if any."""
    files = sorted(run_folder.glob("epoch_*.pt"))
    return files[-1] if files else None


def _resume(run_folder: Path, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> int:
    """Load the newest epoch checkpoint of a run folder.

    Args:
        run_folder: Folder of the interrupted run.
        model: Model to restore.
        optimizer: Optimizer to restore.
        device: Device.

    Returns:
        The epoch to continue with (one after the restored epoch), or 0 if there is no checkpoint.
    """
    path = _latest_epoch_checkpoint(run_folder)
    if path is None:
        logger.info("No epoch checkpoint in %s, starting from scratch", run_folder)
        return 0
    # Loaded onto the CPU: the random-number states must stay CPU byte tensors, and the
    # model and the optimizer move their tensors to the parameters' device themselves.
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rng_cpu"])
    if "rng_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["rng_cuda"])
    logger.info("Resumed from %s (epoch %d)", path, state["epoch"])
    return int(state["epoch"]) + 1


def _save_final(run_folder: Path, model: nn.Module, epoch: int) -> Path:
    """Write the final weights as ``final_model.pt`` (the final epoch; no selection)."""
    path = run_folder / "final_model.pt"
    torch.save({"model": model.state_dict(), "epoch": epoch}, path)
    logger.info("Final weights saved to %s", path)
    return path


def train_backbone(
    dataset: DatasetConfig,
    split: str,
    cfg: MmDiffConfig,
    paths: Paths,
    device: torch.device,
    seed: int,
    deterministic: bool,
    wandb_logger: WandbLogger,
    resume_from: Path | None = None,
    limit_windows: int | None = None,
) -> Path:
    """Phase 1: train the PointTransformer backbone and keep the final epoch.

    Args:
        dataset: Dataset configuration.
        split: Split name.
        cfg: mmDiff hyperparameters.
        paths: Root folders.
        device: Device.
        seed: Random seed.
        deterministic: Deterministic CUDA algorithms.
        wandb_logger: Optional logger.
        resume_from: Run folder of an interrupted run to continue.
        limit_windows: Use only the first windows (smoke tests only).

    Returns:
        The run folder holding ``final_model.pt`` and ``hyperparameters.json``.
    """
    model = build_backbone(dataset, cfg).to(device)
    logger.info("PointTransformer parameters: %d", sum(p.numel() for p in model.parameters()))

    train_set = _limit(build_dataset(dataset, split, "train", paths.data_root), limit_windows)
    loader = DataLoader(
        train_set,
        batch_size=cfg.backbone_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_sequences,
        worker_init_fn=worker_init_fn,
    )
    logger.info("Training windows: %d", len(train_set))

    set_seed(seed, deterministic)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.backbone_lr, weight_decay=cfg.backbone_weight_decay)
    criterion = nn.MSELoss()

    if resume_from is not None:
        run_folder = resume_from
        start_epoch = _resume(run_folder, model, optimizer, device)
    else:
        run_folder = create_run_folder(run_dir(paths, dataset, split, BACKBONE_MODEL_NAME))
        start_epoch = 0
        save_hyperparameters(
            {"model": BACKBONE_MODEL_NAME, "split": split, "epochs": cfg.backbone_epochs, "seed": seed, "unit": "m", **to_dict(dataset, cfg)},
            run_folder,
        )

    scale = dataset.pose_scale_to_m
    for epoch in range(start_epoch, cfg.backbone_epochs):
        model.train()
        total = 0.0
        count = 0
        for radar, gt in tqdm(loader, desc=f"backbone epoch {epoch + 1}/{cfg.backbone_epochs}"):
            radar = radar.to(device).float()
            target = last_frame_pose(gt.to(device).float(), scale).view(-1, dataset.num_keypoints, 3)
            optimizer.zero_grad()
            pts, _ = model(radar)
            loss = criterion(pts, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total += loss.item()
            count += 1
            wandb_logger.log({"backbone/loss": loss.item()})
        logger.info("Backbone epoch %d/%d: MSE %.6f m^2", epoch + 1, cfg.backbone_epochs, total / max(count, 1))
        _save_epoch_checkpoint(run_folder, epoch, model, optimizer)
    _save_final(run_folder, model, cfg.backbone_epochs - 1)
    return run_folder


def load_backbone(dataset: DatasetConfig, cfg: MmDiffConfig, checkpoint: Path, device: torch.device) -> PointTransformerReg:
    """Build the backbone and load the weights of a run.

    Args:
        dataset: Dataset configuration.
        cfg: mmDiff hyperparameters.
        checkpoint: ``final_model.pt`` of a backbone run.
        device: Device.

    Returns:
        The backbone in evaluation mode.
    """
    model = build_backbone(dataset, cfg).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    logger.info("Backbone loaded from %s (epoch %s)", checkpoint, state.get("epoch"))
    return model


def extract_features(
    dataset: DatasetConfig,
    split: str,
    backbone_checkpoint: Path,
    cfg: MmDiffConfig,
    paths: Paths,
    device: torch.device,
    output_dir: Path,
    limit_windows: int | None = None,
) -> Path:
    """Phase 2: dump coarse poses and per-joint features of the train and test windows.

    The windows are processed in dataset order (no shuffling) because the temporal
    history of the denoiser is built from the file order. Poses are stored in metres and
    centred on joint 0, as the original's data generator does.

    Args:
        dataset: Dataset configuration.
        split: Split name.
        backbone_checkpoint: ``final_model.pt`` of a backbone run.
        cfg: mmDiff hyperparameters.
        paths: Root folders.
        device: Device.
        output_dir: Folder for ``mmdiff_train_features.pt`` and ``mmdiff_test_features.pt``.
        limit_windows: Use only the first windows (smoke tests only).

    Returns:
        The output folder.
    """
    model = load_backbone(dataset, cfg, backbone_checkpoint, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dataset.pose_scale_to_m
    for part in ("train", "test"):
        data = _limit(build_dataset(dataset, split, part, paths.data_root, return_meta=True), limit_windows)
        loader = DataLoader(
            data,
            batch_size=cfg.backbone_batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_sequences,
            worker_init_fn=worker_init_fn,
        )
        features, coarse, gts = [], [], []
        lists: dict[str, list[str]] = {}
        with torch.no_grad():
            for radar, gt, metas in tqdm(loader, desc=f"mmdiff {part} features"):
                radar = radar.to(device).float()
                target = last_frame_pose(gt.to(device).float(), scale).view(-1, dataset.num_keypoints, 3)
                pts, feat = model(radar)
                pts = pts - pts[:, :1, :]
                features.append(feat.cpu())
                coarse.append(pts.cpu())
                gts.append(target.cpu())
                for meta in metas:
                    for key, value in meta_fields(dataset, meta).items():
                        lists.setdefault(key + "s", []).append(value)
        coarse_tensor = torch.cat(coarse)
        gt_tensor = torch.cat(gts)
        if coarse_tensor.size(0) != len(data):
            raise RuntimeError(f"Extracted {coarse_tensor.size(0)} windows but the dataset has {len(data)}")
        logger.info(
            "%s: %d windows, PointTransformer MPJPE %.3f cm",
            part,
            gt_tensor.size(0),
            mpjpe(coarse_tensor * CM_PER_M, gt_tensor * CM_PER_M).item(),
        )
        content = {
            "features": torch.cat(features),
            "coarse_pose": coarse_tensor,
            "gt_pose": gt_tensor,
            **lists,
            "metadata": {
                "dataset": dataset.name,
                "split": split,
                "part": part,
                "model": BACKBONE_MODEL_NAME,
                "num_keypoints": dataset.num_keypoints,
                "num_samples": int(gt_tensor.size(0)),
                "feature_dim": int(features[0].size(-1)),
                "unit": "m",
                "centered_on_joint_0": True,
                "order": "dataset",
                "checkpoint": str(backbone_checkpoint),
                "participants": sorted(set(lists.get("participants", []))),
            },
        }
        out = output_dir / f"mmdiff_{part}_features.pt"
        torch.save(content, out)
        logger.info("Saved %s", out)
    return output_dir


def learning_rate_at(cfg: MmDiffConfig, epoch: int) -> float:
    """Learning rate of the denoiser after the batches of ``epoch`` (the original's ``lr_decay``).

    The original multiplies the initial learning rate by ``lr_gamma ** (epoch / lr_decay_epochs)``
    at every epoch that is a multiple of ``lr_decay_epochs`` (including epoch 0).

    Args:
        cfg: mmDiff hyperparameters.
        epoch: Zero-based epoch.

    Returns:
        The learning rate in force from the next epoch on.
    """
    last_decay_epoch = (epoch // cfg.lr_decay_epochs) * cfg.lr_decay_epochs
    return cfg.lr * cfg.lr_gamma ** (last_decay_epoch / cfg.lr_decay_epochs)


def diffusion_losses(
    model: nn.Module,
    betas: torch.Tensor,
    pairs: torch.Tensor,
    cfg: MmDiffConfig,
    features: torch.Tensor,
    gt_pose: torch.Tensor,
    history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Noise-prediction and limb-length losses of one batch, as in the original.

    Args:
        model: GCNdiff.
        betas: Betas ``(T,)``.
        pairs: Limb joint pairs ``(L, 2)``.
        cfg: mmDiff hyperparameters.
        features: Per-joint backbone features ``(B, K, F)``.
        gt_pose: Root-centred ground truth ``(B, K, 3)`` in metres.
        history: Past coarse poses ``(B, K, 3 * past_frames)``.

    Returns:
        Tuple of total loss, noise loss and limb loss.
    """
    n = gt_pose.size(0)
    t = sample_timesteps_antithetic(n, cfg.timesteps, gt_pose.device)
    a = alpha_bar(betas, t)
    noise = torch.randn_like(gt_pose)
    x_t = gt_pose * a.sqrt() + noise * (1.0 - a).sqrt()
    eps, limb_pred = model(torch.cat([history, x_t], dim=-1), t.float(), features)
    loss_diff = (noise - eps).square().sum(dim=(1, 2)).mean(dim=0)
    limb_loss = (limb_pred - limb_lengths(gt_pose, pairs)).abs().sum(dim=-1).mean(dim=0)
    return loss_diff + limb_loss * cfg.limb_loss_weight, loss_diff, limb_loss


def train_diffusion(
    dataset: DatasetConfig,
    split: str,
    feature_folder: Path,
    cfg: MmDiffConfig,
    paths: Paths,
    device: torch.device,
    seed: int,
    deterministic: bool,
    wandb_logger: WandbLogger,
    resume_from: Path | None = None,
) -> Path:
    """Phase 3: train the GCNdiff denoiser on the dumped features and keep the final epoch.

    Args:
        dataset: Dataset configuration.
        split: Split name.
        feature_folder: Folder with ``mmdiff_train_features.pt``.
        cfg: mmDiff hyperparameters.
        paths: Root folders.
        device: Device.
        seed: Random seed.
        deterministic: Deterministic CUDA algorithms.
        wandb_logger: Optional logger.
        resume_from: Run folder of an interrupted run to continue.

    Returns:
        The run folder holding ``final_model.pt`` and ``hyperparameters.json``.
    """
    train_set = MmDiffFeatureDataset(feature_folder / "mmdiff_train_features.pt", dataset.num_keypoints, cfg.past_frames)
    loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_mmdiff,
        worker_init_fn=worker_init_fn,
    )
    model = build_gcndiff(dataset, cfg).to(device)
    logger.info("GCNdiff parameters: %d", sum(p.numel() for p in model.parameters()))
    betas = linear_betas(cfg.timesteps, cfg.beta_start, cfg.beta_end).to(device)
    pairs = limb_pairs(dataset).to(device)

    set_seed(seed, deterministic)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    if resume_from is not None:
        run_folder = resume_from
        start_epoch = _resume(run_folder, model, optimizer, device)
    else:
        run_folder = create_run_folder(run_dir(paths, dataset, split, DIFFUSION_MODEL_NAME))
        start_epoch = 0
        save_hyperparameters(
            {
                "model": DIFFUSION_MODEL_NAME,
                "split": split,
                "epochs": cfg.epochs,
                "seed": seed,
                "feature_dir": str(feature_folder),
                "num_limbs": int(pairs.size(0)),
                "unit": "m",
                **to_dict(dataset, cfg),
            },
            run_folder,
        )
    if start_epoch > 0:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(cfg, start_epoch - 1)

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        sums = torch.zeros(3)
        count = 0
        pbar = tqdm(loader, desc=f"diffusion epoch {epoch + 1}/{cfg.epochs}")
        for features, _, gt_pose, history in pbar:
            features = features.to(device).float()
            gt_pose = gt_pose.to(device).float()
            history = history.to(device).float()
            optimizer.zero_grad()
            loss, loss_diff, limb_loss = diffusion_losses(model, betas, pairs, cfg, features, gt_pose, history)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            sums += torch.tensor([loss.item(), loss_diff.item(), limb_loss.item()])
            count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            wandb_logger.log({"diffusion/loss": loss.item(), "diffusion/noise": loss_diff.item(), "diffusion/limb": limb_loss.item()})
        lr = learning_rate_at(cfg, epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        means = (sums / max(count, 1)).tolist()
        logger.info(
            "Diffusion epoch %d/%d: loss %.4f (noise %.4f, limb %.4f), lr now %.2e", epoch + 1, cfg.epochs, means[0], means[1], means[2], lr
        )
        wandb_logger.log({"epoch/loss": means[0], "epoch/lr": lr})
        _save_epoch_checkpoint(run_folder, epoch, model, optimizer)
    _save_final(run_folder, model, cfg.epochs - 1)
    return run_folder
