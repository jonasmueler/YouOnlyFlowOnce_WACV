"""Extract the MG-Prior features of the training and test windows of a split."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from yofo.checkpoints import load_backbone_checkpoint
from yofo.cli import add_common_arguments, resolve_common
from yofo.config import BackboneConfig, DatasetConfig, checkpoint_path, feature_dir
from yofo.data.factory import build_dataset, collate_sequences, meta_fields
from yofo.data.gt import last_frame_pose
from yofo.metrics import mpjpe
from yofo.models.mg_prior import MGPrior
from yofo.seeding import worker_init_fn

logger = logging.getLogger(__name__)


def extract_part(
    model: MGPrior,
    dataset: DatasetConfig,
    split: str,
    part: str,
    data_root: Path,
    batch_size: int,
    n_samples: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run the MG-Prior over one part of a split and collect the feature file content.

    Args:
        model: MG-Prior in evaluation mode.
        dataset: Dataset configuration.
        split: Split name.
        part: ``"train"`` or ``"test"``.
        data_root: Dataset folder.
        batch_size: Batch size.
        n_samples: Latent samples per frame for the mean and covariance.
        num_workers: Data-loader workers.
        device: Device.

    Returns:
        Dictionary with the tensors, metadata lists and a ``metadata`` record.
    """
    data = build_dataset(dataset, split, part, data_root, return_meta=True)
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sequences,
        worker_init_fn=worker_init_fn,
    )
    features, mu_outs, covs, gts = [], [], [], []
    lists: dict[str, list[str]] = {}
    with torch.no_grad():
        for radar, gt, metas in tqdm(loader, desc=f"{part} features"):
            radar = radar.to(device).float()
            gt_flat = last_frame_pose(gt.to(device).float(), dataset.gt_scale_to_cm)
            _, _, _, mu_out, cov = model(radar, n_samples=n_samples)
            batch = gt_flat.size(0)
            features.append(model.backbone_features.cpu())
            mu_outs.append(mu_out.view(batch, dataset.num_keypoints, 3).cpu())
            covs.append(cov.cpu())
            gts.append(gt_flat.view(batch, dataset.num_keypoints, 3).cpu())
            for meta in metas:
                for key, value in meta_fields(dataset, meta).items():
                    lists.setdefault(key + "s", []).append(value)

    mu_out_tensor = torch.cat(mu_outs)
    gt_tensor = torch.cat(gts)
    logger.info("%s: %d windows, backbone MPJPE %.3f cm", part, gt_tensor.size(0), mpjpe(mu_out_tensor, gt_tensor).item())
    return {
        "features": torch.cat(features),
        "mu_out": mu_out_tensor,
        "cov": torch.cat(covs),
        "gt_pose": gt_tensor,
        **lists,
        "metadata": {
            "dataset": dataset.name,
            "split": split,
            "part": part,
            "num_keypoints": dataset.num_keypoints,
            "num_samples": int(gt_tensor.size(0)),
            "feat_dim": int(features[0].size(1)),
            "n_samples": n_samples,
            "center_gt_to_root": dataset.center_gt_to_root,
            "participants": sorted(set(lists.get("participants", []))),
        },
    }


def main() -> None:
    """Parse arguments and write the two feature files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=None, help="MG-Prior checkpoint (default: released checkpoint location).")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output folder (default: <feature_root>/<dataset>/<split>).")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default per dataset).")
    parser.add_argument("--n_samples", type=int, default=BackboneConfig.n_samples_extract, help="Latent samples per frame.")
    parser.add_argument("--num_workers", type=int, default=4, help="Data-loader workers.")
    args = parser.parse_args()

    dataset, split, paths, device = resolve_common(args)
    backbone_cfg = BackboneConfig()
    checkpoint = args.checkpoint or checkpoint_path(paths, dataset, split)
    output_dir = args.output_dir or feature_dir(paths, dataset, split)
    batch_size = args.batch_size or dataset.extraction_batch_size
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint} (run download_checkpoints.py or train_backbone.py).")

    model = MGPrior(dataset, backbone_cfg).to(device)
    load_backbone_checkpoint(model, checkpoint)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for part in ("train", "test"):
        content = extract_part(model, dataset, split, part, paths.data_root, batch_size, args.n_samples, args.num_workers, device)
        content["metadata"]["checkpoint"] = str(checkpoint)
        out = output_dir / f"{part}_features.pt"
        torch.save(content, out)
        logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
