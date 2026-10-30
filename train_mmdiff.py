"""Train the mmDiff baseline: PointTransformer backbone, feature extraction and denoiser."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from yofo.cli import add_common_arguments, add_wandb_arguments, resolve_common
from yofo.config import DETERMINISTIC, MmDiffConfig, feature_dir, run_dir, to_dict
from yofo.training.mmdiff import BACKBONE_MODEL_NAME, extract_features, train_backbone, train_diffusion
from yofo.training.wandb_logging import WandbLogger

logger = logging.getLogger(__name__)

PHASES = ("backbone", "extract", "diffusion", "all")


def latest_backbone_checkpoint(run_root_folder: Path) -> Path:
    """Return ``final_model.pt`` of the newest backbone run under a folder."""
    runs = sorted(run_root_folder.glob("run_*/final_model.pt"))
    if not runs:
        raise SystemExit(f"No backbone run with final_model.pt under {run_root_folder}; run --phase backbone first.")
    return runs[-1]


def main() -> None:
    """Parse arguments and run the requested phases."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    add_wandb_arguments(parser, "yofo_mmdiff")
    parser.add_argument("--phase", choices=PHASES, default="all", help="Phase to run.")
    parser.add_argument("--backbone_checkpoint", type=Path, default=None, help="Backbone final_model.pt for extract (default: newest run).")
    parser.add_argument("--feature_dir", type=Path, default=None, help="Folder of the mmDiff feature files (default: <feature_root>/<dataset>/<split>).")
    parser.add_argument("--resume_from", type=Path, default=None, help="Run folder of an interrupted backbone or diffusion run.")
    parser.add_argument("--backbone_epochs", type=int, default=MmDiffConfig.backbone_epochs, help="Phase-1 epochs.")
    parser.add_argument("--backbone_batch_size", type=int, default=MmDiffConfig.backbone_batch_size, help="Phase-1 batch size.")
    parser.add_argument("--epochs", type=int, default=MmDiffConfig.epochs, help="Phase-2 epochs.")
    parser.add_argument("--batch_size", type=int, default=MmDiffConfig.batch_size, help="Phase-2 batch size.")
    parser.add_argument("--num_workers", type=int, default=MmDiffConfig.num_workers, help="Data-loader workers.")
    parser.add_argument("--limit_windows", type=int, default=None, help="Smoke tests only: use the first N windows of each part.")
    args = parser.parse_args()

    needs_data = args.phase in ("backbone", "extract", "all")
    dataset, split, paths, device = resolve_common(args, needs_data=needs_data)
    deterministic = DETERMINISTIC and not args.no_deterministic
    cfg = dataclasses.replace(
        MmDiffConfig(),
        backbone_epochs=args.backbone_epochs,
        backbone_batch_size=args.backbone_batch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    folder = args.feature_dir or feature_dir(paths, dataset, split)
    wandb_logger = WandbLogger(args.wandb, args.wandb_project, {"split": split, "phase": args.phase, **to_dict(dataset, cfg)})

    backbone_checkpoint = args.backbone_checkpoint
    if args.phase in ("backbone", "all"):
        resume = args.resume_from if args.phase == "backbone" else None
        run_folder = train_backbone(dataset, split, cfg, paths, device, args.seed, deterministic, wandb_logger, resume, args.limit_windows)
        backbone_checkpoint = run_folder / "final_model.pt"
    if args.phase in ("extract", "all"):
        if backbone_checkpoint is None:
            backbone_checkpoint = latest_backbone_checkpoint(run_dir(paths, dataset, split, BACKBONE_MODEL_NAME))
        extract_features(dataset, split, backbone_checkpoint, cfg, paths, device, folder, args.limit_windows)
    if args.phase in ("diffusion", "all"):
        if not (folder / "mmdiff_train_features.pt").is_file():
            raise SystemExit(f"{folder}/mmdiff_train_features.pt not found; run --phase extract first.")
        resume = args.resume_from if args.phase == "diffusion" else None
        run_folder = train_diffusion(dataset, split, folder, cfg, paths, device, args.seed, deterministic, wandb_logger, resume)
        logger.info("Done. Checkpoint: %s", run_folder / "final_model.pt")
    wandb_logger.finish()


if __name__ == "__main__":
    main()
