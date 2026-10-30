"""Train the MG-Prior + DiffPose baseline on extracted features."""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from yofo.cli import add_common_arguments, add_wandb_arguments, resolve_common
from yofo.config import DiffPoseConfig, feature_dir, to_dict
from yofo.training.diffpose import train_diffpose
from yofo.training.wandb_logging import WandbLogger

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and train."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser, needs_data=False)
    add_wandb_arguments(parser, "yofo_diffpose")
    parser.add_argument("--feature_dir", type=Path, default=None, help="Folder with train_features.pt (default: <feature_root>/<dataset>/<split>).")
    parser.add_argument("--epochs", type=int, default=DiffPoseConfig.epochs, help="Epochs.")
    parser.add_argument("--batch_size", type=int, default=DiffPoseConfig.batch_size, help="Batch size.")
    parser.add_argument("--lr", type=float, default=DiffPoseConfig.lr, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=DiffPoseConfig.num_workers, help="Data-loader workers.")
    args = parser.parse_args()

    dataset, split, paths, device = resolve_common(args, needs_data=False)
    cfg = dataclasses.replace(DiffPoseConfig(), epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers)
    folder = args.feature_dir or feature_dir(paths, dataset, split)
    if not (folder / "train_features.pt").is_file():
        raise SystemExit(f"{folder}/train_features.pt not found (run extract_features.py first).")
    wandb_logger = WandbLogger(args.wandb, args.wandb_project, {"split": split, **to_dict(dataset, cfg)})
    run_folder = train_diffpose(dataset, split, cfg, paths, folder, device, args.seed, wandb_logger)
    wandb_logger.finish()
    logger.info("Done. Checkpoint: %s", run_folder / "final_model.pt")


if __name__ == "__main__":
    main()
