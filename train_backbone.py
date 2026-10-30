"""Train the MG-Prior (phase 1) on raw radar windows."""

from __future__ import annotations

import argparse
import dataclasses
import logging

from yofo.cli import add_common_arguments, add_wandb_arguments, resolve_common
from yofo.config import DETERMINISTIC, BackboneConfig, to_dict
from yofo.training.backbone import train_backbone
from yofo.training.wandb_logging import WandbLogger

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments and train."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    add_wandb_arguments(parser, "yofo_mg_prior")
    parser.add_argument("--epochs", type=int, default=None, help="Epochs (default per dataset).")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Train the MG-Prior as a standalone model, evaluated on its own.",
    )
    parser.add_argument("--lr", type=float, default=BackboneConfig.lr, help="Learning rate.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (default 32).")
    parser.add_argument("--beta", type=float, default=BackboneConfig.beta, help="KL weight.")
    parser.add_argument("--gamma", type=float, default=BackboneConfig.gamma, help="Variance penalty weight.")
    parser.add_argument("--num_workers", type=int, default=BackboneConfig.num_workers, help="Data-loader workers.")
    args = parser.parse_args()

    dataset, split, paths, device = resolve_common(args)
    cfg = dataclasses.replace(
        BackboneConfig(),
        epochs=args.epochs,
        standalone=args.standalone,
        lr=args.lr,
        beta=args.beta,
        gamma=args.gamma,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
    wandb_logger = WandbLogger(args.wandb, args.wandb_project, {"split": split, **to_dict(dataset, cfg)})
    run_folder = train_backbone(
        dataset, split, cfg, paths, device, args.seed, DETERMINISTIC and not args.no_deterministic, wandb_logger
    )
    wandb_logger.finish()
    logger.info("Done. Checkpoint: %s", run_folder / "final_model.pt")


if __name__ == "__main__":
    main()
