"""Command-line arguments shared by the entry scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from yofo.config import DATASETS, DETERMINISTIC, GLOBAL_SEED, DatasetConfig, Paths, default_data_root, resolve_dataset, resolve_split
from yofo.logging_utils import setup_logging
from yofo.seeding import set_seed


def add_common_arguments(parser: argparse.ArgumentParser, needs_data: bool = True) -> None:
    """Add the dataset, path, seed and logging arguments.

    Args:
        parser: Parser to extend.
        needs_data: Whether the script reads raw dataset files (adds ``--data_root``).
    """
    parser.add_argument(
        "--dataset", required=True, help=f"Dataset: {', '.join(DATASETS)} (legacy spellings mmFI, mmRadPose, MRI accepted)."
    )
    parser.add_argument(
        "--split", default=None, help="Split; MM-Fi: cross_subject, cross_scene, random, loeo_E01 to loeo_E04."
    )
    if needs_data:
        parser.add_argument(
            "--data_root",
            type=Path,
            default=None,
            help="Dataset folder (default: $YOFO_DATA_ROOT or $PATHRAW).",
        )
    parser.add_argument("--feature_root", type=Path, default=Path("features"), help="Folder of extracted features.")
    parser.add_argument("--run_root", type=Path, default=Path("runs"), help="Folder of training runs.")
    parser.add_argument("--results_root", type=Path, default=Path("results"), help="Folder of evaluation results.")
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoints"), help="Folder of downloaded checkpoints.")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Random seed.")
    parser.add_argument(
        "--no_deterministic",
        action="store_true",
        help="Disable deterministic CUDA algorithms (faster, not bit-reproducible).",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available).")
    parser.add_argument("--log_level", default="INFO", help="Logging level.")


def add_wandb_arguments(parser: argparse.ArgumentParser, default_project: str) -> None:
    """Add the optional Weights & Biases arguments.

    Args:
        parser: Parser to extend.
        default_project: Project name used when ``--wandb`` is given.
    """
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases (off by default).")
    parser.add_argument("--wandb_project", default=default_project, help="Weights & Biases project.")


def resolve_common(args: argparse.Namespace, needs_data: bool = True) -> tuple[DatasetConfig, str, Paths, torch.device]:
    """Set up logging and seeding and resolve the dataset, split, paths and device.

    Args:
        args: Parsed arguments from a parser extended by :func:`add_common_arguments`.
        needs_data: Whether a dataset root is required.

    Returns:
        Tuple of dataset configuration, split name, paths and device.
    """
    setup_logging(args.log_level)
    set_seed(args.seed, deterministic=DETERMINISTIC and not args.no_deterministic)
    dataset = resolve_dataset(args.dataset)
    split = resolve_split(dataset, args.split)
    data_root = getattr(args, "data_root", None) or default_data_root()
    if needs_data and data_root is None:
        raise SystemExit("Give --data_root or set YOFO_DATA_ROOT to the dataset folder.")
    paths = Paths(
        data_root=data_root if data_root is not None else Path("."),
        feature_root=args.feature_root,
        run_root=args.run_root,
        results_root=args.results_root,
        checkpoint_root=args.checkpoint_root,
    )
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return dataset, split, paths, device
