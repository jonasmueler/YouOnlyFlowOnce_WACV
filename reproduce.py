"""Run checkpoint download, feature extraction, flow training and evaluation for one split."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from yofo.cli import add_common_arguments, resolve_common
from yofo.config import feature_dir, run_dir

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent

COLUMNS = ("MPJPE", "PA-MPJPE", "@50", "@90", "@95", "ECE", "Sharp.")


def run(cmd: list[str]) -> None:
    """Run a command in the repository folder and stop on failure."""
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def latest_result(folder: Path, prefix: str) -> dict:
    """Load the newest evaluation JSON with a given prefix."""
    files = sorted(folder.glob(f"{prefix}*.json"))
    if not files:
        raise FileNotFoundError(f"No {prefix}*.json in {folder}")
    with open(files[-1], encoding="utf-8") as f:
        return json.load(f)["overall"]


def row(metrics: dict) -> tuple[float, ...]:
    """Order the overall metrics as table columns."""
    return (
        metrics["mpjpe"],
        metrics["pa_mpjpe"],
        metrics["coverage_50"],
        metrics["coverage_90"],
        metrics["coverage_95"],
        metrics["ece"],
        metrics["sharpness"],
    )


def print_table(title: str, values: tuple[float, ...]) -> None:
    """Print the overall metrics of one evaluation as a one-row table."""
    print(f"\n{title}")
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    print("| " + " | ".join(f"{v:.3f}" for v in values) + " |")


def main() -> None:
    """Parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--skip_download", action="store_true", help="Checkpoint is already in place.")
    parser.add_argument("--skip_extract", action="store_true", help="Features are already extracted.")
    args = parser.parse_args()
    dataset, split, paths, _ = resolve_common(args)

    py = sys.executable
    common = ["--dataset", dataset.key, "--split", split, "--seed", str(args.seed)]
    if args.no_deterministic:
        common.append("--no_deterministic")
    data = ["--data_root", str(paths.data_root)]
    roots = [
        "--feature_root", str(paths.feature_root),
        "--run_root", str(paths.run_root),
        "--results_root", str(paths.results_root),
        "--checkpoint_root", str(paths.checkpoint_root),
    ]

    if not args.skip_download:
        run([py, "download_checkpoints.py", "--dataset", dataset.key, "--dest", str(paths.checkpoint_root)])
    if not args.skip_extract:
        run([py, "extract_features.py", *common, *data, *roots])
    run([py, "train_flow.py", *common, *roots])
    flow_runs = sorted((run_dir(paths, dataset, split, "norm_flow_graph")).glob("run_*"))
    checkpoint = flow_runs[-1] / "final_model.pt"
    run([py, "evaluate.py", "--model", "flow", *common, *roots, "--checkpoint", str(checkpoint),
         "--feature_dir", str(feature_dir(paths, dataset, split))])
    results_folder = paths.results_root / dataset.name
    print_table(f"MG-Prior + NFPG on {dataset.name} ({split})", row(latest_result(results_folder, f"eval_flow_{split}")))


if __name__ == "__main__":
    main()
