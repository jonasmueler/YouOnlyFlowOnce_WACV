"""Evaluate a trained model on the test part of a split and write the metrics to a JSON file."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from yofo.cli import add_common_arguments, resolve_common
from yofo.config import BackboneConfig, DiffPoseConfig, EvalConfig, FlowConfig, MmDiffConfig, checkpoint_path, feature_dir, to_dict
from yofo.evaluation.common import log_summary
from yofo.evaluation.diffpose import evaluate_diffpose
from yofo.evaluation.flow import evaluate_flow
from yofo.evaluation.gaussian import evaluate_gaussian
from yofo.evaluation.mmdiff import SECTIONS, evaluate_mmdiff
from yofo.runs import timestamp, write_json

logger = logging.getLogger(__name__)


def main() -> None:
    """Parse arguments, evaluate and write the results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--model", required=True, choices=("gaussian", "flow", "diffpose", "mmdiff"), help="Model to evaluate.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint (gaussian default: released checkpoint location).")
    parser.add_argument("--feature_dir", type=Path, default=None, help="Folder with the test feature file (flow, diffpose, mmdiff).")
    parser.add_argument("--n_samples", type=int, default=EvalConfig.n_samples, help="Hypotheses per frame.")
    parser.add_argument("--batch_size", type=int, default=EvalConfig.batch_size, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=EvalConfig.num_workers, help="Data-loader workers.")
    parser.add_argument("--no_save", action="store_true", help="Do not write the JSON file.")
    args = parser.parse_args()

    dataset, split, paths, device = resolve_common(args, needs_data=args.model == "gaussian")
    eval_cfg = EvalConfig(batch_size=args.batch_size, num_workers=args.num_workers, n_samples=args.n_samples)
    folder = args.feature_dir or feature_dir(paths, dataset, split)

    if args.model == "gaussian":
        checkpoint = args.checkpoint or checkpoint_path(paths, dataset, split)
        model_cfg = BackboneConfig()
        results = evaluate_gaussian(dataset, split, checkpoint, model_cfg, eval_cfg, paths, device)
    elif args.model == "flow":
        checkpoint = _require_checkpoint(args.checkpoint)
        model_cfg = FlowConfig()
        results = evaluate_flow(dataset, checkpoint, folder, model_cfg, eval_cfg, device)
    elif args.model == "diffpose":
        checkpoint = _require_checkpoint(args.checkpoint)
        model_cfg = DiffPoseConfig()
        results = evaluate_diffpose(dataset, checkpoint, folder, model_cfg, eval_cfg, device)
    else:
        checkpoint = _require_checkpoint(args.checkpoint)
        model_cfg = MmDiffConfig()
        results = evaluate_mmdiff(dataset, checkpoint, folder, model_cfg, eval_cfg, device, args.seed)

    if args.model == "mmdiff":
        for section in SECTIONS:
            logger.info("%s", section)
            log_summary(results[section], dataset)
    else:
        log_summary(results, dataset)
    if not args.no_save:
        record = {
            "metadata": {
                "model": args.model,
                "dataset": dataset.name,
                "split": split,
                "checkpoint": str(checkpoint),
                "feature_dir": str(folder) if args.model != "gaussian" else None,
                "timestamp": timestamp(),
                **to_dict(eval_cfg, model_cfg),
            },
            **results,
        }
        out = write_json(record, paths.results_root / dataset.name / f"eval_{args.model}_{split}_{timestamp()}.json")
        logger.info("Results written to %s", out)


def _require_checkpoint(checkpoint: Path | None) -> Path:
    """Return the checkpoint path or exit with a message."""
    if checkpoint is None or not checkpoint.is_file():
        raise SystemExit("--checkpoint must point to a final_model.pt of a training run.")
    return checkpoint


if __name__ == "__main__":
    main()
