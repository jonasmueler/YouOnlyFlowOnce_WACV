"""Print the mmDiff rows of evaluation JSON files as LaTeX table cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROW_LABELS = {
    "pointtransformer": "PointTransformer~\\cite{zhao2021point}",
    "mmdiff_eta0": "mmDiff ($\\eta{=}0$)~\\cite{fan2024diffusion}",
    "mmdiff_eta1": "mmDiff ($\\eta{=}1$)~\\cite{fan2024diffusion}",
}


def cells(section: dict[str, Any], mmradpose: bool) -> list[str]:
    """Format the cells of one row.

    Args:
        section: Result section with ``overall`` and the per-group dictionaries.
        mmradpose: Insert the per-angle MPJPE columns.

    Returns:
        Formatted cells.
    """
    overall = section["overall"]
    out = [f"{overall['mpjpe']:.3f}", f"{overall['pa_mpjpe']:.3f}"]
    if mmradpose:
        angles = section.get("per_angle", {})
        out += [f"{angles[angle]['mpjpe']:.3f}" for angle in sorted(angles)]
    if "ece" in overall:
        out += [
            f"{overall['coverage_50']:.3f}",
            f"{overall['coverage_90']:.3f}",
            f"{overall['coverage_95']:.3f}",
            f"{overall['ece']:.3f}",
            f"{overall['sharpness']:.3f}",
        ]
    else:
        out += ["--"] * 5
    return out


def main() -> None:
    """Print the rows of every file."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="eval_mmdiff_*.json files.")
    args = parser.parse_args()
    for path in args.files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data["metadata"]
        print(f"% {path} ({meta['dataset']} {meta['split']}, {data['sampler']['n_hypotheses']} hypotheses)")
        for key, label in ROW_LABELS.items():
            print(f"{label} & " + " & ".join(cells(data[key], meta["dataset"] == "mmRadPose")) + " \\\\")
        print()


if __name__ == "__main__":
    main()
