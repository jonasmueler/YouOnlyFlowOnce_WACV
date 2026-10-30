"""Keep only the MG-Prior weights and the epoch of a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

KEEP_PREFIXES = ("backbone.", "head.")


def strip(src: Path, dst: Path) -> tuple[str, list[str]]:
    """Write a weights-only copy of a checkpoint.

    Args:
        src: Checkpoint with ``model`` (and possibly ``optimizer``, ``epoch``).
        dst: Output file.

    Returns:
        Hex SHA256 of the written file and the names of the dropped model entries.
    """
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model = {k: v for k, v in checkpoint["model"].items() if k.startswith(KEEP_PREFIXES)}
    dropped = [k for k in checkpoint["model"] if k not in model]
    stripped = {"model": model}
    if "epoch" in checkpoint:
        stripped["epoch"] = checkpoint["epoch"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, dst)
    return hashlib.sha256(dst.read_bytes()).hexdigest(), dropped


def main() -> None:
    """Parse arguments and strip one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    args = parser.parse_args()
    digest, dropped = strip(args.src, args.dst)
    print(f"{digest}  {args.dst}  {args.dst.stat().st_size} bytes  dropped {len(dropped)} entries")


if __name__ == "__main__":
    main()
