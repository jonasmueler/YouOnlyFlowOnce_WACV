"""Download the released MG-Prior checkpoints from Google Drive and verify their checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

try:
    import gdown
except ImportError:  # gdown is only needed for downloading
    gdown = None

from yofo.logging_utils import setup_logging

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent
MANIFEST = REPO / "checkpoints" / "manifest.json"


def sha256(path: Path) -> str:
    """SHA256 of a file.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> list[dict[str, Any]]:
    """Read the checkpoint manifest."""
    with open(MANIFEST, encoding="utf-8") as f:
        return json.load(f)["files"]


def fetch(entries: list[dict[str, Any]], dest: Path, verify_only: bool) -> int:
    """Download and verify the listed files.

    Args:
        entries: Manifest entries with ``path``, ``id``, ``sha256`` and ``size``.
        dest: Local checkpoint folder.
        verify_only: Only check files that exist, never download.

    Returns:
        Number of files that are missing or failed verification.
    """
    failed = 0
    for entry in entries:
        target = dest / entry["path"]
        if target.is_file() and sha256(target) == entry["sha256"]:
            logger.info("ok       %s", entry["path"])
            continue
        if verify_only:
            logger.error("missing or corrupt %s", entry["path"])
            failed += 1
            continue
        if gdown is None:
            sys.exit("gdown is missing: pip install gdown")
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("download %s (%d MB)", entry["path"], entry["size"] // 1_000_000)
        gdown.download(id=entry["id"], output=str(target), quiet=False)
        if not target.is_file() or sha256(target) != entry["sha256"]:
            logger.error("checksum mismatch for %s", entry["path"])
            failed += 1
        else:
            logger.info("verified %s", entry["path"])
    return failed


def main() -> None:
    """Parse arguments and fetch the checkpoints."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=None, help="Only this dataset (mmfi, mmradpose, mri).")
    parser.add_argument("--dest", type=Path, default=REPO / "checkpoints", help="Checkpoint folder.")
    parser.add_argument("--verify_only", action="store_true", help="Verify without downloading.")
    args = parser.parse_args()
    setup_logging()

    entries = load_manifest()
    if args.dataset:
        entries = [e for e in entries if e["dataset"] == args.dataset.lower()]
        if not entries:
            sys.exit(f"No checkpoints listed for dataset '{args.dataset}'.")
    failed = fetch(entries, args.dest, args.verify_only)
    if failed:
        sys.exit(f"{failed} file(s) missing or corrupt.")
    logger.info("All %d checkpoint(s) present and verified.", len(entries))


if __name__ == "__main__":
    main()
