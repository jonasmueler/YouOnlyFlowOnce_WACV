"""Compare two feature files tensor by tensor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

EXACT_KEYS = ("features", "gt_pose")
NOISY_KEYS = ("mu_out", "cov")
LIST_ALIASES = {"subjects": "participants"}


def load(path: Path) -> dict[str, Any]:
    """Load a feature file and normalise legacy list names."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    for old, new in LIST_ALIASES.items():
        if old in data and new not in data:
            data[new] = data.pop(old)
    return data


def compare(ours: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Print a comparison and return whether the deterministic parts match exactly.

    Args:
        ours: Feature file content under test.
        reference: Reference feature file content.

    Returns:
        ``True`` when the exact keys and metadata lists are identical.
    """
    ok = True
    for key in EXACT_KEYS + NOISY_KEYS:
        a, b = ours.get(key), reference.get(key)
        if a is None or b is None:
            print(f"{key:10s} missing in one file")
            ok = ok and key not in EXACT_KEYS
            continue
        if a.shape != b.shape:
            print(f"{key:10s} shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
            ok = False
            continue
        diff = (a.double() - b.double()).abs()
        exact = torch.equal(a, b)
        print(f"{key:10s} shape={tuple(a.shape)} identical={exact} max|diff|={diff.max().item():.3e} mean|diff|={diff.mean().item():.3e}")
        if key in EXACT_KEYS and not exact:
            ok = False
    for key in ("participants", "scenes", "actions", "angles", "exercises"):
        a, b = ours.get(key), reference.get(key)
        if a is None and b is None:
            continue
        same = a is not None and b is not None and list(a) == list(b)
        print(f"{key:10s} list identical={same}")
        ok = ok and same
    print("RESULT:", "MATCH" if ok else "DIFFERENT")
    return ok


def main() -> None:
    """Parse arguments and compare."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ours", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if compare(load(args.ours), load(args.reference)) else 1)


if __name__ == "__main__":
    main()
