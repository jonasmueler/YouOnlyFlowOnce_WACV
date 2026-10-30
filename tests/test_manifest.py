"""The checkpoint manifest lists one checkpoint per setting with id, size and SHA256."""

import json
import re
from pathlib import Path

from yofo.config import DATASETS

MANIFEST = Path(__file__).resolve().parents[1] / "checkpoints" / "manifest.json"


def test_manifest_covers_every_setting() -> None:
    with open(MANIFEST, encoding="utf-8") as f:
        entries = json.load(f)["files"]
    # Released MG-Prior checkpoints exist for the main-table settings; the
    # leave-one-environment-out folds of Table 2 are trained from scratch.
    expected = {(d.key, split) for d in DATASETS.values() for split in d.splits if not split.startswith("loeo_")}
    assert {(e["dataset"], e["split"]) for e in entries} == expected
    for entry in entries:
        assert entry["path"] == f"{entry['dataset']}/{entry['split']}/gaussian/backbone.pt"
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        assert re.fullmatch(r"[A-Za-z0-9_-]{20,}", entry["id"])
        assert entry["size"] > 40_000_000
