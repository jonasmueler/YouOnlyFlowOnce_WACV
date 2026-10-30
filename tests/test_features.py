"""Feature file round trip and the temporal history rule."""

from pathlib import Path

import torch

from yofo.data.features import PrecomputedFeatureDataset, collate_with_meta


def _write_file(path: Path, n: int = 10, k: int = 17) -> None:
    torch.save(
        {
            "features": torch.randn(n, 512),
            "mu_out": torch.arange(n, dtype=torch.float).view(n, 1, 1).expand(n, k, 3).clone(),
            "cov": torch.eye(3 * k).expand(n, 3 * k, 3 * k).clone(),
            "gt_pose": torch.randn(n, k, 3),
            "participants": [f"s{i % 2}" for i in range(n)],
            "scenes": ["E01"] * n,
            "actions": ["A01"] * n,
            "metadata": {"dataset": "mmFI"},
        },
        path,
    )


def test_round_trip_and_history(tmp_path: Path) -> None:
    path = tmp_path / "test_features.pt"
    _write_file(path)
    data = PrecomputedFeatureDataset(path, num_keypoints=17, past_frames=6, return_meta=True)
    assert len(data) == 10 and data.has_sample_meta
    features, mu_out, cov, gt, history, meta = data[8]
    assert features.shape == (512,) and mu_out.shape == (17, 3) and cov.shape == (51, 51) and gt.shape == (17, 3)
    assert history.shape == (17, 18)
    assert history[0, 0::3].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    assert meta.participant == "s0" and meta.scene == "E01"


def test_history_at_sequence_start_uses_following_samples(tmp_path: Path) -> None:
    path = tmp_path / "train_features.pt"
    _write_file(path)
    data = PrecomputedFeatureDataset(path, num_keypoints=17, past_frames=6)
    history = data[2][4]
    assert history[0, 0::3].tolist() == [2.0, 3.0, 4.0, 5.0, 0.0, 1.0]


def test_collate(tmp_path: Path) -> None:
    path = tmp_path / "test_features.pt"
    _write_file(path)
    data = PrecomputedFeatureDataset(path, num_keypoints=17, past_frames=6, return_meta=True)
    batch = collate_with_meta([data[0], data[1]])
    assert batch[0].shape == (2, 512) and batch[4].shape == (2, 17, 18) and len(batch[5]) == 2
