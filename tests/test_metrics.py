"""Metrics on hand-built inputs."""

import math

import numpy as np
import torch

from yofo.metrics import expected_calibration_error, interval_coverage, mpjpe, pa_mpjpe, sharpness


def test_mpjpe_hand_value() -> None:
    preds = torch.zeros(1, 2, 3)
    gts = torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]])
    assert math.isclose(mpjpe(preds, gts).item(), 3.0)
    assert math.isclose(mpjpe(preds.view(1, -1), gts.view(1, -1)).item(), 3.0)


def test_pa_mpjpe_is_invariant_to_similarity_transforms() -> None:
    torch.manual_seed(0)
    gt = torch.randn(4, 17, 3)
    angle = 0.7
    rot = torch.tensor([[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    pred = 1.3 * gt @ rot.T + 2.0
    assert pa_mpjpe(pred, gt).item() < 1e-4
    assert mpjpe(pred, gt).item() > 1.0


def test_sharpness_is_mean_variance() -> None:
    samples = torch.tensor([[[0.0, 2.0], [1.0, 1.0]]])
    assert math.isclose(sharpness(samples).item(), 1.0)


def test_coverage_and_ece_on_calibrated_samples() -> None:
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(2000, 3))
    samples = gt[:, :, None] + rng.normal(size=(2000, 3, 400))
    gt_obs = gt + rng.normal(size=gt.shape)
    coverage = interval_coverage(samples, gt_obs, levels=(0.5, 0.9))
    assert abs(coverage[0.5] - 0.5) < 0.03 and abs(coverage[0.9] - 0.9) < 0.03
    assert expected_calibration_error(samples, gt_obs) < 0.02


def test_ece_detects_overconfidence() -> None:
    rng = np.random.default_rng(1)
    gt = rng.normal(size=(2000, 3))
    narrow = gt[:, :, None] + 0.1 * rng.normal(size=(2000, 3, 400))
    gt_obs = gt + rng.normal(size=gt.shape)
    assert expected_calibration_error(narrow, gt_obs) > 0.2
    assert interval_coverage(narrow, gt_obs, levels=(0.95,))[0.95] < 0.2
