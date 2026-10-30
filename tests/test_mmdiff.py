"""The mmDiff re-implementation: structure, diffusion arithmetic, data rules and losses."""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from yofo.baselines.mmdiff import MmDiffFeatureDataset, build_backbone, build_gcndiff, collate_mmdiff, limb_lengths, limb_pairs
from yofo.config import MMFI, MMRADPOSE, MRI, MmDiffConfig, Paths
from yofo.data.features import PrecomputedFeatureDataset, history_indices
from yofo.data.mmfi import scene_subjects, split_forms
from yofo.evaluation.common import EvalCollector
from yofo.config import EvalConfig
from yofo.evaluation.mmdiff import centre_history, sample_hypotheses
from yofo.models.chebconv import ChebConv
from yofo.models.skeleton import COCO_SKELETON_EDGES, MMRADPOSE_SKELETON_EDGES, adjacency_from_edges, skeleton_adjacency
from yofo.third_party.mmdiff import (
    GCNdiff,
    PointTransformerReg,
    alpha_bar,
    ddim_sequence,
    generalized_steps,
    linear_betas,
    sample_timesteps_antithetic,
)
from yofo.training.mmdiff import diffusion_losses, learning_rate_at, train_diffusion
from yofo.training.wandb_logging import WandbLogger


def _count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_parameter_counts_match_the_original() -> None:
    torch.manual_seed(0)
    # 8,009,123 is the PointTransformer count of Table S3 (26 joints, 5 point features).
    assert _count(PointTransformerReg(input_dim=5, num_keypoints=26)) == 8_009_123
    assert _count(PointTransformerReg(input_dim=5, num_keypoints=17)) == 8_004_515
    assert _count(PointTransformerReg(input_dim=7, num_keypoints=26)) == 8_009_187
    # GCNdiff in the original's MM-Fi configuration (8 past frames, 16 limbs).
    assert _count(GCNdiff(skeleton_adjacency("coco", 17), 17, past_frames=8, num_limbs=16)) == 17_668_262
    cfg = MmDiffConfig()
    assert _count(build_gcndiff(MMFI, cfg)) == 17_561_876
    assert _count(build_gcndiff(MMRADPOSE, cfg)) == 39_171_686
    assert _count(build_backbone(MMRADPOSE, cfg)) == 8_009_187


def test_gcndiff_structure_and_initialisation_quirk() -> None:
    torch.manual_seed(0)
    model = build_gcndiff(MRI, MmDiffConfig())
    assert model.gconv_output.weight.shape[-1] == 3 * 7
    assert model.limb_linear[1].in_features == 7 * 17 * 3 and model.limb_linear[3].out_features == 16
    assert len(model.gconv_layers) == 5 and len(model.atten_layers) == 5
    first, second = model.atten_layers[0], model.atten_layers[1]
    assert torch.equal(first.self_attn.linears[0].weight, second.self_attn.linears[0].weight)
    assert torch.equal(first.feed_forward.gconv1.fc.weight, second.feed_forward.gconv1.fc.weight)
    assert not torch.equal(model.gconv_layers[0].gconv1.gconv.weight, model.gconv_layers[1].gconv1.gconv.weight)
    assert ChebConv(3, 4, K=2).weight.shape[0] == 3
    model.eval()
    x = torch.randn(2, 17, 21)
    eps, limbs = model(x, torch.tensor([24.0, 0.0]), torch.randn(2, 17, 32))
    assert eps.shape == (2, 17, 3) and limbs.shape == (2, 16)
    cond = model.global_embedding(torch.zeros(2, 17, 32)) + model.temporal_embedding(x[:, :, :18])
    eps2, _ = model(x, torch.tensor([24.0, 0.0]), torch.zeros(2, 17, 32), condition=cond)
    eps3, _ = model(x, torch.tensor([24.0, 0.0]), torch.zeros(2, 17, 32))
    assert torch.allclose(eps2, eps3, atol=1e-6)


@pytest.mark.parametrize("num_pts,edges", [(17, COCO_SKELETON_EDGES), (26, MMRADPOSE_SKELETON_EDGES)])
def test_adjacency_equals_the_original_formula(num_pts: int, edges: torch.Tensor) -> None:
    e = np.array(edges, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(e.shape[0]), (e[:, 0], e[:, 1])), shape=(num_pts, num_pts), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(num_pts)
    rowsum = np.array(adj.sum(1)).flatten()
    expected = sp.diags(1.0 / rowsum).dot(adj).todense()
    assert np.allclose(adjacency_from_edges(num_pts, edges).numpy(), expected)


def test_diffusion_arithmetic() -> None:
    betas = linear_betas(51, 1e-4, 1e-3)
    assert betas.dtype == torch.float32 and betas.shape == (51,)
    assert abs(betas[0].item() - 1e-4) < 1e-9 and abs(betas[-1].item() - 1e-3) < 1e-9
    values = {t: alpha_bar(betas, torch.tensor([t])).item() for t in (-1, 0, 12, 24, 50)}
    assert values[-1] == 1.0
    assert abs(values[0] - 0.9999) < 1e-6
    assert abs(values[12] - 0.997299) < 1e-6
    assert abs(values[24] - 0.992130) < 1e-6
    assert abs(values[50] - 0.972331) < 1e-6
    assert ddim_sequence(25, 2) == [0, 12, 24]
    assert ddim_sequence(25, 25) == list(range(25))


def test_antithetic_pairs() -> None:
    torch.manual_seed(1)
    n = 10
    t = sample_timesteps_antithetic(n, 51, torch.device("cpu"))
    half = n // 2 + 1
    for i in range(n - half):
        assert t[i].item() + t[half + i].item() == 50


def test_sampler_is_deterministic_for_eta_zero_and_noise_free_at_the_end() -> None:
    torch.manual_seed(0)
    betas = linear_betas(51, 1e-4, 1e-3)
    net = torch.nn.Linear(3, 3)

    def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return net(x) * 0.01

    x0 = torch.randn(4, 17, 3)
    a = generalized_steps(model_fn, x0, [0, 12, 24], betas, 0.0)
    b = generalized_steps(model_fn, x0, [0, 12, 24], betas, 0.0)
    assert torch.allclose(a, b)
    # One-step closed form: from t=24 to t=-1 the output is the clean-pose estimate.
    at = alpha_bar(betas, torch.tensor([24]))
    eps = model_fn(x0, torch.tensor([24.0]))
    expected = (x0 - eps * (1 - at).sqrt()) / at.sqrt()
    assert torch.allclose(generalized_steps(model_fn, x0, [24], betas, 1.0), expected, atol=1e-6)
    g1 = torch.Generator().manual_seed(1)
    g2 = torch.Generator().manual_seed(2)
    s1 = generalized_steps(model_fn, x0, list(range(25)), betas, 1.0, g1)
    s2 = generalized_steps(model_fn, x0, list(range(25)), betas, 1.0, g2)
    assert not torch.allclose(s1, s2)


def test_training_losses_follow_the_original() -> None:
    torch.manual_seed(0)
    cfg = MmDiffConfig()
    model = build_gcndiff(MRI, cfg)
    betas = linear_betas(cfg.timesteps, cfg.beta_start, cfg.beta_end)
    pairs = limb_pairs(MRI)
    gt = torch.randn(6, 17, 3) * 0.3
    assert limb_lengths(gt, pairs).shape == (6, 16)
    assert torch.allclose(limb_lengths(gt, pairs)[:, 0], torch.norm(gt[:, 0] - gt[:, 1], dim=-1))
    torch.manual_seed(3)
    total, noise_loss, limb_loss = diffusion_losses(model, betas, pairs, cfg, torch.randn(6, 17, 32), gt, torch.randn(6, 17, 18))
    assert torch.isfinite(total) and torch.allclose(total, noise_loss + 10.0 * limb_loss)
    assert noise_loss.item() > 10.0  # a sum over 51 coordinates, not a mean
    assert learning_rate_at(cfg, 0) == 2e-5 and learning_rate_at(cfg, 59) == 2e-5
    assert abs(learning_rate_at(cfg, 60) - 1.8e-5) < 1e-12 and abs(learning_rate_at(cfg, 99) - 1.8e-5) < 1e-12


def test_history_rule_is_shared_with_the_flow(tmp_path: Path) -> None:
    assert history_indices(0, 6) == [0, 1, 2, 3, 4, 5]
    assert history_indices(2, 6) == [2, 3, 4, 5, 0, 1]
    assert history_indices(10, 6) == [4, 5, 6, 7, 8, 9]
    n, k = 12, 17
    coarse = torch.arange(n, dtype=torch.float).view(n, 1, 1).expand(n, k, 3).clone()
    torch.save(
        {
            "features": torch.randn(n, 512),
            "mu_out": coarse,
            "cov": torch.eye(3 * k).expand(n, 3 * k, 3 * k).clone(),
            "gt_pose": torch.randn(n, k, 3),
        },
        tmp_path / "flow.pt",
    )
    torch.save(
        {
            "features": torch.randn(n, k, 32),
            "coarse_pose": coarse,
            "gt_pose": torch.randn(n, k, 3),
            "participants": [f"s{i % 3}" for i in range(n)],
            "metadata": {"dataset": "MRI"},
        },
        tmp_path / "mmdiff_test_features.pt",
    )
    flow = PrecomputedFeatureDataset(tmp_path / "flow.pt", k, past_frames=6)
    mm = MmDiffFeatureDataset(tmp_path / "mmdiff_test_features.pt", k, past_frames=6, return_meta=True)
    for idx in (0, 2, 5, 11):
        assert torch.equal(flow.history(idx), mm.history(idx))
    features, pose, gt, history, meta = mm[7]
    assert features.shape == (k, 32) and pose.shape == (k, 3) and history.shape == (k, 18) and meta.participant == "s1"
    batch = collate_mmdiff([mm[0], mm[1]])
    assert batch[3].shape == (2, k, 18) and len(batch[4]) == 2


def test_loeo_splits_hold_out_one_environment() -> None:
    assert scene_subjects("E01") == [f"S{i:02d}" for i in range(1, 11)]
    assert scene_subjects("E04") == [f"S{i:02d}" for i in range(31, 41)]
    train, test = split_forms(3, "loeo_E02", 0)
    assert list(test) == [f"S{i:02d}" for i in range(11, 21)]
    assert list(train) == [f"S{i:02d}" for i in range(1, 11)] + [f"S{i:02d}" for i in range(21, 41)]
    assert all(len(actions) == 27 for actions in list(train.values()) + list(test.values()))
    train, test = split_forms(3, "loeo_E01", 0)
    assert list(test) == scene_subjects("E01") and list(train)[0] == "S11"


def test_centre_history_and_hypothesis_batching() -> None:
    torch.manual_seed(0)
    history = torch.randn(3, 17, 18)
    centred = centre_history(history, 17, 6)
    assert torch.allclose(centred.view(3, 17, 6, 3)[:, 0], torch.zeros(3, 6, 3), atol=1e-7)
    cfg = MmDiffConfig()
    model = build_gcndiff(MRI, cfg).eval()
    betas = linear_betas(cfg.timesteps, cfg.beta_start, cfg.beta_end)
    features = torch.randn(3, 17, 32)
    start = torch.randn(3, 17, 3) * 0.2
    g = torch.Generator().manual_seed(0)
    hyp = sample_hypotheses(model, features, centred, start, [0, 1, 2], betas, 1.0, 4, max_rows=8, generator=g)
    assert hyp.shape == (3, 4, 17, 3) and torch.isfinite(hyp).all()
    det = sample_hypotheses(model, features, centred, start, [0, 12, 24], betas, 0.0, 1, max_rows=8, generator=None)
    det_again = sample_hypotheses(model, features, centred, start, [0, 12, 24], betas, 0.0, 1, max_rows=1, generator=None)
    assert torch.allclose(det, det_again, atol=1e-6)


def test_collector_without_samples_reports_point_metrics_only() -> None:
    collector = EvalCollector()
    preds = torch.randn(5, 17, 3)
    collector.add(preds, preds + 1.0, None, {"participant": ["a", "a", "b", "b", "b"]})
    results = collector.aggregate(EvalConfig())
    assert set(results["overall"]) == {"mpjpe", "pa_mpjpe", "n"}
    assert abs(results["overall"]["mpjpe"] - 3**0.5) < 1e-3
    assert set(results["per_participant"]) == {"a", "b"}


def test_diffusion_training_resumes_from_the_epoch_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    n, k = 8, 17
    folder = tmp_path / "features"
    folder.mkdir()
    torch.save(
        {
            "features": torch.randn(n, k, 32),
            "coarse_pose": torch.randn(n, k, 3) * 0.2,
            "gt_pose": torch.randn(n, k, 3) * 0.2,
            "participants": ["s1"] * n,
            "metadata": {"dataset": "MRI"},
        },
        folder / "mmdiff_train_features.pt",
    )
    cfg = MmDiffConfig(epochs=1, batch_size=4, num_workers=0)
    paths = Paths(data_root=tmp_path, run_root=tmp_path / "runs")
    logger = WandbLogger(False, "test", {})
    run_folder = train_diffusion(MRI, "protocol2", folder, cfg, paths, torch.device("cpu"), 42, False, logger)
    assert (run_folder / "epoch_000.pt").is_file() and (run_folder / "final_model.pt").is_file()
    state = torch.load(run_folder / "epoch_000.pt", map_location="cpu", weights_only=False)
    assert state["rng_cpu"].dtype == torch.uint8 and state["rng_cpu"].device.type == "cpu"
    cfg2 = MmDiffConfig(epochs=2, batch_size=4, num_workers=0)
    resumed = train_diffusion(MRI, "protocol2", folder, cfg2, paths, torch.device("cpu"), 42, False, logger, resume_from=run_folder)
    assert resumed == run_folder and (run_folder / "epoch_001.pt").is_file()
    final = torch.load(run_folder / "final_model.pt", map_location="cpu", weights_only=False)
    assert final["epoch"] == 1
