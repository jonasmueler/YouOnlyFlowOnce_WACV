"""Measure parameters, FLOPs and per-frame latency of all models in a causal online setting."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from yofo.baselines.diffpose import build_diffpose
from yofo.baselines.mmdiff import build_backbone, build_gcndiff
from yofo.config import MMRADPOSE, BackboneConfig, DiffPoseConfig, FlowConfig, MmDiffConfig
from yofo.logging_utils import setup_logging
from yofo.models.flow import sample_laplace
from yofo.models.mg_prior import MGPrior
from yofo.seeding import set_seed
from yofo.third_party.mmdiff import GCNdiff, PointTransformerReg, ddim_sequence, generalized_steps, linear_betas
from yofo.training.flow import build_flow

logger = logging.getLogger(__name__)

CM_PER_M = 100.0


class CircularBuffer:
    """Sliding window of the last MG-Prior mean poses.

    Args:
        size: Number of poses kept.
        num_keypoints: Joints per pose.
    """

    def __init__(self, size: int, num_keypoints: int) -> None:
        self.size = size
        self.num_keypoints = num_keypoints
        self.buffer: list[torch.Tensor] = []

    def add(self, pose: torch.Tensor) -> None:
        """Append a pose ``(K, 3)`` and drop the oldest one when the window is full."""
        self.buffer.append(pose.view(self.num_keypoints, 3).clone())
        if len(self.buffer) > self.size:
            self.buffer.pop(0)

    def history(self) -> torch.Tensor:
        """Past poses as ``(1, K, 3 * size)``."""
        return torch.cat(self.buffer, dim=-1).unsqueeze(0)

    def is_full(self) -> bool:
        """Whether the window holds ``size`` poses."""
        return len(self.buffer) == self.size


def sync(device: torch.device) -> None:
    """Wait for all queued CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    """Collects the wall-clock time of named components of one frame."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.timing: dict[str, float] = {}
        self._t0 = 0.0

    def start(self) -> None:
        """Start a component."""
        sync(self.device)
        self._t0 = time.perf_counter()

    def stop(self, name: str) -> None:
        """Stop the current component and record it."""
        sync(self.device)
        self.timing[name] = time.perf_counter() - self._t0

    def total(self) -> dict[str, float]:
        """Timings with the sum added as ``total``."""
        return {**self.timing, "total": sum(self.timing.values())}


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def gflops(fn: Callable[[], Any], device: torch.device) -> float:
    """FLOPs of one call of ``fn`` in GFLOPs, measured with ``torch.profiler``."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad(), torch.profiler.profile(activities=activities, with_flops=True) as prof:
        fn()
    return sum(e.flops for e in prof.key_averages() if e.flops is not None) / 1e9


def statistics(per_run: list[dict[str, float]], keys: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Mean, standard deviation, minimum and maximum over runs, in milliseconds.

    Args:
        per_run: Per-run mean time of every component (seconds).
        keys: Components to report; the ``total`` is their sum. Default: all components.

    Returns:
        Statistics per component and for ``total``.
    """
    keys = [k for k in per_run[0] if k != "total"] if keys is None else list(keys)
    stats: dict[str, dict[str, float]] = {}
    for key in keys + ["total"]:
        if key == "total":
            values = np.array([sum(run[k] for k in keys) for run in per_run]) * 1000.0
        else:
            values = np.array([run[key] for run in per_run]) * 1000.0
        stats[key] = {
            "mean_ms": float(values.mean()),
            "std_ms": float(values.std()),
            "min_ms": float(values.min()),
            "max_ms": float(values.max()),
        }
    return stats


def gaussian_stats_steps(model: MGPrior, features: torch.Tensor, n_samples: int, timer: Timer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode, sample, decode and aggregate with the GCN Gaussian head, timed per component.

    Args:
        model: MG-Prior.
        features: Backbone features ``(1, feat_dim)``.
        n_samples: Hypotheses.
        timer: Timer recording ``latent_encoding``, ``latent_sampling``, ``decoding`` and ``aggregation``.

    Returns:
        Hypotheses ``(1, K*3, n_samples)``, mean ``(1, K*3)`` and covariance ``(1, K*3, K*3)``.
    """
    head = model.head
    timer.start()
    mu, logvar = head.get_latent(features)
    timer.stop("latent_encoding")

    timer.start()
    std = torch.exp(0.5 * logvar) * head.varianceScaling
    eps = torch.randn(n_samples, mu.shape[0], mu.shape[1], device=mu.device)
    z = (mu.unsqueeze(0) + std.unsqueeze(0) * eps).permute(1, 0, 2)
    timer.stop("latent_sampling")

    timer.start()
    decoded = head._apply_gcn(head.decoder(z))
    preds = decoded.permute(0, 2, 1)
    timer.stop("decoding")

    timer.start()
    mu_out, cov = model.compute_gaussian_stats(preds)
    timer.stop("aggregation")
    return preds, mu_out, cov


def frame_mg_prior(model: MGPrior, pc: torch.Tensor, n_samples: int, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    """One frame of the MG-Prior: backbone, head and aggregation."""
    timer = Timer(device)
    timer.start()
    features = model.backbone(pc)
    timer.stop("backbone")
    _, mu_out, _ = gaussian_stats_steps(model, features, n_samples, timer)
    return mu_out, timer.total()


def frame_nfpg(
    model: MGPrior,
    flow: nn.Module,
    cond_module: nn.Module,
    pc: torch.Tensor,
    history: torch.Tensor,
    n_samples: int,
    prior_samples: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """One frame of MH-NFPG: backbone, the MG-Prior mean for the buffer, conditioning, Laplace sampling and the inverse flow.

    The flow is conditioned on the MG-Prior means of the past frames, so every frame computes
    one prior mean with the Gaussian head and returns it for the circular buffer.
    """
    timer = Timer(device)
    timer.start()
    features = model.backbone(pc)
    timer.stop("backbone")
    _, mu_out, _ = gaussian_stats_steps(model, features, prior_samples, timer)

    timer.start()
    cond = cond_module(features, history)
    timer.stop("conditioning")

    timer.start()
    z = sample_laplace((n_samples, flow.in_dim), device=device, dtype=cond.dtype)
    timer.stop("laplace_sampling")

    timer.start()
    poses, _ = flow(z, cond=cond.expand(n_samples, -1), reverse=True)
    timer.stop("flow")
    return poses, mu_out, timer.total()


def frame_diffpose(
    model: MGPrior, diffusion: nn.Module, pc: torch.Tensor, n_hypotheses: int, prior_samples: int, device: torch.device
) -> tuple[torch.Tensor, dict[str, float]]:
    """One frame of MG-Prior + DiffPose: prior statistics, conditioning once per hypothesis, 25 denoising steps."""
    timer = Timer(device)
    timer.start()
    features = model.backbone(pc)
    timer.stop("backbone")
    _, mu_out, cov = gaussian_stats_steps(model, features, prior_samples, timer)

    timer.start()
    num_joints = diffusion.num_joints
    mu_rep = (mu_out / CM_PER_M).view(1, num_joints, 3).expand(n_hypotheses, -1, -1)
    cov_rep = (cov / CM_PER_M**2).expand(n_hypotheses, -1, -1)
    feat_rep = features.expand(n_hypotheses, -1)
    cond = diffusion.condition_emb(mu_rep, cov_rep, 1, feat_rep).squeeze(1)
    timer.stop("conditioning")

    timer.start()
    x = torch.randn(n_hypotheses, num_joints * 3, device=device) * diffusion.noise_scale
    for step in reversed(range(diffusion.num_timesteps)):
        t = torch.full((n_hypotheses,), step, device=device, dtype=torch.long)
        x = diffusion.p_sample(x, t, cond)
    timer.stop("denoising")
    return x / diffusion.scaling_3d_pose * CM_PER_M, timer.total()


def frame_mmdiff(
    backbone: PointTransformerReg,
    gcndiff: GCNdiff,
    betas: torch.Tensor,
    seq_eta0: list[int],
    seq_eta1: list[int],
    pc: torch.Tensor,
    history: torch.Tensor,
    n_hypotheses: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """One frame of mmDiff: backbone, conditioning once, then both samplers.

    The deterministic sampler refines the coarse pose with one hypothesis; the
    stochastic sampler draws ``n_hypotheses`` hypotheses in the batch dimension. The
    coarse pose of the backbone is returned for the temporal buffer.

    Args:
        backbone: PointTransformer.
        gcndiff: Denoiser.
        betas: Betas ``(T,)``.
        seq_eta0: Timesteps of the deterministic sampler.
        seq_eta1: Timesteps of the stochastic sampler.
        pc: Point cloud ``(1, T, N, F)``.
        history: Past coarse poses ``(1, K, 3 * past_frames)``.
        n_hypotheses: Hypotheses of the stochastic sampler.
        device: Device.

    Returns:
        Coarse pose ``(1, K, 3)`` and the timings.
    """
    timer = Timer(device)
    timer.start()
    pts, feat = backbone(pc)
    timer.stop("pointtransformer_backbone")
    start = pts - pts[:, :1, :]

    timer.start()
    global_emb = gcndiff.global_embedding(feat)
    timer.stop("global_conditioning")
    timer.start()
    past_emb = gcndiff.temporal_embedding(history)
    timer.stop("temporal_conditioning")
    cond = global_emb + past_emb

    timer.start()
    generalized_steps(lambda x, t: gcndiff(torch.cat([history, x], dim=-1), t, feat, condition=cond)[0], start, seq_eta0, betas, 0.0)
    timer.stop("denoising_eta0")

    hist_n = history.expand(n_hypotheses, -1, -1)
    feat_n = feat.expand(n_hypotheses, -1, -1)
    cond_n = cond.expand(n_hypotheses, -1, -1)
    timer.start()
    generalized_steps(
        lambda x, t: gcndiff(torch.cat([hist_n, x], dim=-1), t, feat_n, condition=cond_n)[0],
        start.expand(n_hypotheses, -1, -1),
        seq_eta1,
        betas,
        1.0,
    )
    timer.stop("denoising_eta1")
    return pts, timer.total()


def run_model(
    name: str,
    frame_fn: Callable[[torch.Tensor, CircularBuffer], tuple[torch.Tensor, dict[str, float]]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float]]:
    """Time a model over ``args.num_runs`` runs of ``args.num_samples`` frames each.

    Args:
        name: Model name for the progress bar.
        frame_fn: Processes one point-cloud frame given the buffer and returns a pose ``(K, 3)``
            to append to the buffer plus the timings.
        args: Benchmark settings.
        device: Device.

    Returns:
        Per-run mean time of every component, in seconds.
    """
    dataset = MMRADPOSE
    per_run: list[dict[str, float]] = []
    with torch.no_grad():
        for _ in tqdm(range(args.num_runs), desc=name):
            buffer = CircularBuffer(args.past_frames, dataset.num_keypoints)
            while not buffer.is_full():
                buffer.add(torch.randn(dataset.num_keypoints, 3, device=device))
            timings: list[dict[str, float]] = []
            for frame in range(args.past_frames + args.num_samples):
                pc = torch.randn(1, dataset.seq_len, dataset.max_points_per_frame, dataset.input_dim, device=device)
                pose, timing = frame_fn(pc, buffer)
                buffer.add(pose)
                if frame >= args.past_frames:
                    timings.append(timing)
            per_run.append({k: float(np.mean([t[k] for t in timings])) for k in timings[0]})
    return per_run


def benchmark(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Build the three models and benchmark them.

    Args:
        args: Benchmark settings.
        device: Device.

    Returns:
        Parameters, GFLOPs and timings per model and component.
    """
    dataset = MMRADPOSE
    n = args.num_hypotheses
    results: dict[str, Any] = {}

    model = MGPrior(dataset, BackboneConfig()).to(device).eval()
    pc = torch.randn(1, dataset.seq_len, dataset.max_points_per_frame, dataset.input_dim, device=device)
    with torch.no_grad():
        features = model.backbone(pc)
    z = torch.randn(1, n, BackboneConfig().d_latent, device=device)
    head = model.head
    params_backbone = count_parameters(model.backbone)
    params_encoder = count_parameters(head.mu) + count_parameters(head.logvar) + 1
    params_decoder = count_parameters(head.decoder) + count_parameters(head.gconv1) + count_parameters(head.gconv2)
    gflops_backbone = gflops(lambda: model.backbone(pc), device)
    gflops_encoder = gflops(lambda: head.get_latent(features), device)
    gflops_decoder = gflops(lambda: head._apply_gcn(head.decoder(z)), device)
    z_prior = torch.randn(1, args.prior_samples, BackboneConfig().d_latent, device=device)
    gflops_decoder_prior = gflops(lambda: head._apply_gcn(head.decoder(z_prior)), device)

    logger.info("MG-Prior")
    results["mg_prior"] = {
        "components": {
            "backbone": {"params": params_backbone, "gflops": gflops_backbone},
            "latent_encoding": {"params": params_encoder, "gflops": gflops_encoder},
            "latent_sampling": {"params": None, "gflops": None},
            "decoding": {"params": params_decoder, "gflops": gflops_decoder},
            "aggregation": {"params": None, "gflops": None},
        },
        "timing": statistics(
            run_model(
                "MG-Prior", lambda pc_, _buf: _as_pose(*frame_mg_prior(model, pc_, n, device), dataset.num_keypoints), args, device
            )
        ),
    }

    logger.info("MH-NFPG")
    flow, cond_module = build_flow(dataset, dataclasses.replace(FlowConfig(), past_frames=args.past_frames))
    flow = flow.to(device).eval()
    cond_module = cond_module.to(device).eval()
    history = torch.randn(1, dataset.num_keypoints, 3 * args.past_frames, device=device)
    with torch.no_grad():
        cond = cond_module(features, history)
    zl = torch.randn(n, flow.in_dim, device=device)
    results["mh_nfpg"] = {
        "components": {
            "backbone": {"params": params_backbone, "gflops": gflops_backbone},
            "latent_encoding": {"params": params_encoder, "gflops": gflops_encoder},
            "latent_sampling": {"params": None, "gflops": None},
            "decoding": {"params": params_decoder, "gflops": gflops_decoder_prior},
            "aggregation": {"params": None, "gflops": None},
            "conditioning": {"params": count_parameters(cond_module), "gflops": gflops(lambda: cond_module(features, history), device)},
            "laplace_sampling": {"params": None, "gflops": None},
            "flow": {"params": count_parameters(flow), "gflops": gflops(lambda: flow(zl, cond=cond.expand(n, -1), reverse=True), device)},
        },
        "timing": statistics(
            run_model(
                "MH-NFPG",
                lambda pc_, buf: _as_pose(*frame_nfpg(model, flow, cond_module, pc_, buf.history(), n, args.prior_samples, device)[1:], dataset.num_keypoints),
                args,
                device,
            )
        ),
    }

    logger.info("MG-Prior + DiffPose")
    diffusion = build_diffpose(dataset, DiffPoseConfig()).to(device).eval()
    with torch.no_grad():
        _, mu_out, cov = gaussian_stats_steps(model, features, args.prior_samples, Timer(device))
        mu_rep = (mu_out / CM_PER_M).view(1, dataset.num_keypoints, 3).expand(n, -1, -1)
        cov_rep = (cov / CM_PER_M**2).expand(n, -1, -1)
        feat_rep = features.expand(n, -1)
        cond_d = diffusion.condition_emb(mu_rep, cov_rep, 1, feat_rep).squeeze(1)
    x = torch.randn(n, dataset.output_dim, device=device)
    t = torch.zeros(n, device=device, dtype=torch.long)
    results["mg_prior_diffpose"] = {
        "components": {
            "backbone": {"params": params_backbone, "gflops": gflops_backbone},
            "latent_encoding": {"params": params_encoder, "gflops": gflops_encoder},
            "latent_sampling": {"params": None, "gflops": None},
            "decoding": {"params": params_decoder, "gflops": gflops_decoder_prior},
            "aggregation": {"params": None, "gflops": None},
            "conditioning": {
                "params": count_parameters(diffusion.condition_emb),
                "gflops": gflops(lambda: diffusion.condition_emb(mu_rep, cov_rep, 1, feat_rep), device),
            },
            "denoising": {
                "params": count_parameters(diffusion.denoise_fn),
                "gflops": diffusion.num_timesteps * gflops(lambda: diffusion.p_sample(x, t, cond_d), device),
            },
        },
        "timing": statistics(
            run_model(
                "MG-Prior + DiffPose",
                lambda pc_, _buf: _mean_pose(*frame_diffpose(model, diffusion, pc_, n, args.prior_samples, device), dataset.num_keypoints),
                args,
                device,
            )
        ),
    }

    logger.info("mmDiff")
    mm_cfg = dataclasses.replace(MmDiffConfig(), past_frames=args.past_frames)
    backbone = build_backbone(dataset, mm_cfg).to(device).eval()
    gcndiff = build_gcndiff(dataset, mm_cfg).to(device).eval()
    betas = linear_betas(mm_cfg.timesteps, mm_cfg.beta_start, mm_cfg.beta_end).to(device)
    seq_eta0 = ddim_sequence(mm_cfg.test_horizon, mm_cfg.eta0_steps)
    seq_eta1 = ddim_sequence(mm_cfg.test_horizon, mm_cfg.eta1_steps)
    with torch.no_grad():
        pts, feat = backbone(pc)
        cond_1 = gcndiff.global_embedding(feat) + gcndiff.temporal_embedding(history)
    x_1 = torch.cat([history, pts], dim=-1)
    t_1 = torch.zeros(1, device=device)
    x_n = x_1.expand(n, -1, -1)
    feat_n = feat.expand(n, -1, -1)
    cond_n = cond_1.expand(n, -1, -1)
    t_n = torch.zeros(n, device=device)
    params_global = count_parameters(gcndiff.gconv_input1)
    params_temporal = count_parameters(gcndiff.gconv_input_t) + count_parameters(gcndiff.temperal_conv)
    params_denoising = count_parameters(gcndiff) - params_global - params_temporal
    mm_components = {
        "pointtransformer_backbone": {"params": count_parameters(backbone), "gflops": gflops(lambda: backbone(pc), device)},
        "global_conditioning": {"params": params_global, "gflops": gflops(lambda: gcndiff.global_embedding(feat), device)},
        "temporal_conditioning": {"params": params_temporal, "gflops": gflops(lambda: gcndiff.temporal_embedding(history), device)},
    }
    gflops_call_1 = gflops(lambda: gcndiff(x_1, t_1, feat, condition=cond_1), device)
    gflops_call_n = gflops(lambda: gcndiff(x_n, t_n, feat_n, condition=cond_n), device)
    mm_per_run = run_model(
        "mmDiff",
        lambda pc_, buf: _as_pose(*frame_mmdiff(backbone, gcndiff, betas, seq_eta0, seq_eta1, pc_, buf.history(), n, device), dataset.num_keypoints),
        args,
        device,
    )
    shared_keys = list(mm_components)
    results["mmdiff_eta0"] = {
        "components": {**mm_components, "denoising_eta0": {"params": params_denoising, "gflops": len(seq_eta0) * gflops_call_1}},
        "timing": statistics(mm_per_run, shared_keys + ["denoising_eta0"]),
    }
    results["mmdiff_eta1"] = {
        "components": {**mm_components, "denoising_eta1": {"params": params_denoising, "gflops": len(seq_eta1) * gflops_call_n}},
        "timing": statistics(mm_per_run, shared_keys + ["denoising_eta1"]),
    }

    for entry in results.values():
        comps = entry["components"]
        entry["parameters_total"] = sum(c["params"] or 0 for c in comps.values())
        entry["gflops_total"] = sum(c["gflops"] or 0.0 for c in comps.values())
    return results


def _as_pose(mu_out: torch.Tensor, timing: dict[str, float], num_keypoints: int) -> tuple[torch.Tensor, dict[str, float]]:
    """Reshape the MG-Prior mean to ``(K, 3)``."""
    return mu_out.view(num_keypoints, 3), timing


def _mean_pose(samples: torch.Tensor, timing: dict[str, float], num_keypoints: int) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean of hypotheses ``(n, K*3)`` as ``(K, 3)``."""
    return samples.mean(dim=0).view(num_keypoints, 3), timing


COMPONENT_LABELS = {
    "backbone": "Spatiotemporal Transformer Backbone",
    "latent_encoding": "Gaussian Head - Encoder",
    "latent_sampling": "Gaussian Head - Latent Sampling",
    "decoding": "Gaussian Head - Decoder (with GCN)",
    "aggregation": "Aggregation (mean, covariance)",
    "conditioning": "Conditioning Module",
    "laplace_sampling": "Standard Laplace Sampling",
    "flow": "Real NVP Normalizing Flow",
    "denoising": "Diffusion Denoiser (25 steps)",
    "pointtransformer_backbone": "PointTransformer Backbone",
    "global_conditioning": "Global Conditioning",
    "temporal_conditioning": "Temporal Conditioning",
    "denoising_eta0": "GCNdiff Denoising incl. limb head (3 steps, 1 hyp.)",
    "denoising_eta1": "GCNdiff Denoising incl. limb head (25 steps)",
}
MODEL_LABELS = {
    "mg_prior": "MG-Prior",
    "mh_nfpg": "MG-Prior + NFPG",
    "mg_prior_diffpose": "MG-Prior + DiffPose",
    "mmdiff_eta0": "mmDiff eta=0 (det., 1 hyp.)",
    "mmdiff_eta1": "mmDiff eta=1",
}


def markdown_table(results: dict[str, Any], n_hypotheses: int) -> str:
    """Render the per-component and summary tables as Markdown."""
    lines = [f"## Runtime benchmark ({n_hypotheses} hypotheses, batch size 1)", ""]
    for name, entry in results.items():
        lines += [f"### {MODEL_LABELS[name]}", "", "| Component | Parameters | GFLOPs | Time (ms) |", "|---|---|---|---|"]
        for comp, values in entry["components"].items():
            timing = entry["timing"][comp]
            params = f"{values['params']:,}" if values["params"] is not None else "---"
            flops = f"{values['gflops']:.3f}" if values["gflops"] is not None else "---"
            lines.append(f"| {COMPONENT_LABELS[comp]} | {params} | {flops} | {timing['mean_ms']:.3f} ± {timing['std_ms']:.3f} |")
        total = entry["timing"]["total"]
        lines.append(f"| **Total** | **{entry['parameters_total']:,}** | **{entry['gflops_total']:.3f}** | **{total['mean_ms']:.3f} ± {total['std_ms']:.3f}** |")
        lines.append("")
    slowest = max(e["timing"]["total"]["mean_ms"] for e in results.values())
    lines += ["### Summary", "", "| Model | Params (M) | GFLOPs | Time (ms) | Speed-up |", "|---|---|---|---|---|"]
    for name, entry in results.items():
        total = entry["timing"]["total"]
        lines.append(
            f"| {MODEL_LABELS[name]} | {entry['parameters_total'] / 1e6:.3f} | {entry['gflops_total']:.3f} | "
            f"{total['mean_ms']:.3f} ± {total['std_ms']:.3f} | {slowest / total['mean_ms']:.2f}x |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments, run the benchmark and write JSON and Markdown."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"), help="Output JSON (a .md table is written next to it).")
    parser.add_argument("--num_samples", type=int, default=100, help="Timed frames per run.")
    parser.add_argument("--num_runs", type=int, default=50, help="Runs; the standard deviation is across runs.")
    parser.add_argument("--num_hypotheses", type=int, default=200, help="Hypotheses per frame.")
    parser.add_argument("--past_frames", type=int, default=FlowConfig.past_frames, help="Temporal context of the flow and warm-up frames.")
    parser.add_argument(
        "--prior_samples",
        type=int,
        default=BackboneConfig.n_samples_extract,
        help="Latent samples of the MG-Prior mean and covariance that condition the flow and DiffPose.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Torch device.")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the random weights and inputs.")
    args = parser.parse_args()
    setup_logging()
    set_seed(args.seed, deterministic=False)
    device = torch.device(args.device)

    results = benchmark(args, device)
    record = {
        "metadata": {
            "dataset": MMRADPOSE.name,
            "num_keypoints": MMRADPOSE.num_keypoints,
            "input_shape": [1, MMRADPOSE.seq_len, MMRADPOSE.max_points_per_frame, MMRADPOSE.input_dim],
            "num_hypotheses": args.num_hypotheses,
            "prior_samples": args.prior_samples,
            "timed_frames_per_run": args.num_samples,
            "num_runs": args.num_runs,
            "past_frames": args.past_frames,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch": torch.__version__,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    table = markdown_table(results, args.num_hypotheses)
    args.output.with_suffix(".md").write_text(table, encoding="utf-8")
    print(table)
    logger.info("Results written to %s and %s", args.output, args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
