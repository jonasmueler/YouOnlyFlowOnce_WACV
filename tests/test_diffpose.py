"""The DiffPose baseline builds, trains one step and samples on the CPU."""

import torch

from yofo.baselines.diffpose import build_diffpose
from yofo.config import MRI, DiffPoseConfig


def test_diffpose_loss_and_sampling() -> None:
    torch.manual_seed(0)
    cfg = DiffPoseConfig(num_samples=4, n_layers=1, feedforward_dim=32, hidden_dim=64, timesteps=3)
    model = build_diffpose(MRI, cfg)
    assert model.condition_emb.output_dim == 2 * 128 * 17
    batch = 2
    mu = torch.randn(batch, 17, 3) * 10
    cov = torch.eye(51).expand(batch, 51, 51).clone() * 4.0
    gt = mu + torch.randn(batch, 17, 3)
    features = torch.randn(batch, 512)
    loss = model(gt, mu, cov, features)
    assert torch.isfinite(loss)
    model.eval()
    samples = model.sample(mu, cov, features, n_hypotheses=5, max_hypotheses_per_batch=2)
    assert samples.shape == (batch, 5, 17, 3)
    assert torch.isfinite(samples).all()
