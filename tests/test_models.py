"""Model construction, shapes and flow invertibility on the CPU."""

import pytest
import torch

from yofo.config import MMFI, MMRADPOSE, MRI, BackboneConfig, FlowConfig
from yofo.models.flow import RealNVPFlow, flow_nll, sample_poses
from yofo.models.mg_prior import MGPrior
from yofo.training.flow import build_flow


@pytest.mark.parametrize("dataset", [MMFI, MMRADPOSE, MRI])
def test_mg_prior_forward(dataset) -> None:
    torch.manual_seed(0)
    cfg = BackboneConfig(num_layers=1, d_model=64, num_heads=4, dim_feedforward=64, feat_dim=32, d_latent=16, gcn_hidden=8)
    model = MGPrior(dataset, cfg).eval()
    x = torch.randn(2, dataset.seq_len, dataset.max_points_per_frame, dataset.input_dim)
    with torch.no_grad():
        preds, mu, logvar, mu_out, cov = model(x, n_samples=7)
    d = dataset.output_dim
    assert preds.shape == (2, d, 7)
    assert mu.shape == (2, 16) and logvar.shape == (2, 16)
    assert mu_out.shape == (2, d) and cov.shape == (2, d, d)
    assert model.backbone_features.shape == (2, 32)
    assert torch.allclose(cov, cov.transpose(1, 2))


def test_mg_prior_state_dict_keys_match_release() -> None:
    model = MGPrior(MMRADPOSE, BackboneConfig())
    keys = set(model.state_dict())
    assert "backbone.cls_token" in keys and "head.varianceScaling" in keys and "head.adj" in keys
    assert "head.gconv1.weight" in keys and "head.decoder.1.bias" in keys
    assert sum(p.numel() for p in model.backbone.parameters()) == 8_945_152


def test_flow_round_trip() -> None:
    torch.manual_seed(0)
    flow = RealNVPFlow(in_dim=78, num_layers=8, hidden=256, cond_dim=256).eval()
    x = torch.randn(5, 78)
    cond = torch.randn(5, 256)
    z, logdet_fwd = flow(x, cond=cond, reverse=False)
    x_rec, logdet_rev = flow(z, cond=cond, reverse=True)
    assert torch.allclose(x, x_rec, atol=1e-5)
    assert torch.allclose(logdet_fwd, -logdet_rev, atol=1e-5)
    assert torch.isfinite(flow_nll(flow, x, cond))
    samples = sample_poses(flow, 11, cond)
    assert samples.shape == (5, 11, 78)


def test_flow_parameter_counts() -> None:
    flow, cond = build_flow(MMRADPOSE, FlowConfig())
    assert sum(p.numel() for p in flow.parameters()) == 1_533_152
    assert "adj" not in cond.state_dict()
    out = cond(torch.randn(3, 512), torch.randn(3, 26, 18))
    assert out.shape == (3, 256)
