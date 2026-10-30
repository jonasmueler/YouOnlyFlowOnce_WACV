"""mmDiff (Fan et al., ECCV 2024) re-implemented for the baselines of the paper.

Original code: https://github.com/FanJunqiao/mmDiff (MIT License, Copyright (c) 2023).
The PointTransformer backbone, the GCNdiff denoiser, the noise schedule and the DDIM
sampler follow the original; the joint count, the number of past frames and the limb
count are arguments.
"""

from yofo.third_party.mmdiff.diffusion import (
    alpha_bar,
    ddim_sequence,
    generalized_steps,
    linear_betas,
    sample_timesteps_antithetic,
)
from yofo.third_party.mmdiff.gcndiff import UPSTREAM_UNUSED_PREFIXES, GCNdiff, load_upstream_state
from yofo.third_party.mmdiff.point_transformer import PointTransformerReg

__all__ = [
    "GCNdiff",
    "PointTransformerReg",
    "UPSTREAM_UNUSED_PREFIXES",
    "alpha_bar",
    "ddim_sequence",
    "generalized_steps",
    "linear_betas",
    "load_upstream_state",
    "sample_timesteps_antithetic",
]
