"""DiffPose (Holmquist and Wandt, ICCV 2023) adapted to radar pose estimation.

Original code: https://github.com/bastianwandt/DiffPose (MIT License, Copyright (c) 2023
Karl Holmquist). The Gaussian-mixture conditioning of the original was replaced by
samples from the MG-Prior; the denoiser and the diffusion process follow the original.
"""

from yofo.third_party.diffpose.condition_embedding import GaussianSamplingPoseformer, SinusoidalPosEmb
from yofo.third_party.diffpose.denoiser import DiffPoseDenoiser, ResBlock
from yofo.third_party.diffpose.diffusion import MultiHypothesisDiffusion
from yofo.third_party.diffpose.schedules import cosine_beta_schedule, extract, linear_beta_schedule

__all__ = [
    "DiffPoseDenoiser",
    "GaussianSamplingPoseformer",
    "MultiHypothesisDiffusion",
    "ResBlock",
    "SinusoidalPosEmb",
    "cosine_beta_schedule",
    "extract",
    "linear_beta_schedule",
]
