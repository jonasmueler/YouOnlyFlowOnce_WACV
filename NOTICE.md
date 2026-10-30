# Third-party notices

This repository is released under the PolyForm Noncommercial License 1.0.0 (see
`LICENSE`). The following parts derive from other projects.

## DiffPose (`yofo/third_party/diffpose/`)

Adapted from DiffPose: Multi-hypothesis Human Pose Estimation using Diffusion Models
(Karl Holmquist and Bastian Wandt, ICCV 2023), https://github.com/bastianwandt/DiffPose,
MIT License, Copyright (c) 2023 Karl Holmquist. The license text is in
`yofo/third_party/diffpose/LICENSE`. Changes: the Gaussian-mixture heatmap conditioning
was replaced by samples from the MG-Prior together with a projection of the backbone
feature; the denoiser, the noise schedules and the diffusion process follow the original.

## mmDiff (`yofo/third_party/mmdiff/`)

Adapted from mmDiff: Diffusion Model is a Good Pose Estimator from 3D RF-Vision
(Junqiao Fan, Jianfei Yang, Yuecong Xu and Lihua Xie, ECCV 2024),
https://github.com/FanJunqiao/mmDiff, MIT License, Copyright (c) 2023. The license text
is in `yofo/third_party/mmdiff/LICENSE`. Changes: the joint count, the number of past
frames and the limb count are arguments; the modules the original defines but never
runs (local radar conditioning) are omitted; `einops` calls were replaced by reshapes.

## MM-Fi toolbox (`yofo/data/mmfi.py`)

The reading of the MM-Fi radar files and the definition of the cross-subject,
cross-environment and random splits follow the official MM-Fi toolbox,
https://github.com/ybhbingo/MMFi_dataset (Yang et al., NeurIPS 2023 Datasets and
Benchmarks). The code in this repository is a re-implementation of the parts needed to
read the mmWave modality; no file of the toolbox is redistributed.

## ChebNet graph convolution (`yofo/models/chebconv.py`)

The Chebyshev graph convolution follows the implementation used by GraFormer
(Zhao et al., CVPR 2022), https://github.com/Graformer/GraFormer, and mmDiff
(Fan et al., ECCV 2024).
