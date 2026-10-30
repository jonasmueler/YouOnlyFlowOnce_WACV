# Baselines

The paper compares MH-NFPG with five baselines on every dataset. All of them are part
of this repository.

## MG-Prior (phase 1 alone)

The backbone with the GCN Gaussian head, trained as described in the paper
(`train_backbone.py`). At test time 200 latent samples are decoded and refined per
frame; the sample mean is the point estimate, the samples give ECE, coverage and
sharpness. `evaluate.py --model gaussian` evaluates it on the raw point clouds with the
same windows, hypotheses and metrics as the other models.

## MG-Prior + DiffPose

DiffPose (Holmquist and Wandt, ICCV 2023) is a diffusion model over 3D poses whose
denoiser is conditioned on a joint-wise transformer embedding of pose samples. In the
original, the samples come from heatmap-based Gaussian mixtures of a 2D detector. Here
they come from the MG-Prior:

* For every hypothesis, the MG-Prior mean and 31 samples of its Gaussian are embedded
  per joint (sinusoidal coordinate embedding), weighted by their Gaussian likelihood
  (softmax over the log-densities), summed, combined with a joint identity embedding and
  passed through a four-layer transformer over the joints. The output is concatenated
  with a projection of the 512-dimensional backbone feature (`yofo/third_party/diffpose/condition_embedding.py`).
* The denoiser is the MLP of the original with two residual blocks of width 256
  (`denoiser.py`); the diffusion process uses 25 steps with the cosine schedule, poses
  in metres scaled by 10, clipping of the denoised estimate and the noise-prediction
  objective with an L2 loss (`diffusion.py`).
* Training: 100 epochs on the same extracted features as the flow, that is on the
  released MG-Prior of each setting, batch size 128, Adam with learning rate 1e-4,
  gradient clipping at 1.0, exponential moving average of the weights with decay 0.995;
  the EMA weights are evaluated (`train_diffpose.py`, `evaluate.py --model diffpose`).
* Evaluation: 200 hypotheses per frame, generated in chunks of 32, each with its own
  conditioning samples; their mean is the point estimate.

Upstream code: https://github.com/bastianwandt/DiffPose (MIT License).

## mmDiff and PointTransformer

mmDiff (Fan et al., ECCV 2024) refines the coarse pose of a PointTransformer backbone
with a graph-convolutional diffusion model conditioned on the backbone's per-joint
features, the past coarse poses and predicted limb lengths. The code in
`yofo/third_party/mmdiff/` is a re-implementation of the authors' code
(https://github.com/FanJunqiao/mmDiff, MIT, commit `29d7df2`) in the configuration they
use for sparse radar point clouds (no local radar conditioning). Module names follow the
original, and the implementation was checked against the authors' released MM-Fi
weights: coarse poses, features, noise predictions, limb lengths and both samplers give
identical outputs on the same inputs.

### Model (as in the original)

* **PointTransformer** (`point_transformer.py`): the points of the five input frames
  form one point set; a first transformer block at width 32 is followed by four stages
  at widths 64, 128, 256 and 512, each a pointwise MLP with batch normalisation and a
  point transformer block (vector attention over the 16 nearest neighbours). Learnable
  joint tokens are concatenated in front of the point features and mixed by a five-layer
  pre-norm transformer (width 512, 4 heads); the joint tokens are projected to a
  32-dimensional per-joint feature and from it to the coarse pose.
* **GCNdiff** (`gcndiff.py`, `graformer.py`): width 96, five graph attention layers
  (multi-head attention over the joints and a graph network with a learnable adjacency)
  alternating with five residual Chebyshev graph convolution blocks (three Chebyshev
  terms) into which the timestep embedding and the condition are added. The condition is
  the sum of the global embedding (graph convolution of the per-joint features) and the
  temporal embedding (graph convolution of every past coarse pose followed by two
  temporal convolutions and max pooling). The limb head predicts the limb lengths from
  the flattened pose history and current noisy pose and adds their embedding to the
  timestep embedding. The attention layers and graph networks are deep copies of one
  initialised module, as in the original. The skeleton graph and limb pairs are the COCO
  edges for MM-Fi and mRI (16 limbs) and the mmRadPose edges (25 limbs).
* **Diffusion** (`diffusion.py`): 51 training timesteps with a linear beta schedule from
  1e-4 to 1e-3 (ᾱ is 0.9999 at t=0, 0.99730 at t=12, 0.99213 at t=24, 0.97233 at t=50).

### Training (as in the original's `mmDiff.yml` and paper)

* Phase 1: Adam, learning rate 1e-4, weight decay 1e-8, gradient clipping 1.0, mean
  squared error on the root-centred pose of the last frame in metres, 5 epochs, batch 32.
* Feature extraction: coarse poses and features of the training and test windows in
  dataset order; coarse poses are centred on joint 0 as in the original's data generator.
* Phase 2: Adam, learning rate 2e-5 decayed by 0.9 at epoch 60, gradient clipping 1.0,
  100 epochs, batch 1024 (the value of the mmDiff paper; the yml of the code says 128);
  timesteps drawn in antithetic pairs (t, 50−t); loss = squared noise error summed over
  joints and coordinates and averaged over the batch, plus 10 times the L1 error of the
  limb lengths summed over the limbs. The temporal history of a window is the coarse
  poses of the six preceding windows in file order (the rule of the flow,
  `yofo.data.features.history_indices`); the original uses eight.
* Both phases keep the weights of the final epoch. The original tests after every epoch
  and keeps the best test epoch, and maintains an EMA of the weights that it never uses;
  neither is done here.

### Inference

* **PointTransformer**: the coarse pose of the backbone.
* **mmDiff (η=0)**: the deterministic sampler of the authors. The coarse pose is used as
  the state at timestep 24 without added noise and refined with DDIM (η=0) over the
  timesteps 24, 12 and 0 (`test_num_diffusion_timesteps` 25 and `test_timesteps` 2 of the
  original configuration), three denoiser calls.
* **mmDiff (η=1)**: the same start and the same horizon, every timestep from 24 to 0 with
  the stochastic DDIM update (η=1), 200 hypotheses per window drawn in parallel in the
  batch dimension; the hypothesis mean is the point estimate and the hypotheses give ECE,
  coverage and sharpness. The original has no stochastic variant.
* The global and temporal conditioning is computed once per window and shared by the
  hypotheses and the denoising steps; the denoiser output is identical to a plain forward
  pass. Outputs are evaluated as they are, like those of the other models; the original
  additionally subtracts the predicted root joint at test time.

### Settings of the experiments

Input windows, splits, test windows and ground-truth centring are those of the other
models of this repository (five aggregated frames of five raw frames each; 322, 320 and
320 points per aggregated frame on MM-Fi, mmRadPose and mRI). The leave-one-environment-out
splits of Table 2 (`loeo_E01` to `loeo_E04`) train on the thirty subjects of three
environments and test on the ten subjects of the fourth.

Project page: https://fanjunqiao.github.io/mmDiff-site/
