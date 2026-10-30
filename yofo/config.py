"""Configuration of datasets, models and paths.

Scripts build their configuration from these dataclasses and command-line arguments;
nothing is read from environment variables at import time.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GLOBAL_SEED = 42
"""Seed used for every run of the paper."""

DETERMINISTIC = True
"""Deterministic CUDA algorithms are enabled by default."""

DATA_ROOT_ENV = "YOFO_DATA_ROOT"
"""Environment variable that may hold the dataset root; ``PATHRAW`` is a legacy alias."""

LEGACY_DATA_ROOT_ENV = "PATHRAW"


@dataclass(frozen=True)
class DatasetConfig:
    """Static description of one dataset.

    Attributes:
        name: Dataset name as spelled in run folders and feature files.
        key: Lower-case command-line name.
        num_keypoints: Number of skeleton joints.
        input_dim: Features per radar point.
        max_points_per_frame: Points per aggregated frame after padding or truncation.
        splits: Split names accepted on the command line.
        default_split: Split used when none is given.
        skeleton: Skeleton edge set, ``"coco"`` or ``"mmradpose"``.
        seq_len: Aggregated frames per model input.
        frames_per_aggregate: Raw radar frames concatenated into one aggregated frame.
        center_gt_to_root: Whether ground-truth poses are centred on joint 0.
        gt_scale_to_cm: Factor that converts loader ground truth to centimetres.
        batch_size: Batch size for phase-1 training.
        extraction_batch_size: Batch size for feature extraction.
        mmfi_protocol: MM-Fi activity protocol (3 = all activities).
        mmfi_seed: Seed of the MM-Fi random split.
        train_participants: mmRadPose training participants.
        test_participants: mmRadPose test participants.
        exclude_exercises: mmRadPose exercises left out (the static T-pose).
        mri_protocol: mRI protocol (2 = participant split, 16 train and 4 test).
        mri_seed: Seed recorded for the mRI loader.
        mri_clean_gt: Drop annotation glitches (pose jumps) from the mRI windows.
        mri_filter_jitter: Smooth mRI ground truth inside clean runs.
        mri_filter_extreme_poses: Drop mRI frames with coordinates above ``mri_max_pose_value``.
        mri_max_pose_value: Threshold in metres for the extreme-pose filter.
    """

    name: str
    key: str
    num_keypoints: int
    input_dim: int
    max_points_per_frame: int
    splits: tuple[str, ...]
    default_split: str
    skeleton: str
    seq_len: int = 5
    frames_per_aggregate: int = 5
    center_gt_to_root: bool = True
    gt_scale_to_cm: float = 1.0
    batch_size: int = 32
    extraction_batch_size: int = 32
    mmfi_protocol: int = 3
    mmfi_seed: int = 0
    train_participants: tuple[str, ...] = ()
    test_participants: tuple[str, ...] = ()
    exclude_exercises: tuple[str, ...] = ()
    mri_protocol: int = 2
    mri_seed: int = 42
    mri_clean_gt: bool = True
    mri_filter_jitter: bool = False
    mri_filter_extreme_poses: bool = True
    mri_max_pose_value: float = 5.0

    @property
    def output_dim(self) -> int:
        """Flattened pose dimension (joints times three)."""
        return self.num_keypoints * 3

    @property
    def pose_scale_to_m(self) -> float:
        """Factor that converts loader ground truth to metres (the unit of the mmDiff baseline)."""
        return self.gt_scale_to_cm / 100.0


MMFI = DatasetConfig(
    name="mmFI",
    key="mmfi",
    num_keypoints=17,
    input_dim=5,
    max_points_per_frame=322,
    splits=("cross_subject", "cross_scene", "random", "loeo_E01", "loeo_E02", "loeo_E03", "loeo_E04"),
    default_split="cross_subject",
    skeleton="coco",
    gt_scale_to_cm=100.0,
)

MMRADPOSE = DatasetConfig(
    name="mmRadPose",
    key="mmradpose",
    num_keypoints=26,
    input_dim=7,
    max_points_per_frame=320,
    splits=("default",),
    default_split="default",
    skeleton="mmradpose",
    train_participants=("p1", "p2", "p4", "p5", "p6", "p7", "p8", "p10", "p11", "p12"),
    test_participants=("p3", "p9"),
    exclude_exercises=("0",),
)

MRI = DatasetConfig(
    name="MRI",
    key="mri",
    num_keypoints=17,
    input_dim=5,
    max_points_per_frame=320,
    splits=("protocol2",),
    default_split="protocol2",
    skeleton="coco",
    extraction_batch_size=64,
)

DATASETS: dict[str, DatasetConfig] = {d.key: d for d in (MMFI, MMRADPOSE, MRI)}
"""Datasets by command-line key."""

_DATASET_ALIASES = {
    "mmfi": "mmfi",
    "mm-fi": "mmfi",
    "mmradpose": "mmradpose",
    "mri": "mri",
}

BACKBONE_EPOCHS = {"mmFI": 5, "mmRadPose": 15, "MRI": 19}
"""Phase-1 epochs per dataset for the backbone the flow is trained on."""

PRIOR_EPOCHS = {"mmFI": 8, "mmRadPose": 15, "MRI": 16}
"""Phase-1 epochs per dataset when the MG-Prior is trained as a standalone model."""

FLOW_EPOCHS = {"mmFI": 10, "mmRadPose": 25, "MRI": 6}
"""Phase-2 epochs per dataset."""


def resolve_dataset(name: str) -> DatasetConfig:
    """Look up a dataset by any accepted spelling.

    Args:
        name: ``mmfi``, ``mmFI``, ``mmradpose``, ``mmRadPose``, ``mri`` or ``MRI``.

    Returns:
        The matching dataset configuration.

    Raises:
        ValueError: If the name is unknown.
    """
    key = _DATASET_ALIASES.get(name.lower())
    if key is None:
        raise ValueError(f"Unknown dataset '{name}'. Choose one of {sorted(DATASETS)}.")
    return DATASETS[key]


def resolve_split(dataset: DatasetConfig, split: str | None) -> str:
    """Validate a split name for a dataset.

    Args:
        dataset: Dataset configuration.
        split: Requested split or ``None`` for the default.

    Returns:
        The validated split name.

    Raises:
        ValueError: If the split does not exist for the dataset.
    """
    if split is None:
        return dataset.default_split
    if split not in dataset.splits:
        raise ValueError(
            f"Split '{split}' is not defined for {dataset.name}; choose one of {dataset.splits}."
        )
    return split


@dataclass(frozen=True)
class BackboneConfig:
    """MG-Prior: spatiotemporal transformer with the GCN Gaussian head (phase 1).

    Attributes:
        epochs: Training epochs; ``None`` selects the per-dataset value.
        standalone: Train the MG-Prior as a standalone model (own schedule)
            instead of as the backbone of the flow.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        betas: AdamW betas.
        lr_decay: Multiplicative learning-rate decay per epoch.
        grad_clip: Gradient-norm clipping.
        beta: Weight of the KL term.
        gamma: Weight of the variance penalty in the Gaussian NLL.
        n_samples_train: Latent samples per forward pass during training.
        n_samples_extract: Latent samples used when features are extracted.
        d_model: Transformer width.
        num_heads: Attention heads.
        num_layers: Transformer blocks.
        dim_feedforward: MLP width inside each block.
        dropout: Dropout inside the transformer.
        feat_dim: Backbone output (feature) dimension.
        d_latent: Dimension of the latent Gaussian.
        gcn_hidden: Hidden width of the GCN refinement.
        gcn_k: Chebyshev polynomial order of the GCN.
        cov_eps: Jitter added to the sample covariance diagonal.
        num_workers: Data-loader workers for training.
        batch_size: Training batch size; ``None`` selects the dataset's value.
    """

    epochs: int | None = None
    standalone: bool = False
    lr: float = 1e-4
    weight_decay: float = 1e-6
    betas: tuple[float, float] = (0.9, 0.999)
    lr_decay: float = 0.99
    grad_clip: float = 1.0
    beta: float = 15.0
    gamma: float = 1.0
    n_samples_train: int = 128
    n_samples_extract: int = 512
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    feat_dim: int = 512
    d_latent: int = 256
    gcn_hidden: int = 128
    gcn_k: int = 2
    cov_eps: float = 1e-4
    num_workers: int = 1
    batch_size: int | None = None

    def batch_size_for(self, dataset: DatasetConfig) -> int:
        """Return the training batch size for a dataset."""
        return self.batch_size if self.batch_size is not None else dataset.batch_size

    def epochs_for(self, dataset: DatasetConfig) -> int:
        """Return the number of training epochs for a dataset.

        An explicit ``epochs`` wins; otherwise the standalone or the flow-backbone
        schedule of the dataset is used.
        """
        if self.epochs is not None:
            return self.epochs
        schedule = PRIOR_EPOCHS if self.standalone else BACKBONE_EPOCHS
        return schedule[dataset.name]


@dataclass(frozen=True)
class FlowConfig:
    """NFPG: conditional RealNVP flow with graph conditioning (phase 2).

    Attributes:
        epochs: Training epochs; ``None`` selects the per-dataset value.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        batch_size: Training batch size.
        grad_clip: Gradient-norm clipping.
        feat_dim: Backbone feature dimension consumed by the conditioning module.
        hid_dim: GCN hidden width inside the conditioning module.
        global_feat_dim: Per-keypoint width after projecting the backbone feature.
        past_frames: Past backbone predictions used as temporal context.
        cond_dim: Dimension of the conditioning vector.
        flow_num_layers: Affine coupling layers.
        flow_hidden: Hidden width of the coupling networks.
        dropout: Dropout inside the coupling and conditioning networks.
        temporal_fidelity: Full-width temporal convolution instead of the bottleneck.
        n_samples_test: Hypotheses drawn at evaluation time.
        num_workers: Data-loader workers for training.
    """

    epochs: int | None = None
    lr: float = 1e-4
    weight_decay: float = 1e-6
    batch_size: int = 32
    grad_clip: float = 1.0
    feat_dim: int = 512
    hid_dim: int = 96
    global_feat_dim: int = 64
    past_frames: int = 6
    cond_dim: int = 256
    flow_num_layers: int = 8
    flow_hidden: int = 256
    dropout: float = 0.1
    temporal_fidelity: bool = False
    n_samples_test: int = 200
    num_workers: int = 4

    def epochs_for(self, dataset: DatasetConfig) -> int:
        """Return the number of training epochs for a dataset."""
        return self.epochs if self.epochs is not None else FLOW_EPOCHS[dataset.name]


@dataclass(frozen=True)
class DiffPoseConfig:
    """MG-Prior + DiffPose baseline: diffusion model conditioned on MG-Prior samples.

    Attributes:
        timesteps: Diffusion steps.
        beta_schedule: Noise schedule.
        scaling_3d_pose: Pose scaling applied before diffusion.
        clip_denoised: Clip the denoised estimate.
        noise_scale: Scale of the sampling noise.
        loss_type: Training loss.
        objective: Prediction target of the denoiser.
        condition_dim: Width of the conditioning embedding.
        embedding_dim: Width of the pose embedding.
        num_samples: Gaussian samples drawn from the MG-Prior per pose.
        positional_embedding: Positional embedding of the joint-wise transformer.
        n_heads: Attention heads of the joint-wise transformer.
        n_layers: Layers of the joint-wise transformer.
        feedforward_dim: MLP width of the joint-wise transformer.
        backbone_feature_dim: Backbone feature dimension.
        backbone_cond_hidden_dim: Hidden width of the backbone-feature projection.
        backbone_cond_output_dim: Output width of the backbone-feature projection.
        hidden_dim: Denoiser width.
        use_batchnorm: Batch normalisation inside the denoiser.
        batch_size: Training batch size.
        test_batch_size: Evaluation batch size.
        epochs: Training epochs.
        lr: Learning rate.
        grad_clip: Gradient-norm clipping.
        ema: Keep an exponential moving average of the weights.
        ema_rate: EMA decay.
        n_hypotheses: Hypotheses drawn at evaluation time.
        max_hypotheses_per_batch: Hypotheses sampled per chunk at evaluation time.
        past_frames: Past backbone predictions in the feature files.
        num_workers: Data-loader workers.
    """

    timesteps: int = 25
    beta_schedule: str = "cosine"
    scaling_3d_pose: float = 10.0
    clip_denoised: bool = True
    noise_scale: float = 1.0
    loss_type: str = "l2"
    objective: str = "pred_noise"
    condition_dim: int = 128
    embedding_dim: int = 64
    num_samples: int = 32
    positional_embedding: str = "sin"
    n_heads: int = 4
    n_layers: int = 4
    feedforward_dim: int = 512
    backbone_feature_dim: int = 512
    backbone_cond_hidden_dim: int = 256
    backbone_cond_output_dim: int = 256
    hidden_dim: int = 256
    use_batchnorm: bool = False
    batch_size: int = 128
    test_batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    grad_clip: float = 1.0
    ema: bool = True
    ema_rate: float = 0.995
    n_hypotheses: int = 200
    max_hypotheses_per_batch: int = 32
    past_frames: int = 6
    num_workers: int = 4


@dataclass(frozen=True)
class MmDiffConfig:
    """mmDiff baseline: PointTransformer backbone and GCNdiff denoiser.

    Attributes:
        backbone_epochs: Phase-1 epochs.
        backbone_batch_size: Phase-1 batch size.
        backbone_lr: Phase-1 learning rate (Adam).
        backbone_weight_decay: Phase-1 weight decay.
        grad_clip: Gradient-norm clipping in both phases.
        feature_dim: Width of the per-joint backbone feature.
        nblocks: Point transformer blocks.
        nneighbor: Neighbours per point.
        transformer_dim: Attention width of the point transformer blocks.
        hid_dim: Width of the graph layers of the denoiser.
        num_layers: Graph attention layers of the denoiser.
        n_head: Attention heads of the denoiser.
        dropout: Dropout of the attention residual connections.
        past_frames: Past coarse poses in the temporal context.
        timesteps: Diffusion steps of the training schedule.
        beta_start: First beta of the linear schedule.
        beta_end: Last beta of the linear schedule.
        epochs: Phase-2 epochs.
        batch_size: Phase-2 batch size.
        lr: Phase-2 learning rate (Adam).
        lr_gamma: Learning-rate factor applied every ``lr_decay_epochs`` epochs.
        lr_decay_epochs: Period of the learning-rate decay.
        limb_loss_weight: Weight of the limb-length loss.
        test_horizon: Timesteps of the inference schedule (the sampler starts at
            ``test_horizon - 1`` from the coarse pose).
        eta0_steps: Requested steps of the deterministic sampler; with the uniform skip
            ``test_horizon // eta0_steps`` this visits the timesteps 24, 12 and 0.
        eta1_steps: Steps of the stochastic sampler (every timestep of the horizon).
        n_hypotheses: Hypotheses drawn by the stochastic sampler.
        test_batch_size: Windows per evaluation batch.
        max_rows_per_forward: Rows (windows times hypotheses) per denoiser forward pass.
        num_workers: Data-loader workers.
    """

    backbone_epochs: int = 5
    backbone_batch_size: int = 32
    backbone_lr: float = 1e-4
    backbone_weight_decay: float = 1e-8
    grad_clip: float = 1.0
    feature_dim: int = 32
    nblocks: int = 5
    nneighbor: int = 16
    transformer_dim: int = 128
    hid_dim: int = 96
    num_layers: int = 5
    n_head: int = 4
    dropout: float = 0.25
    past_frames: int = 6
    timesteps: int = 51
    beta_start: float = 1e-4
    beta_end: float = 1e-3
    epochs: int = 100
    batch_size: int = 1024
    lr: float = 2e-5
    lr_gamma: float = 0.9
    lr_decay_epochs: int = 60
    limb_loss_weight: float = 10.0
    test_horizon: int = 25
    eta0_steps: int = 2
    eta1_steps: int = 25
    n_hypotheses: int = 200
    test_batch_size: int = 64
    max_rows_per_forward: int = 12800
    num_workers: int = 4


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation protocol shared by all models.

    Attributes:
        batch_size: Evaluation batch size.
        num_workers: Data-loader workers.
        n_samples: Hypotheses per frame for every probabilistic model.
        coverage_levels: Credible-interval levels reported as coverage.
    """

    batch_size: int = 64
    num_workers: int = 4
    n_samples: int = 200
    coverage_levels: tuple[float, ...] = (0.30, 0.50, 0.90, 0.95)


@dataclass(frozen=True)
class Paths:
    """Root folders used by the scripts.

    Attributes:
        data_root: Dataset folder.
        feature_root: Folder with extracted features.
        run_root: Folder with training runs.
        results_root: Folder with evaluation results.
        checkpoint_root: Folder with downloaded checkpoints.
    """

    data_root: Path
    feature_root: Path = Path("features")
    run_root: Path = Path("runs")
    results_root: Path = Path("results")
    checkpoint_root: Path = Path("checkpoints")


def default_data_root() -> Path | None:
    """Return the dataset root from the environment, if set."""
    for variable in (DATA_ROOT_ENV, LEGACY_DATA_ROOT_ENV):
        value = os.environ.get(variable)
        if value:
            return Path(value)
    return None


def feature_dir(paths: Paths, dataset: DatasetConfig, split: str) -> Path:
    """Return the folder holding ``train_features.pt`` and ``test_features.pt``."""
    return paths.feature_root / dataset.name / split


def run_dir(paths: Paths, dataset: DatasetConfig, split: str, model: str) -> Path:
    """Return the folder under which ``run_NNN`` folders of a model are created."""
    return paths.run_root / dataset.name / split / model


def checkpoint_path(paths: Paths, dataset: DatasetConfig, split: str) -> Path:
    """Return the default location of a released MG-Prior checkpoint."""
    return paths.checkpoint_root / dataset.key / split / "gaussian" / "backbone.pt"


def to_dict(*configs: Any) -> dict[str, Any]:
    """Serialise dataclass instances into one JSON-friendly dictionary.

    Args:
        *configs: Dataclass instances; each is stored under its class name.

    Returns:
        Dictionary of dictionaries with paths converted to strings.
    """
    out: dict[str, Any] = {}
    for cfg in configs:
        fields = dataclasses.asdict(cfg)
        out[type(cfg).__name__] = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in fields.items()
        }
    return out
