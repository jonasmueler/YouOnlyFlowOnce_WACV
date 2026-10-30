"""The configuration defaults equal the values reported in the paper."""

from yofo.config import (
    BACKBONE_EPOCHS,
    FLOW_EPOCHS,
    MMFI,
    MMRADPOSE,
    MRI,
    PRIOR_EPOCHS,
    BackboneConfig,
    DiffPoseConfig,
    EvalConfig,
    FlowConfig,
    MmDiffConfig,
    resolve_dataset,
    resolve_split,
)


def test_epochs_match_paper() -> None:
    assert BACKBONE_EPOCHS == {"mmFI": 5, "mmRadPose": 15, "MRI": 19}
    assert FLOW_EPOCHS == {"mmFI": 10, "mmRadPose": 25, "MRI": 6}
    assert BackboneConfig().epochs_for(MMFI) == 5
    assert FlowConfig().epochs_for(MRI) == 6
    assert PRIOR_EPOCHS == {"mmFI": 8, "mmRadPose": 15, "MRI": 16}
    assert BackboneConfig(standalone=True).epochs_for(MMFI) == 8
    assert BackboneConfig(standalone=True, epochs=3).epochs_for(MMFI) == 3


def test_dataset_shapes() -> None:
    assert (MMFI.num_keypoints, MMFI.input_dim, MMFI.max_points_per_frame) == (17, 5, 322)
    assert (MMRADPOSE.num_keypoints, MMRADPOSE.input_dim, MMRADPOSE.max_points_per_frame) == (26, 7, 320)
    assert (MRI.num_keypoints, MRI.input_dim, MRI.max_points_per_frame) == (17, 5, 320)
    for dataset in (MMFI, MMRADPOSE, MRI):
        assert dataset.seq_len == 5 and dataset.frames_per_aggregate == 5 and dataset.center_gt_to_root


def test_mmdiff_hyperparameters() -> None:
    cfg = MmDiffConfig()
    assert (cfg.backbone_epochs, cfg.backbone_batch_size, cfg.backbone_lr, cfg.backbone_weight_decay) == (5, 32, 1e-4, 1e-8)
    assert (cfg.feature_dim, cfg.nblocks, cfg.nneighbor, cfg.transformer_dim) == (32, 5, 16, 128)
    assert (cfg.hid_dim, cfg.num_layers, cfg.n_head, cfg.dropout, cfg.past_frames) == (96, 5, 4, 0.25, 6)
    assert (cfg.timesteps, cfg.beta_start, cfg.beta_end) == (51, 1e-4, 1e-3)
    assert (cfg.epochs, cfg.batch_size, cfg.lr, cfg.lr_gamma, cfg.lr_decay_epochs, cfg.limb_loss_weight) == (100, 1024, 2e-5, 0.9, 60, 10.0)
    assert (cfg.test_horizon, cfg.eta0_steps, cfg.eta1_steps, cfg.n_hypotheses, cfg.grad_clip) == (25, 2, 25, 200, 1.0)
    assert (MMFI.pose_scale_to_m, MMRADPOSE.pose_scale_to_m, MRI.pose_scale_to_m) == (1.0, 0.01, 0.01)


def test_splits_and_participants() -> None:
    assert MMFI.splits == ("cross_subject", "cross_scene", "random", "loeo_E01", "loeo_E02", "loeo_E03", "loeo_E04")
    assert len(MMRADPOSE.train_participants) == 10 and MRI.max_points_per_frame == 320
    assert MMRADPOSE.train_participants == ("p1", "p2", "p4", "p5", "p6", "p7", "p8", "p10", "p11", "p12")
    assert MMRADPOSE.test_participants == ("p3", "p9")
    assert MRI.mri_clean_gt and not MRI.mri_filter_jitter and MRI.mri_filter_extreme_poses


def test_training_hyperparameters() -> None:
    backbone = BackboneConfig()
    assert (backbone.lr, backbone.weight_decay, backbone.beta, backbone.gamma) == (1e-4, 1e-6, 15.0, 1.0)
    assert (backbone.n_samples_train, backbone.n_samples_extract) == (128, 512)
    assert (backbone.d_model, backbone.num_heads, backbone.num_layers, backbone.dim_feedforward) == (512, 8, 4, 1024)
    flow = FlowConfig()
    assert (flow.flow_num_layers, flow.flow_hidden, flow.cond_dim, flow.past_frames) == (8, 256, 256, 6)
    assert (flow.lr, flow.weight_decay, flow.batch_size, flow.n_samples_test) == (1e-4, 1e-6, 32, 200)
    diffpose = DiffPoseConfig()
    assert (diffpose.timesteps, diffpose.num_samples, diffpose.n_hypotheses, diffpose.epochs) == (25, 32, 200, 100)
    assert EvalConfig().n_samples == 200 and EvalConfig().coverage_levels == (0.30, 0.50, 0.90, 0.95)


def test_resolve() -> None:
    assert resolve_dataset("mmFI") is MMFI and resolve_dataset("MRI") is MRI and resolve_dataset("mmRadPose") is MMRADPOSE
    assert resolve_split(MMFI, None) == "cross_subject"
    assert resolve_split(MMRADPOSE, None) == "default"
