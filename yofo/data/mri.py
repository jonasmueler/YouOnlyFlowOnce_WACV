"""mRI radar point clouds as windowed sequences with frame aggregation.

Layout: ``<root>/radar/singleframe/subjectN.csv`` holds the radar points per camera
frame and ``<root>/pose_labels/subjectN_all_labels.cpl`` the refined 17-joint poses in
metres. Protocol 2 trains on the first 16 subjects in name order
(subject1 to subject5 and subject10 to subject20) and tests on subject6 to subject9.
"""

from __future__ import annotations

import csv
import logging
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 17
POINT_DIM = 5
NUM_TRAIN_SUBJECTS = 16


def detect_spike_frames(poses: np.ndarray, spike_thr: float = 0.3, median_kernel: int = 5) -> np.ndarray:
    """Flag frames whose annotation jumps (triangulation failures).

    The frame-to-frame joint velocity is compared with the velocity of a median-filtered
    copy; a transition whose excess exceeds ``spike_thr`` marks both of its frames.

    Args:
        poses: Poses ``(T, 3, 17)`` or ``(T, 17, 3)``.
        spike_thr: Velocity excess (metres per frame) above which a transition is a spike.
        median_kernel: Kernel size of the median filter.

    Returns:
        Boolean mask ``(T,)``, ``True`` for spike frames.
    """
    p = poses.transpose(0, 2, 1) if poses.shape[1] == 3 and poses.shape[2] == NUM_KEYPOINTS else poses
    desp = np.zeros_like(p)
    for j in range(p.shape[1]):
        for d in range(3):
            desp[:, j, d] = median_filter(p[:, j, d], size=median_kernel, mode="nearest")
    vel_raw = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    vel_desp = np.linalg.norm(np.diff(desp, axis=0), axis=-1)
    bad_transition = (vel_raw - vel_desp > spike_thr).any(axis=1)
    bad = np.zeros(p.shape[0], dtype=bool)
    bad[:-1] |= bad_transition
    bad[1:] |= bad_transition
    return bad


def smooth_poses_within_runs(
    poses: np.ndarray,
    bad_frames: np.ndarray,
    median_kernel: int = 5,
    savgol_window: int = 7,
    savgol_polyorder: int = 2,
) -> np.ndarray:
    """Smooth poses temporally inside consecutive spike-free runs.

    Args:
        poses: Poses ``(T, 3, 17)`` or ``(T, 17, 3)``.
        bad_frames: Spike mask ``(T,)``.
        median_kernel: Kernel size of the median filter.
        savgol_window: Window of the Savitzky-Golay filter.
        savgol_polyorder: Polynomial order of the Savitzky-Golay filter.

    Returns:
        Smoothed poses with the input shape; spike frames are unchanged.
    """
    transposed = poses.shape[1] == 3 and poses.shape[2] == NUM_KEYPOINTS
    p = poses.transpose(0, 2, 1).copy() if transposed else poses.copy()
    valid_idx = np.where(~bad_frames)[0]
    if len(valid_idx) == 0:
        return poses
    runs = np.split(valid_idx, np.where(np.diff(valid_idx) > 1)[0] + 1)
    for run in runs:
        if len(run) < savgol_window:
            continue
        seg = p[run]
        smoothed = np.zeros_like(seg)
        for j in range(seg.shape[1]):
            for d in range(3):
                smoothed[:, j, d] = savgol_filter(
                    median_filter(seg[:, j, d], size=median_kernel, mode="nearest"),
                    savgol_window,
                    savgol_polyorder,
                    mode="nearest",
                )
        p[run] = smoothed
    return p.transpose(0, 2, 1) if transposed else p


@dataclass(frozen=True)
class WindowMeta:
    """Metadata of one window.

    Attributes:
        subject: Subject name, for example ``"subject3"``.
        start_frame_idx: Index of the first frame in the subject's valid-frame list.
        end_frame_idx: Index of the last frame (inclusive).
    """

    subject: str
    start_frame_idx: int
    end_frame_idx: int


@dataclass
class _SubjectData:
    """Loaded frames of one subject."""

    radar_frames: list[np.ndarray]
    gt_indices: list[int]
    gt_kps: np.ndarray
    bad_frames: np.ndarray

    @property
    def num_frames(self) -> int:
        """Number of valid frames."""
        return len(self.radar_frames)


class MRIDataset(Dataset):
    """Sliding windows of aggregated mRI frames with the matching poses.

    Args:
        root_dir: Dataset root with ``radar/singleframe`` and ``pose_labels``.
        split: ``"train"`` or ``"test"``.
        sequence_length: Aggregated frames per window.
        frames_per_aggregate: Raw frames per aggregated frame.
        max_points_per_frame: Points per aggregated frame after padding or random subsampling.
        filter_extreme_poses: Drop frames with any coordinate above ``max_pose_value``.
        max_pose_value: Threshold of the extreme-pose filter in metres.
        clean_gt: Build windows only inside spike-free runs (annotation glitches removed).
        filter_jitter: Smooth the poses inside clean runs (requires ``clean_gt``).
        normalized: Centre every pose on joint 0.
        return_meta: Also return :class:`WindowMeta`.
    """

    def __init__(
        self,
        root_dir: Path,
        split: str,
        sequence_length: int = 5,
        frames_per_aggregate: int = 5,
        max_points_per_frame: int = 320,
        filter_extreme_poses: bool = True,
        max_pose_value: float = 5.0,
        clean_gt: bool = True,
        filter_jitter: bool = False,
        normalized: bool = True,
        return_meta: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root_dir).expanduser().resolve()
        self.split = split
        self.sequence_length = sequence_length
        self.frames_per_aggregate = frames_per_aggregate
        self.max_points_per_frame = max_points_per_frame
        self.filter_extreme_poses = filter_extreme_poses
        self.max_pose_value = max_pose_value
        self.clean_gt = clean_gt
        self.filter_jitter = filter_jitter
        self.normalized = normalized
        self.return_meta = return_meta

        radar_dir = self.root / "radar" / "singleframe"
        pose_dir = self.root / "pose_labels"
        if not radar_dir.is_dir() or not pose_dir.is_dir():
            raise FileNotFoundError(f"Expected {radar_dir} and {pose_dir}")
        all_subjects = [
            f.stem for f in sorted(radar_dir.glob("subject*.csv")) if (pose_dir / f"{f.stem}_all_labels.cpl").exists()
        ]
        if not all_subjects:
            raise RuntimeError(f"No subject pairs found in {self.root}")

        sorted_subjects = sorted(all_subjects)
        train_subjects = sorted_subjects[:NUM_TRAIN_SUBJECTS]
        test_subjects = sorted_subjects[NUM_TRAIN_SUBJECTS:]
        if split == "train":
            self.subjects = train_subjects
        elif split == "test":
            self.subjects = test_subjects
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train' or 'test'.")

        self._data: dict[str, _SubjectData] = {}
        for subject in all_subjects:
            if subject not in self.subjects:
                continue
            data = self._load_subject(subject)
            if data is not None:
                self._data[subject] = data
        self.windows = self._index_windows()
        logger.info("mRI protocol 2 %s: subjects %s, %d windows", split, self.subjects, len(self.windows))

    def _load_subject(self, subject: str) -> _SubjectData | None:
        """Load the radar frames and poses of one subject and align them by camera frame."""
        radar_path = self.root / "radar" / "singleframe" / f"{subject}.csv"
        pose_path = self.root / "pose_labels" / f"{subject}_all_labels.cpl"
        with open(pose_path, "rb") as f:
            pose_data = pickle.load(f)
        gt_kps = pose_data["refined_gt_kps"]
        if self.clean_gt:
            bad_frames_full = detect_spike_frames(gt_kps)
            if self.filter_jitter:
                gt_kps = smooth_poses_within_runs(gt_kps, bad_frames_full)

        camera_frame_groups: dict[int, list[list[float]]] = defaultdict(list)
        with open(radar_path, newline="") as f:
            for row in csv.DictReader(f):
                camera_frame_groups[int(row["Camera Frame"])].append(
                    [float(row["X"]), float(row["Y"]), float(row["Z"]), float(row["Doppler"]), float(row["Intensity"])]
                )

        radar_frames: list[np.ndarray] = []
        gt_indices: list[int] = []
        for camera_frame in sorted(camera_frame_groups):
            gt_idx = camera_frame
            if not 0 <= gt_idx < gt_kps.shape[0]:
                continue
            if np.allclose(gt_kps[gt_idx], 0):
                continue
            if self.filter_extreme_poses and np.abs(gt_kps[gt_idx].T).max() > self.max_pose_value:
                continue
            radar_frames.append(np.array(camera_frame_groups[camera_frame]))
            gt_indices.append(gt_idx)
        if not radar_frames:
            logger.warning("No valid frames for %s", subject)
            return None
        if self.clean_gt:
            bad_frames = np.array([bad_frames_full[gi] for gi in gt_indices], dtype=bool)
        else:
            bad_frames = np.zeros(len(radar_frames), dtype=bool)
        return _SubjectData(radar_frames, gt_indices, gt_kps, bad_frames)

    def _index_windows(self) -> list[WindowMeta]:
        """Windows with stride one for the subjects of the split."""
        windows: list[WindowMeta] = []
        w = self.sequence_length
        for subject, data in self._data.items():
            if subject not in self.subjects:
                continue
            if self.clean_gt:
                valid_idx = np.where(~data.bad_frames)[0]
                if len(valid_idx) == 0:
                    continue
                for run in np.split(valid_idx, np.where(np.diff(valid_idx) > 1)[0] + 1):
                    for i in range(len(run) - w + 1):
                        windows.append(WindowMeta(subject, int(run[i]), int(run[i + w - 1])))
            else:
                for start in range(0, data.num_frames - w + 1):
                    windows.append(WindowMeta(subject, start, start + w - 1))
        return windows

    def _fit_points(self, points: np.ndarray) -> np.ndarray:
        """Random subsample or zero-pad a point cloud to ``max_points_per_frame`` rows."""
        n = len(points)
        if n > self.max_points_per_frame:
            return points[np.random.choice(n, self.max_points_per_frame, replace=False)]
        if n < self.max_points_per_frame:
            return np.vstack([points, np.zeros((self.max_points_per_frame - n, POINT_DIM))])
        return points

    def _aggregate(self, data: _SubjectData, start: int, count: int) -> np.ndarray:
        """Concatenate ``count`` raw frames from ``start`` and fit them to the point budget."""
        parts = [data.radar_frames[i] for i in range(start, start + count) if i < data.num_frames and data.radar_frames[i].shape[0] > 0]
        if not parts:
            return self._fit_points(np.zeros((1, POINT_DIM), dtype=np.float64))
        return self._fit_points(np.concatenate(parts, axis=0))

    @staticmethod
    def _pose(data: _SubjectData, frame_idx: int) -> np.ndarray:
        """Pose of a valid frame as ``(17, 3)`` in centimetres."""
        return data.gt_kps[data.gt_indices[frame_idx]].T * 100.0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        """Return ``(radar, pose)`` or ``(radar, pose, meta)`` for one window.

        Args:
            idx: Window index.

        Returns:
            Radar ``(sequence_length, max_points_per_frame, 5)`` and poses
            ``(sequence_length, 17, 3)`` in centimetres as float tensors.
        """
        win = self.windows[idx]
        data = self._data[win.subject]
        radar_seq = []
        pose_seq = []
        for frame_idx in range(win.start_frame_idx, win.end_frame_idx + 1):
            if self.frames_per_aggregate > 1:
                agg_start = max(0, frame_idx - self.frames_per_aggregate + 1)
                radar_seq.append(self._aggregate(data, agg_start, frame_idx - agg_start + 1))
            else:
                radar_seq.append(self._fit_points(data.radar_frames[frame_idx]))
            pose_seq.append(self._pose(data, frame_idx))
        radar = np.stack(radar_seq, axis=0)
        pose = np.stack(pose_seq, axis=0)
        if self.normalized:
            pose = pose - pose[:, 0:1, :]
        radar_t = torch.from_numpy(radar).float()
        pose_t = torch.from_numpy(pose).float()
        if self.return_meta:
            return radar_t, pose_t, win
        return radar_t, pose_t
