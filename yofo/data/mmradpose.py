"""mmRadPose radar point clouds as windowed sequences with frame aggregation.

Layout: ``<root>/pX/angleY/<exercise>/<leaf>/targetlist_64.npy`` holds the radar frames
``(T, 64, 7)`` and ``skeleton.npy`` the poses ``(T, 26, 3)`` or ``(T, 78)`` in metres.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

NUM_KEYPOINTS = 26


@dataclass(frozen=True)
class SequenceMeta:
    """One recorded sequence of the dataset."""

    participant: str
    angle: str
    exercise: str
    leaf: str
    x_path: Path
    y_path: Path


@dataclass(frozen=True)
class WindowMeta:
    """Metadata of one window.

    Attributes:
        participant: Participant ID, for example ``"p3"``.
        angle: Radar aspect angle folder, for example ``"angle0"``.
        exercise: Exercise ID.
        leaf: Remaining path below the exercise folder.
        seq_idx: Index of the parent sequence.
        start: Raw index of the first aggregated frame.
        end: Raw index of the last aggregated frame (the target frame).
    """

    participant: str
    angle: str
    exercise: str
    leaf: str
    seq_idx: int
    start: int
    end: int


class RadarPoseSequenceDataset(Dataset):
    """Sliding windows of aggregated mmRadPose frames with the matching poses.

    Args:
        root_dir: Dataset root (``mmRadPose_pointclouds`` or its parent).
        participants: Participants to load.
        time_window: Aggregated frames per window.
        frames_per_aggregate: Raw frames per aggregated frame.
        max_points: Points per aggregated frame after padding or truncation.
        exclude_exercises: Exercise IDs to skip.
        normalized: Centre every pose on joint 0.
        return_meta: Also return :class:`WindowMeta`.
    """

    def __init__(
        self,
        root_dir: Path,
        participants: tuple[str, ...],
        time_window: int = 5,
        frames_per_aggregate: int = 5,
        max_points: int = 320,
        exclude_exercises: tuple[str, ...] = (),
        normalized: bool = True,
        return_meta: bool = False,
    ) -> None:
        super().__init__()
        root = Path(root_dir).expanduser().resolve()
        if (root / "mmRadPose_pointclouds").is_dir():
            root = root / "mmRadPose_pointclouds"
        if not root.is_dir():
            raise FileNotFoundError(f"mmRadPose root not found: {root}")
        self.root = root
        self.time_window = time_window
        self.frames_per_aggregate = frames_per_aggregate
        self.max_points = max_points
        self.normalized = normalized
        self.return_meta = return_meta

        sequences = self._index_sequences()
        wanted = set(participants)
        sequences = [s for s in sequences if s.participant in wanted]
        excluded = set(exclude_exercises)
        sequences = [s for s in sequences if s.exercise not in excluded]
        if not sequences:
            raise RuntimeError(f"No mmRadPose sequences under {root} for participants {sorted(wanted)}.")
        self.sequences = sequences

        self._x: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        for seq in self.sequences:
            x, y = self._load_sequence(seq)
            self._x.append(x)
            self._y.append(y)
        self.windows = self._index_windows()
        logger.info("mmRadPose %s: %d sequences, %d windows", sorted(wanted), len(self.sequences), len(self.windows))

    def _index_sequences(self) -> list[SequenceMeta]:
        """Find every ``skeleton.npy`` with its ``targetlist_64.npy`` and sort them."""
        sequences: list[SequenceMeta] = []
        for sk in self.root.rglob("skeleton.npy"):
            tl = sk.parent / "targetlist_64.npy"
            if not tl.exists():
                raise FileNotFoundError(f"Missing targetlist_64.npy next to {sk}")
            parts = sk.parent.relative_to(self.root).parts
            if len(parts) < 3:
                raise ValueError(f"Unexpected path depth: {sk.parent}")
            sequences.append(
                SequenceMeta(
                    participant=parts[0],
                    angle=parts[1],
                    exercise=parts[2],
                    leaf="/".join(parts[3:]),
                    x_path=tl,
                    y_path=sk,
                )
            )
        if not sequences:
            raise RuntimeError(f"No sequence pairs found under {self.root}")
        sequences.sort(key=lambda m: (m.participant, m.angle, m.exercise, m.leaf, str(m.x_path)))
        return sequences

    @staticmethod
    def _load_sequence(seq: SequenceMeta) -> tuple[np.ndarray, np.ndarray]:
        """Load radar frames and poses; poses are returned in centimetres."""
        x = np.load(seq.x_path)
        y = np.load(seq.y_path)
        if y.ndim == 2 and y.shape[1] == NUM_KEYPOINTS * 3:
            y = y.reshape(y.shape[0], NUM_KEYPOINTS, 3)
        return x, y * 100.0

    def _index_windows(self) -> list[WindowMeta]:
        """Windows with stride one, starting where full aggregation history exists."""
        windows: list[WindowMeta] = []
        min_start = self.frames_per_aggregate - 1 if self.frames_per_aggregate > 1 else 0
        for seq_idx, seq in enumerate(self.sequences):
            num_frames = self._x[seq_idx].shape[0]
            if num_frames < min_start + self.time_window:
                continue
            for start in range(min_start, num_frames - self.time_window + 1):
                windows.append(
                    WindowMeta(
                        participant=seq.participant,
                        angle=seq.angle,
                        exercise=seq.exercise,
                        leaf=seq.leaf,
                        seq_idx=seq_idx,
                        start=start,
                        end=start + self.time_window - 1,
                    )
                )
        return windows

    def _pad_or_truncate(self, points: np.ndarray) -> np.ndarray:
        """Bring a point cloud to exactly ``max_points`` rows."""
        n = points.shape[0]
        if n == self.max_points:
            return points
        if n > self.max_points:
            return points[: self.max_points]
        padding = np.zeros((self.max_points - n, points.shape[1]), dtype=points.dtype)
        return np.concatenate([points, padding], axis=0)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        """Return ``(radar, pose)`` or ``(radar, pose, meta)`` for one window.

        Args:
            idx: Window index.

        Returns:
            Radar ``(time_window, max_points, 7)`` and poses ``(time_window, 26, 3)``
            in centimetres as float tensors.
        """
        win = self.windows[idx]
        x_seq = self._x[win.seq_idx]
        y_seq = self._y[win.seq_idx]
        if self.frames_per_aggregate > 1:
            radar_list = []
            pose_list = []
            for offset in range(self.time_window):
                raw_idx = win.start + offset
                agg_start = max(0, raw_idx - self.frames_per_aggregate + 1)
                frames = x_seq[agg_start : raw_idx + 1]
                radar_frame = frames.reshape(-1, frames.shape[-1])
                radar_list.append(self._pad_or_truncate(radar_frame))
                pose_list.append(y_seq[raw_idx])
            x_window = np.stack(radar_list, axis=0)
            y_target = np.stack(pose_list, axis=0)
        else:
            x_window = x_seq[win.start : win.end + 1]
            y_target = y_seq[win.start : win.end + 1]
        if self.normalized:
            y_target = y_target - y_target[:, 0:1, :]
        radar = torch.from_numpy(x_window).float()
        pose = torch.from_numpy(y_target).float()
        if self.return_meta:
            return radar, pose, win
        return radar, pose
