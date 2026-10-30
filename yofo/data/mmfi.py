"""MM-Fi radar point clouds as windowed sequences with frame aggregation.

Reading of the raw files and the subject splits follow the official MM-Fi toolbox
(https://github.com/ybhbingo/MMFi_dataset): every sequence folder
``<root>/<scene>/<subject>/<action>/mmwave/frameNNN.bin`` holds 297 frames of
``float64`` points with five values each (x, y, z, Doppler, intensity), and
``ground_truth.npy`` holds the 297 poses with 17 COCO joints in metres.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

FRAMES_PER_SEQUENCE = 297
POINT_DIM = 5

ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 41)]
ALL_ACTIONS = [f"A{i:02d}" for i in range(1, 28)]
DAILY_ACTIONS = [
    "A02", "A03", "A04", "A05", "A13", "A14", "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A27",
]
REHABILITATION_ACTIONS = [
    "A01", "A06", "A07", "A08", "A09", "A10", "A11", "A12", "A15", "A16", "A24", "A25", "A26",
]

CROSS_SUBJECT_TRAIN = [
    "S01", "S02", "S03", "S04", "S06", "S07", "S08", "S09", "S11", "S12", "S13", "S14",
    "S16", "S17", "S18", "S19", "S21", "S22", "S23", "S24", "S26", "S27", "S28", "S29",
    "S31", "S32", "S33", "S34", "S36", "S37", "S38", "S39",
]
CROSS_SUBJECT_TEST = ["S05", "S10", "S15", "S20", "S25", "S30", "S35", "S40"]
CROSS_SCENE_TRAIN = [f"S{i:02d}" for i in range(1, 31)]
CROSS_SCENE_TEST = [f"S{i:02d}" for i in range(31, 41)]
RANDOM_SPLIT_RATIO = 0.8
SCENES = ["E01", "E02", "E03", "E04"]
LOEO_PREFIX = "loeo_"
"""Leave-one-environment-out splits are named ``loeo_E01`` to ``loeo_E04``."""


def scene_subjects(scene: str) -> list[str]:
    """The ten subjects recorded in an environment, in subject order."""
    if scene not in SCENES:
        raise ValueError(f"Unknown MM-Fi scene '{scene}'. Choose one of {SCENES}.")
    first = SCENES.index(scene) * 10 + 1
    return [f"S{i:02d}" for i in range(first, first + 10)]


@dataclass
class MMFiSampleMeta:
    """Metadata of one MM-Fi window.

    Attributes:
        subject: Subject ID, for example ``"S01"``.
        scene: Environment ID ``"E01"`` to ``"E04"``.
        action: Action ID, for example ``"A01"``.
        seq_idx: Index of the sequence in the split.
        window_idx: Index of the window in the dataset.
    """

    subject: str
    scene: str
    action: str
    seq_idx: int
    window_idx: int


def scene_of(subject: str) -> str:
    """Environment in which a subject was recorded (ten subjects per environment)."""
    number = int(subject[1:])
    if 1 <= number <= 40:
        return f"E{(number - 1) // 10 + 1:02d}"
    raise ValueError(f"Subject {subject} does not exist in MM-Fi.")


def protocol_actions(protocol: int) -> list[str]:
    """Actions of an MM-Fi protocol (1 daily, 2 rehabilitation, 3 all)."""
    if protocol == 1:
        return DAILY_ACTIONS
    if protocol == 2:
        return REHABILITATION_ACTIONS
    return ALL_ACTIONS


def split_forms(protocol: int, split_strategy: str, seed: int) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Subject-to-actions mapping of the training and test parts of a split.

    Reproduces ``decode_config`` of the MM-Fi toolbox, including the per-action
    reseeded permutation of the random split and the insertion order of the subjects.
    The leave-one-environment-out splits ``loeo_E0x`` train on the thirty subjects of
    the other three environments, ordered by environment, and test on the ten subjects
    of the held-out environment.

    Args:
        protocol: Activity protocol.
        split_strategy: ``"cross_subject"``, ``"cross_scene"``, ``"random"`` or
            ``"loeo_E01"`` to ``"loeo_E04"``.
        seed: Seed of the random split.

    Returns:
        Tuple of the training and test forms.
    """
    actions = protocol_actions(protocol)
    train_form: dict[str, list[str]] = {}
    test_form: dict[str, list[str]] = {}
    if split_strategy.startswith(LOEO_PREFIX):
        held_out = split_strategy[len(LOEO_PREFIX) :]
        for scene in SCENES:
            if scene == held_out:
                continue
            for subject in scene_subjects(scene):
                train_form[subject] = actions
        for subject in scene_subjects(held_out):
            test_form[subject] = actions
    elif split_strategy == "random":
        rs = seed
        for action in actions:
            np.random.seed(rs)
            idx = np.random.permutation(len(ALL_SUBJECTS))
            cut = int(np.floor(RANDOM_SPLIT_RATIO * len(ALL_SUBJECTS)))
            subjects_train = np.array(ALL_SUBJECTS)[idx[:cut]].tolist()
            subjects_test = np.array(ALL_SUBJECTS)[idx[cut:]].tolist()
            for subject in ALL_SUBJECTS:
                if subject in subjects_train:
                    train_form.setdefault(subject, []).append(action)
                if subject in subjects_test:
                    test_form.setdefault(subject, []).append(action)
            rs += 1
    elif split_strategy == "cross_scene":
        for subject in CROSS_SCENE_TRAIN:
            train_form[subject] = actions
        for subject in CROSS_SCENE_TEST:
            test_form[subject] = actions
    elif split_strategy == "cross_subject":
        for subject in CROSS_SUBJECT_TRAIN:
            train_form[subject] = actions
        for subject in CROSS_SUBJECT_TEST:
            test_form[subject] = actions
    else:
        raise ValueError(f"Unknown MM-Fi split '{split_strategy}'.")
    return train_form, test_form


@dataclass(frozen=True)
class MMFiSequence:
    """One recorded sequence (subject, action) of the dataset."""

    scene: str
    subject: str
    action: str
    radar_dir: Path
    gt_path: Path


def list_sequences(root: Path, form: dict[str, list[str]]) -> list[MMFiSequence]:
    """Enumerate the sequences of a split in the toolbox order.

    Args:
        root: Dataset root with the ``E0x`` folders.
        form: Subject-to-actions mapping of the split.

    Returns:
        Sequences in the order subjects first, actions second.
    """
    sequences: list[MMFiSequence] = []
    for subject, actions in form.items():
        scene = scene_of(subject)
        for action in actions:
            folder = root / scene / subject / action
            sequences.append(
                MMFiSequence(
                    scene=scene,
                    subject=subject,
                    action=action,
                    radar_dir=folder / "mmwave",
                    gt_path=folder / "ground_truth.npy",
                )
            )
    return sequences


def read_mmwave_sequence(radar_dir: Path) -> list[np.ndarray]:
    """Read the radar frames of one sequence.

    Args:
        radar_dir: Folder with ``frameNNN.bin`` files.

    Returns:
        List of point clouds ``(N_t, 5)`` in frame order.
    """
    frames: list[np.ndarray] = []
    for bin_file in sorted(radar_dir.glob("frame*.bin")):
        raw = np.frombuffer(bin_file.read_bytes(), dtype=np.float64)
        frames.append(raw.copy().reshape(-1, POINT_DIM))
    return frames


class MMFiDataset(Dataset):
    """Sliding windows of aggregated MM-Fi radar frames with the matching poses.

    Each window holds ``sequence_length`` aggregated frames. The aggregated frame at raw
    index ``t`` concatenates the points of raw frames ``t - frames_per_aggregate + 1`` to
    ``t`` and is paired with the pose of frame ``t``. Windows advance with stride one.

    Args:
        root: Dataset root with the ``E01`` to ``E04`` folders.
        split: ``"train"`` or ``"test"``.
        split_strategy: ``"cross_subject"``, ``"cross_scene"`` or ``"random"``.
        protocol: Activity protocol (3 = all activities).
        sequence_length: Aggregated frames per window.
        frames_per_aggregate: Raw frames per aggregated frame.
        max_points: Points per aggregated frame after padding or truncation.
        normalized: Centre every pose on joint 0.
        seed: Seed of the random split.
        return_meta: Also return :class:`MMFiSampleMeta`.
    """

    def __init__(
        self,
        root: Path,
        split: str,
        split_strategy: str,
        protocol: int = 3,
        sequence_length: int = 5,
        frames_per_aggregate: int = 5,
        max_points: int = 322,
        normalized: bool = True,
        seed: int = 0,
        return_meta: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.sequence_length = sequence_length
        self.frames_per_aggregate = frames_per_aggregate
        self.max_points = max_points
        self.normalized = normalized
        self.return_meta = return_meta
        if not self.root.is_dir():
            raise FileNotFoundError(f"MM-Fi root not found: {self.root}")

        np.random.seed(seed)
        train_form, test_form = split_forms(protocol, split_strategy, seed)
        form = train_form if split == "train" else test_form
        self.sequences = list_sequences(self.root, form)

        self._radar: list[list[np.ndarray]] = []
        self._poses: list[np.ndarray] = []
        for seq in tqdm(self.sequences, desc=f"Loading MM-Fi {split}"):
            self._radar.append(read_mmwave_sequence(seq.radar_dir))
            self._poses.append(np.load(seq.gt_path))
        self.windows = self._create_windows()
        logger.info("MM-Fi %s/%s: %d sequences, %d windows", split_strategy, split, len(self.sequences), len(self.windows))

    def _create_windows(self) -> list[tuple[int, int]]:
        """Window indices as ``(sequence index, raw index of the first aggregated frame)``."""
        min_start = self.frames_per_aggregate - 1
        max_end = FRAMES_PER_SEQUENCE - 1
        return [
            (seq_idx, start)
            for seq_idx in range(len(self.sequences))
            for start in range(min_start, max_end - self.sequence_length + 2)
        ]

    def _aggregate(self, frames: list[np.ndarray], start: int, count: int) -> np.ndarray:
        """Concatenate ``count`` raw frames from ``start``, skipping empty frames."""
        parts = [frames[i] for i in range(start, start + count) if i < len(frames) and frames[i].shape[0] > 0]
        if not parts:
            return np.zeros((1, POINT_DIM), dtype=np.float64)
        return np.concatenate(parts, axis=0)

    def _pad_or_truncate(self, points: np.ndarray) -> np.ndarray:
        """Bring a point cloud to exactly ``max_points`` rows."""
        n = points.shape[0]
        if n == 0:
            return np.zeros((self.max_points, POINT_DIM), dtype=np.float64)
        if n > self.max_points:
            return points[: self.max_points]
        padding = np.zeros((self.max_points - n, points.shape[1]), dtype=points.dtype)
        return np.concatenate([points, padding], axis=0)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        """Return ``(radar, pose)`` or ``(radar, pose, meta)`` for one window.

        Args:
            index: Window index.

        Returns:
            Radar ``(sequence_length, max_points, 5)`` and poses
            ``(sequence_length, 17, 3)`` in metres as float tensors.
        """
        seq_idx, start = self.windows[index]
        frames = self._radar[seq_idx]
        poses = self._poses[seq_idx]
        radar_list = []
        pose_list = []
        for offset in range(self.sequence_length):
            raw_idx = start + offset
            if self.frames_per_aggregate > 1:
                agg_start = max(0, raw_idx - self.frames_per_aggregate + 1)
                radar_frame = self._aggregate(frames, agg_start, raw_idx + 1 - agg_start)
            else:
                radar_frame = frames[raw_idx]
            pose_frame = np.asarray(poses[raw_idx])
            if self.normalized:
                pose_frame = pose_frame - pose_frame[0]
            radar_list.append(self._pad_or_truncate(radar_frame))
            pose_list.append(pose_frame)
        radar = torch.from_numpy(np.stack(radar_list, axis=0)).float()
        pose = torch.from_numpy(np.stack(pose_list, axis=0)).float()
        if self.return_meta:
            seq = self.sequences[seq_idx]
            meta = MMFiSampleMeta(seq.subject, seq.scene, seq.action, seq_idx, index)
            return radar, pose, meta
        return radar, pose
