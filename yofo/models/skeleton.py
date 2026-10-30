"""Skeleton edge sets and adjacency matrices."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

COCO_SKELETON_EDGES = torch.tensor(
    [
        [0, 1], [0, 2],
        [1, 3], [2, 4],
        [5, 6],
        [5, 7], [7, 9],
        [6, 8], [8, 10],
        [5, 11], [6, 12],
        [11, 12],
        [11, 13], [13, 15],
        [12, 14], [14, 16],
    ],
    dtype=torch.long,
)
"""COCO 17-joint skeleton (MM-Fi and mRI): nose, eyes, ears, shoulders, elbows, wrists,
hips, knees, ankles."""

MMRADPOSE_SKELETON_EDGES = torch.tensor(
    [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
        [0, 6], [6, 7], [7, 8], [8, 9], [9, 10],
        [0, 11], [11, 12],
        [12, 13], [13, 14], [14, 15], [15, 16], [16, 17],
        [12, 18], [18, 19], [19, 20], [20, 21], [21, 22],
        [12, 23], [23, 24], [24, 25],
    ],
    dtype=torch.long,
)
"""mmRadPose 26-joint skeleton: legs, spine, arms and head chains from the hips."""

SKELETONS = {"coco": COCO_SKELETON_EDGES, "mmradpose": MMRADPOSE_SKELETON_EDGES}


def skeleton_edges(name: str) -> torch.Tensor:
    """Return the edge list of a named skeleton.

    Args:
        name: ``"coco"`` or ``"mmradpose"``.

    Returns:
        Edge list ``(E, 2)``.
    """
    try:
        return SKELETONS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown skeleton '{name}'. Choose one of {sorted(SKELETONS)}.") from exc


def _row_normalize(matrix: sp.coo_matrix) -> sp.coo_matrix:
    """Row-normalise a sparse matrix."""
    rowsum = np.array(matrix.sum(1))
    inv = np.power(rowsum, -1).flatten()
    inv[np.isinf(inv)] = 0.0
    return sp.diags(inv).dot(matrix)


def adjacency_from_edges(num_pts: int, edges: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """Dense adjacency matrix of an undirected graph.

    Args:
        num_pts: Number of nodes.
        edges: Edge list ``(E, 2)``.
        normalize: Row-normalise ``A + I``.

    Returns:
        Dense adjacency matrix ``(num_pts, num_pts)``.
    """
    edges_np = np.array(edges, dtype=np.int32)
    data = np.ones(edges_np.shape[0])
    i, j = edges_np[:, 0], edges_np[:, 1]
    adj = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if normalize:
        adj = _row_normalize(adj + sp.eye(adj.shape[0]))
    return torch.tensor(adj.todense(), dtype=torch.float)


def skeleton_adjacency(name: str, num_keypoints: int) -> torch.Tensor:
    """Normalised dense adjacency matrix of a named skeleton.

    Args:
        name: ``"coco"`` or ``"mmradpose"``.
        num_keypoints: Number of joints.

    Returns:
        Adjacency matrix ``(num_keypoints, num_keypoints)``.
    """
    return adjacency_from_edges(num_keypoints, skeleton_edges(name), normalize=True)
