"""Loading and saving of model checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_state_dict_filtered(model: nn.Module, state: dict[str, torch.Tensor]) -> list[str]:
    """Load the entries of ``state`` whose name and shape match the model.

    Args:
        model: Model to update in place.
        state: State dictionary from a checkpoint.

    Returns:
        Names of checkpoint entries that were skipped because the model has no
        parameter of that name or the shapes differ.
    """
    model_state = model.state_dict()
    skipped: list[str] = []
    filtered: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name not in model_state or model_state[name].shape != tensor.shape:
            skipped.append(name)
            continue
        filtered[name] = tensor
    missing = [name for name in model_state if name not in filtered]
    model_state.update(filtered)
    model.load_state_dict(model_state)
    for name in skipped:
        logger.warning("Checkpoint entry '%s' skipped (unknown name or shape mismatch)", name)
    for name in missing:
        logger.warning("Model parameter '%s' not in checkpoint, kept at its initial value", name)
    return skipped


def load_backbone_checkpoint(model: nn.Module, path: Path) -> dict[str, Any]:
    """Load an MG-Prior checkpoint into a model.

    The file holds a dictionary with the key ``model`` (state dictionary) and optionally
    ``optimizer`` and ``epoch``.

    Args:
        model: MG-Prior instance.
        path: Checkpoint file.

    Returns:
        The raw checkpoint dictionary without the model weights.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    load_state_dict_filtered(model, checkpoint["model"])
    logger.info("Backbone checkpoint loaded from %s", path)
    return {k: v for k, v in checkpoint.items() if k != "model"}


def save_backbone_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer | None, epoch: int, path: Path
) -> None:
    """Save an MG-Prior checkpoint.

    Args:
        model: MG-Prior instance.
        optimizer: Optimizer whose state is stored, or ``None`` for weights only.
        epoch: Last completed epoch (zero-based).
        path: Output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {"model": model.state_dict(), "epoch": epoch}
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)
    logger.info("Checkpoint saved to %s", path)


def save_flow_checkpoint(
    flow: nn.Module,
    cond_module: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: Path,
) -> None:
    """Save the flow and its conditioning module.

    Args:
        flow: RealNVP flow.
        cond_module: Conditioning module.
        optimizer: Optimizer.
        epoch: Last completed epoch (one-based).
        path: Output file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "flow_state_dict": flow.state_dict(),
            "cond_module_state_dict": cond_module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )
    logger.info("Checkpoint saved to %s", path)


def load_flow_checkpoint(
    flow: nn.Module,
    cond_module: nn.Module,
    path: Path,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """Load the flow and its conditioning module (and optionally the optimizer).

    Args:
        flow: RealNVP flow.
        cond_module: Conditioning module.
        path: Checkpoint file.
        optimizer: Optimizer to restore, or ``None``.

    Returns:
        The epoch stored in the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    flow.load_state_dict(checkpoint["flow_state_dict"])
    cond_module.load_state_dict(checkpoint["cond_module_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    logger.info("Flow checkpoint loaded from %s (epoch %s)", path, checkpoint.get("epoch"))
    return int(checkpoint.get("epoch", 0))
