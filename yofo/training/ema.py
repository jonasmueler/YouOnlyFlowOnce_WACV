"""Exponential moving average of model parameters."""

from __future__ import annotations

import torch
import torch.nn as nn


class EMAHelper:
    """Keep a shadow copy of all trainable parameters updated by an exponential average.

    Args:
        mu: Decay; the shadow keeps ``mu`` of its value and takes ``1 - mu`` of the parameter.
    """

    def __init__(self, mu: float = 0.999) -> None:
        self.mu = mu
        self.shadow: dict[str, torch.Tensor] = {}

    def register(self, module: nn.Module) -> None:
        """Initialise the shadow from the module's current parameters."""
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module: nn.Module) -> None:
        """Move the shadow towards the module's current parameters."""
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.mu) * param.data + self.mu * self.shadow[name]

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Shadow parameters, loadable into the module with ``strict=False``."""
        return self.shadow
