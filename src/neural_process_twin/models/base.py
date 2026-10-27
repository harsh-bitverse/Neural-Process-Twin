"""Abstract base class and shared interface for PDE surrogate models.

Defines the contract for forward prediction, parameter counting, device tracking,
and input/output tensor shape consistency across CNN and neural operator models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SurrogateModel(nn.Module, ABC):
    """Abstract base class for 2D physical field surrogate models.

    All surrogate models in Neural Process Twin must accept input tensors of shape
    (batch, in_channels, ny, nx) and return solution field predictions of shape
    (batch, out_channels=1, ny, nx).
    """

    def __init__(self, in_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {out_channels}.")
        self._in_channels = in_channels
        self._out_channels = out_channels

    @property
    def in_channels(self) -> int:
        """Number of input channels accepted by the model."""
        return self._in_channels

    @property
    def out_channels(self) -> int:
        """Number of output channels predicted by the model."""
        return self._out_channels

    @property
    def device(self) -> torch.device:
        """Execution device of the model parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def count_parameters(self) -> dict[str, int]:
        """Count total, trainable, and non-trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "non_trainable_params": total - trainable,
        }

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting physical solution fields.

        Args:
            x: Input tensor of shape (batch, in_channels, ny, nx).

        Returns:
            Predicted solution field of shape (batch, out_channels, ny, nx).
        """
        ...
