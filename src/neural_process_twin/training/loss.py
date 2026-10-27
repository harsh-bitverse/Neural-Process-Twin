"""Loss formulations for surrogate model training.

Provides standard supervised data loss functions for continuous field prediction:
- Relative L2 loss (standard evaluation and training objective for neural operators)
- Mean squared error (MSE) loss
- L1 loss
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.utils.config import LossConfig


class RelativeL2Loss(nn.Module):
    """Relative L2 norm loss between predicted and ground-truth spatial fields.

    Computes:
        loss = ||y_pred - y_true||_2 / (||y_true||_2 + eps)
    per sample in the batch, optionally reduced via mean or sum.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-7) -> None:
        """Initialize RelativeL2Loss.

        Args:
            reduction: 'mean', 'sum', or 'none'.
            eps: Small positive constant preventing division by zero.
        """
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"Invalid reduction '{reduction}'. Must be 'mean', 'sum', or 'none'.")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0.0, got {eps}.")
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Compute relative L2 loss.

        Args:
            y_pred: Predicted field tensor of shape (batch, ...)
            y_true: Ground-truth field tensor of shape (batch, ...)

        Returns:
            Relative L2 loss scalar (or vector if reduction='none').
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch: y_pred shape {y_pred.shape} != y_true shape {y_true.shape}."
            )

        batch_size = y_pred.shape[0]
        diff = (y_pred - y_true).reshape(batch_size, -1)
        ref = y_true.reshape(batch_size, -1)

        diff_norm = torch.linalg.norm(diff, ord=2, dim=1)
        ref_norm = torch.linalg.norm(ref, ord=2, dim=1)

        rel_error = diff_norm / (ref_norm + self.eps)

        if self.reduction == "mean":
            return rel_error.mean()
        elif self.reduction == "sum":
            return rel_error.sum()
        else:
            return rel_error


def build_loss(
    config: LossConfig | dict[str, Any],
    grid: Grid2D | None = None,
    normalizer: FieldNormalizer | None = None,
) -> nn.Module:
    """Build loss function module from LossConfig or dictionary.

    Args:
        config: LossConfig instance or dictionary.
        grid: Optional Grid2D instance required for physics-informed losses.
        normalizer: Optional fitted FieldNormalizer for physics loss temperature inversion.

    Returns:
        Callable PyTorch loss module (RelativeL2Loss, MSELoss, or PhysicsAwareLoss).
    """
    physics_enabled = False
    pde_weight = 0.0
    bc_weight = 0.0
    residual_type = "full"
    q_floor = 1.0
    norm_type = "l2"
    data_weight = 1.0

    if isinstance(config, LossConfig):
        loss_type = config.data_loss.type.lower()
        data_weight = config.data_loss.weight
        if config.physics_loss.enabled:
            physics_enabled = True
            pde_weight = config.physics_loss.weight
            bc_weight = config.physics_loss.bc_weight
            residual_type = (
                "interior" if "interior" in config.physics_loss.residual_type else "full"
            )
            q_floor = config.physics_loss.q_floor
            norm_type = config.physics_loss.norm_type
    elif isinstance(config, dict):
        data_loss = config.get("data_loss", {})
        if isinstance(data_loss, dict):
            loss_type = str(data_loss.get("type", "relative_l2")).lower()
            data_weight = float(data_loss.get("weight", 1.0))
        else:
            loss_type = str(config.get("type", "relative_l2")).lower()

        phys = config.get("physics_loss", {})
        if isinstance(phys, dict) and phys.get("enabled", False):
            physics_enabled = True
            pde_weight = float(phys.get("weight", 0.0))
            bc_weight = float(phys.get("bc_weight", 0.0))
            res_str = str(phys.get("residual_type", "full"))
            residual_type = "interior" if "interior" in res_str else "full"
            q_floor = float(phys.get("q_floor", 1.0))
            norm_type = str(phys.get("norm_type", "l2"))
    else:
        raise TypeError(f"Expected LossConfig or dict, got {type(config)}.")

    if physics_enabled and grid is not None:
        from neural_process_twin.training.physics_loss import PhysicsAwareLoss

        return PhysicsAwareLoss(
            grid=grid,
            normalizer=normalizer,
            data_loss_type=loss_type,
            data_weight=data_weight,
            pde_weight=pde_weight,
            bc_weight=bc_weight,
            residual_type=residual_type,
            norm_type=norm_type,
            q_floor=q_floor,
        )

    if loss_type == "relative_l2":
        return RelativeL2Loss()
    elif loss_type == "mse":
        return nn.MSELoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unsupported loss type '{loss_type}'.")
