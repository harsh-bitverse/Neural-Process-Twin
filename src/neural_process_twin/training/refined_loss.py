"""Refined localized and physics-aware loss functions for Milestone 08F.

Provides loss formulations targeting localized peak errors (MaxAE), boundary layer
mismatches, and material interface gradient deviations that standard global relative
L2 losses fail to penalize.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_process_twin.training.loss import RelativeL2Loss


class LogSumExpMaxLoss(nn.Module):
    """Numerically stable smooth approximation to the Chebyshev (L_infinity) norm.

    Computes:
        L_smooth_max = (1 / beta) * log( (1 / N) * sum_{i,j} exp(beta * |e_{i,j}|) )
    where e = y_pred - y_true in physical Kelvin or normalized space, and beta > 0
    is the sharpness parameter. As beta -> infty, L_smooth_max -> max_{i,j} |e_{i,j}|.
    """

    def __init__(self, beta: float = 2.0, eps: float = 1e-8) -> None:
        """Initialize LogSumExpMaxLoss.

        Args:
            beta: Smoothness parameter. Higher values approximate MaxAE more tightly.
            eps: Numerical stability constant.
        """
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"beta must be strictly positive, got {beta}")
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute smooth maximum error loss.

        Args:
            pred: Predicted tensor of shape (B, 1, H, W).
            target: Ground truth tensor of shape (B, 1, H, W).

        Returns:
            Scalar smooth maximum loss tensor.
        """
        diff = torch.abs(pred - target)
        b, _c, h, w = diff.shape
        n_elements = h * w

        # Reshape to (B, N) for per-sample numerical stabilization
        diff_flat = diff.view(b, -1)
        max_val, _ = torch.max(diff_flat, dim=-1, keepdim=True)  # (B, 1)

        # Numerically stable Log-Sum-Exp:
        # log(sum(exp(beta * (x - max_x)))) + beta * max_x - log(N)
        scaled_diff = self.beta * (diff_flat - max_val)
        lse = torch.log(torch.sum(torch.exp(scaled_diff), dim=-1, keepdim=True) + self.eps)
        smooth_max = max_val + (1.0 / self.beta) * (
            lse - torch.log(torch.tensor(n_elements, dtype=torch.float32, device=diff.device))
        )

        return torch.mean(smooth_max)


class BoundaryLayerLoss(nn.Module):
    """Loss function penalizing errors concentrated in boundary layer cells.

    Specifically targets the 1-to-2 cell outer ring adjacent to domain edges where
    Fourier mode truncation produces Gibbs ringing and flux mismatch.
    """

    def __init__(self, ring_width: int = 2) -> None:
        """Initialize BoundaryLayerLoss.

        Args:
            ring_width: Number of boundary cell layers included in the penalty ring.
        """
        super().__init__()
        self.ring_width = int(ring_width)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary layer absolute error loss.

        Args:
            pred: Predicted tensor of shape (B, 1, H, W).
            target: Ground truth tensor of shape (B, 1, H, W).

        Returns:
            Scalar boundary layer loss tensor.
        """
        diff = torch.abs(pred - target)
        h, w = diff.shape[-2], diff.shape[-1]
        rw = min(self.ring_width, h // 2, w // 2)

        # Create boundary ring mask: True on boundary ring, False in interior
        mask = torch.zeros((h, w), dtype=torch.bool, device=diff.device)
        mask[:rw, :] = True
        mask[-rw:, :] = True
        mask[:, :rw] = True
        mask[:, -rw:] = True

        return torch.mean(diff[:, :, mask])


class InterfaceGradientLoss(nn.Module):
    """Loss function weighted by conductivity gradient magnitude |nabla k|.

    Penalizes temperature errors located at steep material interfaces where
    heat flux continuity requires discontinuous temperature derivatives.
    """

    def __init__(self, alpha: float = 2.0, eps: float = 1e-6) -> None:
        """Initialize InterfaceGradientLoss.

        Args:
            alpha: Weighting multiplier for conductivity gradient.
            eps: Numerical stability constant.
        """
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        k_field: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute conductivity interface weighted loss.

        Args:
            pred: Predicted tensor of shape (B, 1, H, W).
            target: Ground truth tensor of shape (B, 1, H, W).
            k_field: Optional conductivity field tensor (B, 1, H, W) or log-k.

        Returns:
            Scalar interface-weighted loss tensor.
        """
        diff = torch.abs(pred - target)
        if k_field is None:
            return torch.mean(diff)

        # Estimate spatial gradients via central differences
        # Pad with replicate border
        k_padded = nn.functional.pad(k_field, (1, 1, 1, 1), mode="replicate")
        grad_x = 0.5 * (k_padded[:, :, 1:-1, 2:] - k_padded[:, :, 1:-1, :-2])
        grad_y = 0.5 * (k_padded[:, :, 2:, 1:-1] - k_padded[:, :, :-2, 1:-1])
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + self.eps)

        # Normalized gradient weight in [1.0, 1.0 + alpha]
        b = grad_mag.shape[0]
        max_grad = torch.amax(grad_mag.view(b, -1), dim=-1, keepdim=True).view(b, 1, 1, 1)
        weights = 1.0 + self.alpha * (grad_mag / (max_grad + self.eps))

        return torch.mean(weights * diff)


class RefinedLocalizedLoss(nn.Module):
    """Composite loss balancing global accuracy and localized peak error reduction.

    Formulation:
        L_total = L_rel_l2 + lambda_max * L_smooth_max + lambda_bnd * L_bnd + lambda_grad * L_grad
    """

    def __init__(
        self,
        lambda_max: float = 0.05,
        lambda_bnd: float = 0.10,
        lambda_grad: float = 0.05,
        beta: float = 2.0,
        bnd_ring_width: int = 2,
    ) -> None:
        """Initialize RefinedLocalizedLoss.

        Args:
            lambda_max: Weight for smooth maximum error penalty.
            lambda_bnd: Weight for boundary layer error penalty.
            lambda_grad: Weight for conductivity interface error penalty.
            beta: LogSumExp sharpness parameter.
            bnd_ring_width: Boundary ring thickness in cells.
        """
        super().__init__()
        self.lambda_max = float(lambda_max)
        self.lambda_bnd = float(lambda_bnd)
        self.lambda_grad = float(lambda_grad)

        self.rel_l2_loss = RelativeL2Loss()
        self.smooth_max_loss = LogSumExpMaxLoss(beta=beta)
        self.boundary_loss = BoundaryLayerLoss(ring_width=bnd_ring_width)
        self.interface_loss = InterfaceGradientLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        k_field: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute composite loss and component breakdown.

        Args:
            pred: Predicted tensor of shape (B, 1, H, W).
            target: Ground truth tensor of shape (B, 1, H, W).
            k_field: Optional conductivity field tensor (B, 1, H, W).

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        l_rel = self.rel_l2_loss(pred, target)
        l_max = self.smooth_max_loss(pred, target)
        l_bnd = self.boundary_loss(pred, target)
        l_grad = self.interface_loss(pred, target, k_field)

        total = (
            l_rel + self.lambda_max * l_max + self.lambda_bnd * l_bnd + self.lambda_grad * l_grad
        )

        metrics = {
            "loss_total": float(total.item()),
            "loss_rel_l2": float(l_rel.item()),
            "loss_smooth_max": float(l_max.item()),
            "loss_bnd": float(l_bnd.item()),
            "loss_grad": float(l_grad.item()),
        }
        return total, metrics
