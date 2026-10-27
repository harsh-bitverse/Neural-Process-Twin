"""Differentiable physics-informed loss modules for steady-state heat conduction.

Provides exact discrete finite-volume operators and normalized, dimensionless loss
formulations matching the conservative reference solver conventions to machine precision:
- DiscretePDEResidual: Vectorized PyTorch finite-volume residual operator r_P = (A*T - b)_P [W].
- DimensionlessPDELoss: Scale-invariant PDE residual loss with safe denominator floor.
- BoundaryConditionLoss: Differentiable Dirichlet face temperature and Neumann flux penalties.
- PhysicsAwareLoss: Compound objective coordinating supervised data, PDE residual, and BC losses.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.training.loss import RelativeL2Loss


class DiscretePDEResidual(nn.Module):
    """Differentiable PyTorch conservative finite-volume operator for steady-state heat conduction.

    Discretizes:
        -\\nabla \\cdot (k \\nabla T) = q
    on a uniform 2D Cartesian grid with cell-centred finite volumes. Internal face conductances
    use harmonic averaging k_f = 2*k_P*k_N / (k_P + k_N). Dirichlet and Neumann boundary conditions
    match the linear system assembly A*T - b down to machine precision.
    """

    def __init__(
        self,
        grid: Grid2D,
        flux_scale: float = 1000.0,
        eps: float = 1e-12,
    ) -> None:
        """Initialize DiscretePDEResidual.

        Args:
            grid: 2D computational domain grid specification.
            flux_scale: Characteristic scale [W/m^2] used in boundary flux channel encoding.
            eps: Numerical stability constant for harmonic averaging.
        """
        super().__init__()
        self.grid = grid
        self.dx = float(grid.dx)
        self.dy = float(grid.dy)
        self.thickness = float(grid.thickness)
        self.nx = int(grid.nx)
        self.ny = int(grid.ny)
        self.flux_scale = float(flux_scale)
        self.eps = float(eps)

        self.vol = self.dx * self.dy * self.thickness
        self.area_x = self.dy * self.thickness
        self.area_y = self.dx * self.thickness

    def forward(
        self,
        temperature_k: torch.Tensor,
        k_field: torch.Tensor,
        q_field: torch.Tensor,
        bc_channels: torch.Tensor,
        t_mean: float | torch.Tensor = 0.0,
        t_std: float | torch.Tensor = 1.0,
        residual_type: str = "full",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute discrete PDE residual r = A*T - b and RHS vector b in physical Watts [W].

        Args:
            temperature_k: Predicted temperature tensor in physical Kelvin [K] of shape (B, 1, ny, nx).
            k_field: Physical thermal conductivity tensor [W/(m K)] of shape (B, 1, ny, nx).
            q_field: Physical volumetric heat source tensor [W/m^3] of shape (B, 1, ny, nx).
            bc_channels: 8-channel boundary condition tensor of shape (B, 8, ny, nx).
            t_mean: Empirical mean temperature [K] used to invert normalized Dirichlet channels.
            t_std: Empirical std temperature [K] used to invert normalized Dirichlet channels.
            residual_type: 'full' for all control volumes, or 'interior' for control volumes
                strictly inside the domain (excluding boundary cells).

        Returns:
            Tuple of (residual_r, rhs_b) in Watts [W].
        """
        _b_size, _, ny, nx = temperature_k.shape
        if ny != self.ny or nx != self.nx:
            raise ValueError(
                f"Input spatial shape ({ny}, {nx}) does not match grid ({self.ny}, {self.nx})."
            )

        # 1. Internal conductive face fluxes
        # East face fluxes: (B, 1, ny, nx - 1)
        k_w = k_field[:, :, :, :-1]
        k_e = k_field[:, :, :, 1:]
        k_face_x = 2.0 * k_w * k_e / (k_w + k_e + self.eps)
        g_e = k_face_x * (self.area_x / self.dx)
        q_e = g_e * (temperature_k[:, :, :, :-1] - temperature_k[:, :, :, 1:])

        # North face fluxes: (B, 1, ny - 1, nx)
        k_s = k_field[:, :, :-1, :]
        k_n = k_field[:, :, 1:, :]
        k_face_y = 2.0 * k_s * k_n / (k_s + k_n + self.eps)
        g_n = k_face_y * (self.area_y / self.dy)
        q_n = g_n * (temperature_k[:, :, :-1, :] - temperature_k[:, :, 1:, :])

        # Net conductive flux out of internal control volumes: A*T
        at_tensor = torch.zeros_like(temperature_k)
        at_tensor[:, :, :, :-1] += q_e
        at_tensor[:, :, :, 1:] -= q_e
        at_tensor[:, :, :-1, :] += q_n
        at_tensor[:, :, 1:, :] -= q_n

        # RHS vector: b = q * V_P + boundary contributions
        b_tensor = q_field * self.vol

        # 2. Boundary face contributions
        # Left boundary (x = x_min, column i = 0)
        dir_mask_l = bc_channels[:, 0:1, :, 0:1]
        dir_val_l = bc_channels[:, 1:2, :, 0:1] * t_std + t_mean
        neu_mask_l = bc_channels[:, 2:3, :, 0:1]
        neu_val_l = bc_channels[:, 3:4, :, 0:1] * self.flux_scale

        g_left = 2.0 * k_field[:, :, :, 0:1] * (self.area_x / self.dx)
        at_tensor[:, :, :, 0:1] += dir_mask_l * g_left * temperature_k[:, :, :, 0:1]
        b_tensor[:, :, :, 0:1] += (
            dir_mask_l * g_left * dir_val_l - neu_mask_l * neu_val_l * self.area_x
        )

        # Right boundary (x = x_max, column i = nx - 1)
        dir_mask_r = bc_channels[:, 0:1, :, -1:]
        dir_val_r = bc_channels[:, 1:2, :, -1:] * t_std + t_mean
        neu_mask_r = bc_channels[:, 2:3, :, -1:]
        neu_val_r = bc_channels[:, 3:4, :, -1:] * self.flux_scale

        g_right = 2.0 * k_field[:, :, :, -1:] * (self.area_x / self.dx)
        at_tensor[:, :, :, -1:] += dir_mask_r * g_right * temperature_k[:, :, :, -1:]
        b_tensor[:, :, :, -1:] += (
            dir_mask_r * g_right * dir_val_r - neu_mask_r * neu_val_r * self.area_x
        )

        # Bottom boundary (y = y_min, row j = 0)
        dir_mask_b = bc_channels[:, 4:5, 0:1, :]
        dir_val_b = bc_channels[:, 5:6, 0:1, :] * t_std + t_mean
        neu_mask_b = bc_channels[:, 6:7, 0:1, :]
        neu_val_b = bc_channels[:, 7:8, 0:1, :] * self.flux_scale

        g_bottom = 2.0 * k_field[:, :, 0:1, :] * (self.area_y / self.dy)
        at_tensor[:, :, 0:1, :] += dir_mask_b * g_bottom * temperature_k[:, :, 0:1, :]
        b_tensor[:, :, 0:1, :] += (
            dir_mask_b * g_bottom * dir_val_b - neu_mask_b * neu_val_b * self.area_y
        )

        # Top boundary (y = y_max, row j = ny - 1)
        dir_mask_t = bc_channels[:, 4:5, -1:, :]
        dir_val_t = bc_channels[:, 5:6, -1:, :] * t_std + t_mean
        neu_mask_t = bc_channels[:, 6:7, -1:, :]
        neu_val_t = bc_channels[:, 7:8, -1:, :] * self.flux_scale

        g_top = 2.0 * k_field[:, :, -1:, :] * (self.area_y / self.dy)
        at_tensor[:, :, -1:, :] += dir_mask_t * g_top * temperature_k[:, :, -1:, :]
        b_tensor[:, :, -1:, :] += (
            dir_mask_t * g_top * dir_val_t - neu_mask_t * neu_val_t * self.area_y
        )

        # Total algebraic residual: r = A*T - b [Watts]
        r_tensor = at_tensor - b_tensor

        if residual_type == "interior":
            if ny <= 2 or nx <= 2:
                return r_tensor, b_tensor
            return r_tensor[:, :, 1:-1, 1:-1], b_tensor[:, :, 1:-1, 1:-1]

        return r_tensor, b_tensor


class DimensionlessPDELoss(nn.Module):
    """Scale-invariant, dimensionless PDE residual loss.

    Normalizes the raw finite-volume residual in Watts by a characteristic problem scale:
        loss_pde = ||r||_2 / (||b||_2 + Q_floor)
    where Q_floor > 0 guarantees numerical stability and strictly non-zero denominator
    under zero-source or near-equilibrium configurations.
    """

    def __init__(
        self,
        grid: Grid2D,
        residual_type: str = "full",
        norm_type: str = "l2",
        q_floor: float = 1.0,
        flux_scale: float = 1000.0,
    ) -> None:
        """Initialize DimensionlessPDELoss.

        Args:
            grid: 2D computational domain grid.
            residual_type: 'full' or 'interior'.
            norm_type: 'l2' (relative L2 norm) or 'mse' (relative mean squared residual).
            q_floor: Characteristic heat flow floor [W] preventing division by zero.
            flux_scale: Characteristic scale [W/m^2] for Neumann flux decoding.
        """
        super().__init__()
        self.residual_type = residual_type
        self.norm_type = norm_type.lower()
        self.q_floor = float(q_floor)
        self.operator = DiscretePDEResidual(grid=grid, flux_scale=flux_scale)

    def forward(
        self,
        temperature_k: torch.Tensor,
        k_field: torch.Tensor,
        q_field: torch.Tensor,
        bc_channels: torch.Tensor,
        t_mean: float | torch.Tensor = 0.0,
        t_std: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Compute dimensionless PDE loss.

        Args:
            temperature_k: Temperature field in physical Kelvin [K] of shape (B, 1, ny, nx).
            k_field: Conductivity field [W/(m K)] of shape (B, 1, ny, nx).
            q_field: Heat source field [W/m^3] of shape (B, 1, ny, nx).
            bc_channels: Encoded boundary conditions of shape (B, 8, ny, nx).
            t_mean: Mean temperature scale [K].
            t_std: Std temperature scale [K].

        Returns:
            Dimensionless scalar loss value.
        """
        r, b = self.operator(
            temperature_k=temperature_k,
            k_field=k_field,
            q_field=q_field,
            bc_channels=bc_channels,
            t_mean=t_mean,
            t_std=t_std,
            residual_type=self.residual_type,
        )

        b_size = temperature_k.shape[0]
        r_flat = r.reshape(b_size, -1)
        b_flat = b.reshape(b_size, -1)

        if self.norm_type == "l2":
            r_norm = torch.linalg.norm(r_flat, ord=2, dim=1)
            b_norm = torch.linalg.norm(b_flat, ord=2, dim=1)
            loss_per_sample = r_norm / (b_norm + self.q_floor)
        elif self.norm_type == "mse":
            r_mse = torch.mean(r_flat**2, dim=1)
            b_mse = torch.mean(b_flat**2, dim=1)
            loss_per_sample = r_mse / (b_mse + (self.q_floor**2))
        else:
            raise ValueError(f"Unsupported norm_type '{self.norm_type}'. Must be 'l2' or 'mse'.")

        return torch.mean(loss_per_sample)


class BoundaryConditionLoss(nn.Module):
    """Differentiable boundary condition loss for Dirichlet and Neumann boundary faces.

    Enforces boundary values directly on boundary control volumes:
    - Dirichlet: (T_face - T_prescribed)^2 / delta_t_char^2
    - Neumann: (q_out_face - q_prescribed)^2 / flux_scale^2
    """

    def __init__(
        self,
        grid: Grid2D,
        delta_t_char: float = 50.0,
        flux_scale: float = 1000.0,
        use_extrapolation: bool = False,
    ) -> None:
        """Initialize BoundaryConditionLoss.

        Args:
            grid: 2D computational domain grid.
            delta_t_char: Characteristic temperature scale [K] for non-dimensionalization.
            flux_scale: Characteristic flux scale [W/m^2] for non-dimensionalization.
            use_extrapolation: If True, uses linear face extrapolation 1.5*T_0 - 0.5*T_1;
                if False, directly evaluates cell-centred temperature against Dirichlet face.
        """
        super().__init__()
        self.grid = grid
        self.dx = float(grid.dx)
        self.dy = float(grid.dy)
        self.delta_t_char = float(delta_t_char)
        self.flux_scale = float(flux_scale)
        self.use_extrapolation = use_extrapolation

    def forward(
        self,
        temperature_k: torch.Tensor,
        k_field: torch.Tensor,
        bc_channels: torch.Tensor,
        t_mean: float | torch.Tensor = 0.0,
        t_std: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Compute boundary condition loss across all 4 boundary edges.

        Args:
            temperature_k: Predicted temperature field in Kelvin [K] of shape (B, 1, ny, nx).
            k_field: Physical conductivity field [W/(m K)] of shape (B, 1, ny, nx).
            bc_channels: Encoded boundary condition channels of shape (B, 8, ny, nx).
            t_mean: Mean temperature scale [K].
            t_std: Std temperature scale [K].

        Returns:
            Scalar boundary loss value.
        """
        t_char = self.delta_t_char if self.delta_t_char > 1e-6 else 50.0
        f_scale = self.flux_scale if self.flux_scale > 1e-6 else 1000.0

        errors: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []

        # 1. Left boundary (i = 0)
        dir_m_l = bc_channels[:, 0:1, :, 0:1]
        dir_v_l = bc_channels[:, 1:2, :, 0:1] * t_std + t_mean
        neu_m_l = bc_channels[:, 2:3, :, 0:1]
        neu_v_l = bc_channels[:, 3:4, :, 0:1] * f_scale

        if self.use_extrapolation and temperature_k.shape[3] > 1:
            t_face_l = 1.5 * temperature_k[:, :, :, 0:1] - 0.5 * temperature_k[:, :, :, 1:2]
        else:
            t_face_l = temperature_k[:, :, :, 0:1]

        err_dir_l = ((t_face_l - dir_v_l) / t_char) ** 2
        errors.append(dir_m_l * err_dir_l)
        weights.append(dir_m_l)

        if temperature_k.shape[3] > 1:
            # Outward conductive heat flux on left face is -k * (dT/dn) where n is outward (-x):
            # q_out = k * (T_0 - T_1) / dx
            q_out_l = (
                k_field[:, :, :, 0:1]
                * (temperature_k[:, :, :, 0:1] - temperature_k[:, :, :, 1:2])
                / self.dx
            )
            err_neu_l = ((q_out_l - neu_v_l) / f_scale) ** 2
            errors.append(neu_m_l * err_neu_l)
            weights.append(neu_m_l)

        # 2. Right boundary (i = nx - 1)
        dir_m_r = bc_channels[:, 0:1, :, -1:]
        dir_v_r = bc_channels[:, 1:2, :, -1:] * t_std + t_mean
        neu_m_r = bc_channels[:, 2:3, :, -1:]
        neu_v_r = bc_channels[:, 3:4, :, -1:] * f_scale

        if self.use_extrapolation and temperature_k.shape[3] > 1:
            t_face_r = 1.5 * temperature_k[:, :, :, -1:] - 0.5 * temperature_k[:, :, :, -2:-1]
        else:
            t_face_r = temperature_k[:, :, :, -1:]

        err_dir_r = ((t_face_r - dir_v_r) / t_char) ** 2
        errors.append(dir_m_r * err_dir_r)
        weights.append(dir_m_r)

        if temperature_k.shape[3] > 1:
            # Outward conductive heat flux on right face (n = +x):
            # q_out = -k * (T_nx-1 - T_nx-2) / dx
            q_out_r = (
                k_field[:, :, :, -1:]
                * (temperature_k[:, :, :, -1:] - temperature_k[:, :, :, -2:-1])
                / self.dx
            )
            err_neu_r = ((q_out_r - neu_v_r) / f_scale) ** 2
            errors.append(neu_m_r * err_neu_r)
            weights.append(neu_m_r)

        # 3. Bottom boundary (j = 0)
        dir_m_b = bc_channels[:, 4:5, 0:1, :]
        dir_v_b = bc_channels[:, 5:6, 0:1, :] * t_std + t_mean
        neu_m_b = bc_channels[:, 6:7, 0:1, :]
        neu_v_b = bc_channels[:, 7:8, 0:1, :] * f_scale

        if self.use_extrapolation and temperature_k.shape[2] > 1:
            t_face_b = 1.5 * temperature_k[:, :, 0:1, :] - 0.5 * temperature_k[:, :, 1:2, :]
        else:
            t_face_b = temperature_k[:, :, 0:1, :]

        err_dir_b = ((t_face_b - dir_v_b) / t_char) ** 2
        errors.append(dir_m_b * err_dir_b)
        weights.append(dir_m_b)

        if temperature_k.shape[2] > 1:
            # Outward flux on bottom face (n = -y):
            # q_out = k * (T_0 - T_1) / dy
            q_out_b = (
                k_field[:, :, 0:1, :]
                * (temperature_k[:, :, 0:1, :] - temperature_k[:, :, 1:2, :])
                / self.dy
            )
            err_neu_b = ((q_out_b - neu_v_b) / f_scale) ** 2
            errors.append(neu_m_b * err_neu_b)
            weights.append(neu_m_b)

        # 4. Top boundary (j = ny - 1)
        dir_m_t = bc_channels[:, 4:5, -1:, :]
        dir_v_t = bc_channels[:, 5:6, -1:, :] * t_std + t_mean
        neu_m_t = bc_channels[:, 6:7, -1:, :]
        neu_v_t = bc_channels[:, 7:8, -1:, :] * f_scale

        if self.use_extrapolation and temperature_k.shape[2] > 1:
            t_face_t = 1.5 * temperature_k[:, :, -1:, :] - 0.5 * temperature_k[:, :, -2:-1, :]
        else:
            t_face_t = temperature_k[:, :, -1:, :]

        err_dir_t = ((t_face_t - dir_v_t) / t_char) ** 2
        errors.append(dir_m_t * err_dir_t)
        weights.append(dir_m_t)

        if temperature_k.shape[2] > 1:
            # Outward flux on top face (n = +y):
            # q_out = -k * (T_ny-1 - T_ny-2) / dy
            q_out_t = (
                k_field[:, :, -1:, :]
                * (temperature_k[:, :, -1:] - temperature_k[:, :, -2:-1, :])
                / self.dy
            )
            err_neu_t = ((q_out_t - neu_v_t) / f_scale) ** 2
            errors.append(neu_m_t * err_neu_t)
            weights.append(neu_m_t)

        total_error = sum(torch.sum(e) for e in errors)
        total_weight = sum(torch.sum(w) for w in weights)

        return total_error / torch.clamp(total_weight, min=1.0)


class PhysicsAwareLoss(nn.Module):
    """Compound physics-aware objective coordinating supervised data, PDE residual, and BC losses.

    Computes:
        loss_total = w_data * loss_data + w_pde * loss_pde + w_bc * loss_bc
    Returning the total loss scalar and a detached dictionary of loss components for telemetry.
    """

    is_physics_aware: bool = True

    def __init__(
        self,
        grid: Grid2D,
        normalizer: FieldNormalizer | None = None,
        data_loss_type: str = "relative_l2",
        data_weight: float = 1.0,
        pde_weight: float = 0.0,
        bc_weight: float = 0.0,
        residual_type: str = "full",
        norm_type: str = "l2",
        q_floor: float = 1.0,
        flux_scale: float = 1000.0,
    ) -> None:
        """Initialize PhysicsAwareLoss.

        Args:
            grid: 2D computational domain grid.
            normalizer: Fitted FieldNormalizer for invertible Kelvin mapping.
            data_loss_type: Supervised data loss ('relative_l2', 'mse', or 'l1').
            data_weight: Weight for supervised data loss.
            pde_weight: Weight for dimensionless PDE residual loss.
            bc_weight: Weight for boundary condition loss.
            residual_type: PDE residual domain ('full' or 'interior').
            norm_type: PDE norm formulation ('l2' or 'mse').
            q_floor: Denominator floor for dimensionless PDE loss.
            flux_scale: Characteristic scale for boundary flux encoding.
        """
        super().__init__()
        self.grid = grid
        self.normalizer = normalizer
        self.data_weight = float(data_weight)
        self.pde_weight = float(pde_weight)
        self.bc_weight = float(bc_weight)
        self.residual_type = residual_type

        # Supervised data loss
        loss_type = data_loss_type.lower()
        if loss_type == "relative_l2":
            self.data_loss_fn = RelativeL2Loss()
        elif loss_type == "mse":
            self.data_loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.data_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unsupported data_loss_type '{data_loss_type}'.")

        # Physics loss modules
        self.pde_loss_fn = DimensionlessPDELoss(
            grid=grid,
            residual_type=residual_type,
            norm_type=norm_type,
            q_floor=q_floor,
            flux_scale=flux_scale,
        )
        delta_t_char = (
            self.normalizer.t_stats.std if self.normalizer and self.normalizer.t_stats else 50.0
        )
        self.bc_loss_fn = BoundaryConditionLoss(
            grid=grid,
            delta_t_char=delta_t_char,
            flux_scale=flux_scale,
        )

    def _get_t_scales(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve temperature mean and std scales for inverse normalization."""
        if self.normalizer is not None and self.normalizer.t_stats is not None:
            t_mean = torch.tensor(self.normalizer.t_stats.mean, dtype=torch.float32, device=device)
            t_std = torch.tensor(self.normalizer.t_stats.std, dtype=torch.float32, device=device)
        else:
            t_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
            t_std = torch.tensor(1.0, dtype=torch.float32, device=device)
        return t_mean, t_std

    def _extract_physics_inputs(
        self,
        batch: dict[str, Any],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract physical k, q, and encoded BC channels from training batch."""
        x = batch["x"].to(device)

        # k_field in physical units [W/(m K)]
        if "k_raw" in batch:
            k_raw = batch["k_raw"].to(device)
        elif self.normalizer is not None and self.normalizer.k_stats is not None:
            k_raw = x[:, 0:1] * self.normalizer.k_stats.std + self.normalizer.k_stats.mean
        else:
            k_raw = x[:, 0:1]

        # q_field in physical units [W/m^3]
        if "q_raw" in batch:
            q_raw = batch["q_raw"].to(device)
        elif self.normalizer is not None and self.normalizer.q_stats is not None:
            q_raw = x[:, 1:2] * self.normalizer.q_stats.std + self.normalizer.q_stats.mean
        else:
            q_raw = x[:, 1:2]

        # Encoded boundary channels: channels 4:12 of x
        if x.shape[1] >= 12:
            bc_channels = x[:, 4:12]
        else:
            raise ValueError(
                f"Batch input tensor x has {x.shape[1]} channels; expected at least 12 channels "
                "to evaluate physics loss with encoded boundary conditions."
            )

        return k_raw, q_raw, bc_channels

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute compound loss and detached component metrics.

        Args:
            y_pred: Predicted normalized temperature tensor of shape (B, 1, ny, nx).
            y_true: Ground-truth normalized temperature tensor of shape (B, 1, ny, nx).
            batch: Full DataLoader batch dictionary containing x, k_raw, q_raw, etc.

        Returns:
            Tuple of (total_loss, metrics_dict) where total_loss has active autograd graph.
        """
        device = y_pred.device
        t_mean, t_std = self._get_t_scales(device)

        # 1. Supervised data loss
        loss_data = self.data_loss_fn(y_pred, y_true)
        loss_total = self.data_weight * loss_data

        loss_pde = torch.tensor(0.0, device=device)
        loss_bc = torch.tensor(0.0, device=device)

        # 2. Physics losses (if enabled and batch provided)
        if (self.pde_weight > 0.0 or self.bc_weight > 0.0) and batch is not None:
            k_raw, q_raw, bc_channels = self._extract_physics_inputs(batch, device)

            # Reversibly map normalized temperature prediction to physical Kelvin [K]
            t_pred_k = y_pred * t_std + t_mean

            if self.pde_weight > 0.0:
                loss_pde = self.pde_loss_fn(
                    temperature_k=t_pred_k,
                    k_field=k_raw,
                    q_field=q_raw,
                    bc_channels=bc_channels,
                    t_mean=t_mean,
                    t_std=t_std,
                )
                loss_total = loss_total + self.pde_weight * loss_pde

            if self.bc_weight > 0.0:
                loss_bc = self.bc_loss_fn(
                    temperature_k=t_pred_k,
                    k_field=k_raw,
                    bc_channels=bc_channels,
                    t_mean=t_mean,
                    t_std=t_std,
                )
                loss_total = loss_total + self.bc_weight * loss_bc

        metrics = {
            "loss_total": float(loss_total.detach().item()),
            "loss_data": float(loss_data.detach().item()),
            "loss_pde": float(loss_pde.detach().item()),
            "loss_bc": float(loss_bc.detach().item()),
        }

        return loss_total, metrics
