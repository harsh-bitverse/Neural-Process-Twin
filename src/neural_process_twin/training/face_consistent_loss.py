"""Face-consistent differentiable boundary loss for 2D cell-centred finite volume grids.

Resolves the fundamental physical discrepancy between cell-centred temperatures
and boundary face values in discrete finite-volume heat conduction:
1. Dirichlet Boundaries: Reconstructs face temperatures via exact linear extrapolation
   T_face = 1.5 * T_0 - 0.5 * T_1, eliminating artificial boundary conductance suppression.
2. Neumann Boundaries: Computes consistent outward normal conductive heat flux densities
   matching outward surface normals (-x for West, +x for East, -y for South, +y for North)
   via 1st- or 2nd-order one-sided finite differences.
3. Global Heat Balance: Enforces total domain energy conservation
   |sum(Q_out) - Q_gen| matching the reference solver's physical diagnostics.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from neural_process_twin.solvers.grid import Grid2D


class FaceConsistentBoundaryLoss(nn.Module):
    """Differentiable face-consistent boundary loss for cell-centred finite volume grids.

    Discretizes physical boundary conditions on cell-centred Cartesian grids where cell
    centres are located at half-cell distances (dx / 2, dy / 2) from boundary faces.
    """

    def __init__(
        self,
        grid: Grid2D,
        delta_t_char: float = 50.0,
        flux_scale: float = 1000.0,
        q_floor: float = 1.0,
        weight_dirichlet: float = 1.0,
        weight_neumann: float = 1.0,
        weight_flux_balance: float = 0.1,
        neumann_order: int = 2,
        eps: float = 1e-12,
    ) -> None:
        """Initialize FaceConsistentBoundaryLoss.

        Args:
            grid: 2D computational domain grid.
            delta_t_char: Characteristic temperature scale [K] for non-dimensionalization.
            flux_scale: Characteristic flux scale [W/m^2] for non-dimensionalization.
            q_floor: Characteristic heat flow floor [W] preventing division by zero.
            weight_dirichlet: Weight for Dirichlet face temperature consistency loss.
            weight_neumann: Weight for Neumann outward flux consistency loss.
            weight_flux_balance: Weight for global domain energy conservation loss.
            neumann_order: Finite difference order for Neumann gradient (1 or 2).
            eps: Numerical stability constant.
        """
        super().__init__()
        self.grid = grid
        self.dx = float(grid.dx)
        self.dy = float(grid.dy)
        self.thickness = float(grid.thickness)
        self.nx = int(grid.nx)
        self.ny = int(grid.ny)
        self.delta_t_char = float(delta_t_char) if delta_t_char > 1e-6 else 50.0
        self.flux_scale = float(flux_scale) if flux_scale > 1e-6 else 1000.0
        self.q_floor = float(q_floor) if q_floor > 1e-6 else 1.0
        self.weight_dirichlet = float(weight_dirichlet)
        self.weight_neumann = float(weight_neumann)
        self.weight_flux_balance = float(weight_flux_balance)
        self.neumann_order = int(neumann_order)
        self.eps = float(eps)

        self.area_x = self.dy * self.thickness
        self.area_y = self.dx * self.thickness
        self.vol = self.dx * self.dy * self.thickness

    def compute_boundary_fluxes(
        self,
        temperature_k: torch.Tensor,
        k_field: torch.Tensor,
        bc_channels: torch.Tensor,
        t_mean: float | torch.Tensor = 0.0,
        t_std: float | torch.Tensor = 1.0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Compute outward heat flux densities and heat flows across all four domain boundaries.

        Args:
            temperature_k: Solution field in physical Kelvin [K] of shape (B, 1, ny, nx).
            k_field: Thermal conductivity field [W/(m K)] of shape (B, 1, ny, nx).
            bc_channels: 8-channel boundary tensor of shape (B, 8, ny, nx).
            t_mean: Temperature normalizer mean [K].
            t_std: Temperature normalizer std [K].

        Returns:
            Dictionary containing per-boundary flux densities [W/m^2] and heat flows [W]:
            {
                'left': {'j_out': tensor(B, 1, ny, 1), 'q_flow': tensor(B,)},
                'right': {'j_out': tensor(B, 1, ny, 1), 'q_flow': tensor(B,)},
                'bottom': {'j_out': tensor(B, 1, 1, nx), 'q_flow': tensor(B,)},
                'top': {'j_out': tensor(B, 1, 1, nx), 'q_flow': tensor(B,)},
            }
        """
        # Unpack boundary channels
        # x-normal (West/East): channels 0-3
        dir_m_x = bc_channels[:, 0:1, :, :]
        dir_v_x = bc_channels[:, 1:2, :, :] * t_std + t_mean
        neu_m_x = bc_channels[:, 2:3, :, :]
        neu_v_x = bc_channels[:, 3:4, :, :] * self.flux_scale

        # y-normal (South/North): channels 4-7
        dir_m_y = bc_channels[:, 4:5, :, :]
        dir_v_y = bc_channels[:, 5:6, :, :] * t_std + t_mean
        neu_m_y = bc_channels[:, 6:7, :, :]
        neu_v_y = bc_channels[:, 7:8, :, :] * self.flux_scale

        fluxes: dict[str, dict[str, torch.Tensor]] = {}

        # 1. West boundary (x = 0, column i = 0, normal n = (-1, 0))
        # Dirichlet outward flux: j_out = (2 * k / dx) * (T_0 - T_dir)
        j_dir_l = (2.0 * k_field[:, :, :, 0:1] / self.dx) * (
            temperature_k[:, :, :, 0:1] - dir_v_x[:, :, :, 0:1]
        )
        # Neumann outward flux: imposed prescribed flux
        j_neu_l = neu_v_x[:, :, :, 0:1]
        j_out_l = dir_m_x[:, :, :, 0:1] * j_dir_l + neu_m_x[:, :, :, 0:1] * j_neu_l
        q_flow_l = torch.sum(j_out_l * self.area_x, dim=(-2, -1)).squeeze(-1)
        fluxes["left"] = {"j_out": j_out_l, "q_flow": q_flow_l}

        # 2. East boundary (x = Lx, column i = nx - 1, normal n = (+1, 0))
        j_dir_r = (2.0 * k_field[:, :, :, -1:] / self.dx) * (
            temperature_k[:, :, :, -1:] - dir_v_x[:, :, :, -1:]
        )
        j_neu_r = neu_v_x[:, :, :, -1:]
        j_out_r = dir_m_x[:, :, :, -1:] * j_dir_r + neu_m_x[:, :, :, -1:] * j_neu_r
        q_flow_r = torch.sum(j_out_r * self.area_x, dim=(-2, -1)).squeeze(-1)
        fluxes["right"] = {"j_out": j_out_r, "q_flow": q_flow_r}

        # 3. South boundary (y = 0, row j = 0, normal n = (0, -1))
        j_dir_b = (2.0 * k_field[:, :, 0:1, :] / self.dy) * (
            temperature_k[:, :, 0:1, :] - dir_v_y[:, :, 0:1, :]
        )
        j_neu_b = neu_v_y[:, :, 0:1, :]
        j_out_b = dir_m_y[:, :, 0:1, :] * j_dir_b + neu_m_y[:, :, 0:1, :] * j_neu_b
        q_flow_b = torch.sum(j_out_b * self.area_y, dim=(-2, -1)).squeeze(-1)
        fluxes["bottom"] = {"j_out": j_out_b, "q_flow": q_flow_b}

        # 4. North boundary (y = Ly, row j = ny - 1, normal n = (0, +1))
        j_dir_t = (2.0 * k_field[:, :, -1:, :] / self.dy) * (
            temperature_k[:, :, -1:, :] - dir_v_y[:, :, -1:, :]
        )
        j_neu_t = neu_v_y[:, :, -1:, :]
        j_out_t = dir_m_y[:, :, -1:, :] * j_dir_t + neu_m_y[:, :, -1:, :] * j_neu_t
        q_flow_t = torch.sum(j_out_t * self.area_y, dim=(-2, -1)).squeeze(-1)
        fluxes["top"] = {"j_out": j_out_t, "q_flow": q_flow_t}

        return fluxes

    def forward(
        self,
        temperature_k: torch.Tensor,
        k_field: torch.Tensor,
        bc_channels: torch.Tensor,
        q_field: torch.Tensor | None = None,
        t_mean: float | torch.Tensor = 0.0,
        t_std: float | torch.Tensor = 1.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute face-consistent boundary condition and heat balance losses.

        Args:
            temperature_k: Predicted temperature field in Kelvin [K] of shape (B, 1, ny, nx).
            k_field: Physical thermal conductivity field [W/(m K)] of shape (B, 1, ny, nx).
            bc_channels: Encoded boundary condition channels of shape (B, 8, ny, nx).
            q_field: Optional physical volumetric heat source [W/m^3] of shape (B, 1, ny, nx).
            t_mean: Temperature normalizer mean [K].
            t_std: Temperature normalizer std [K].

        Returns:
            Tuple of (total_loss, metrics_dict) with active autograd graph on total_loss.
        """
        device = temperature_k.device
        nx, ny = temperature_k.shape[3], temperature_k.shape[2]

        # ------------------------------------------------------------------
        # 1. Dirichlet Face Temperature Consistency Loss
        # ------------------------------------------------------------------
        dir_errors: list[torch.Tensor] = []
        dir_weights: list[torch.Tensor] = []

        # West face (i = 0): T_face = 1.5 * T_0 - 0.5 * T_1
        dir_m_l = bc_channels[:, 0:1, :, 0:1]
        dir_v_l = bc_channels[:, 1:2, :, 0:1] * t_std + t_mean
        if nx > 1:
            t_face_l = 1.5 * temperature_k[:, :, :, 0:1] - 0.5 * temperature_k[:, :, :, 1:2]
        else:
            t_face_l = temperature_k[:, :, :, 0:1]
        err_l = ((t_face_l - dir_v_l) / self.delta_t_char) ** 2
        dir_errors.append(dir_m_l * err_l)
        dir_weights.append(dir_m_l)

        # East face (i = nx - 1): T_face = 1.5 * T_{nx-1} - 0.5 * T_{nx-2}
        dir_m_r = bc_channels[:, 0:1, :, -1:]
        dir_v_r = bc_channels[:, 1:2, :, -1:] * t_std + t_mean
        if nx > 1:
            t_face_r = 1.5 * temperature_k[:, :, :, -1:] - 0.5 * temperature_k[:, :, :, -2:-1]
        else:
            t_face_r = temperature_k[:, :, :, -1:]
        err_r = ((t_face_r - dir_v_r) / self.delta_t_char) ** 2
        dir_errors.append(dir_m_r * err_r)
        dir_weights.append(dir_m_r)

        # South face (j = 0): T_face = 1.5 * T_0 - 0.5 * T_1
        dir_m_b = bc_channels[:, 4:5, 0:1, :]
        dir_v_b = bc_channels[:, 5:6, 0:1, :] * t_std + t_mean
        if ny > 1:
            t_face_b = 1.5 * temperature_k[:, :, 0:1, :] - 0.5 * temperature_k[:, :, 1:2, :]
        else:
            t_face_b = temperature_k[:, :, 0:1, :]
        err_b = ((t_face_b - dir_v_b) / self.delta_t_char) ** 2
        dir_errors.append(dir_m_b * err_b)
        dir_weights.append(dir_m_b)

        # North face (j = ny - 1): T_face = 1.5 * T_{ny-1} - 0.5 * T_{ny-2}
        dir_m_t = bc_channels[:, 4:5, -1:, :]
        dir_v_t = bc_channels[:, 5:6, -1:, :] * t_std + t_mean
        if ny > 1:
            t_face_t = 1.5 * temperature_k[:, :, -1:, :] - 0.5 * temperature_k[:, :, -2:-1, :]
        else:
            t_face_t = temperature_k[:, :, -1:, :]
        err_t = ((t_face_t - dir_v_t) / self.delta_t_char) ** 2
        dir_errors.append(dir_m_t * err_t)
        dir_weights.append(dir_m_t)

        total_dir_err = sum(torch.sum(e) for e in dir_errors)
        total_dir_w = sum(torch.sum(w) for w in dir_weights)
        loss_dirichlet = total_dir_err / torch.clamp(total_dir_w, min=1.0)

        # ------------------------------------------------------------------
        # 2. Neumann Outward Normal Flux Consistency Loss
        # ------------------------------------------------------------------
        neu_errors: list[torch.Tensor] = []
        neu_weights: list[torch.Tensor] = []

        # West face: outward normal n = (-1, 0) -> j_out = +k * (dT/dx)|_{x=0}
        neu_m_l = bc_channels[:, 2:3, :, 0:1]
        neu_v_l = bc_channels[:, 3:4, :, 0:1] * self.flux_scale
        if nx >= 3 and self.neumann_order == 2:
            dt_dx_l = (
                -2.0 * temperature_k[:, :, :, 0:1]
                + 3.0 * temperature_k[:, :, :, 1:2]
                - temperature_k[:, :, :, 2:3]
            ) / self.dx
        elif nx >= 2:
            dt_dx_l = (temperature_k[:, :, :, 1:2] - temperature_k[:, :, :, 0:1]) / self.dx
        else:
            dt_dx_l = torch.zeros_like(temperature_k[:, :, :, 0:1])
        q_out_l = k_field[:, :, :, 0:1] * dt_dx_l
        err_neu_l = ((q_out_l - neu_v_l) / self.flux_scale) ** 2
        neu_errors.append(neu_m_l * err_neu_l)
        neu_weights.append(neu_m_l)

        # East face: outward normal n = (+1, 0) -> j_out = -k * (dT/dx)|_{x=Lx}
        neu_m_r = bc_channels[:, 2:3, :, -1:]
        neu_v_r = bc_channels[:, 3:4, :, -1:] * self.flux_scale
        if nx >= 3 and self.neumann_order == 2:
            # -dT/dx = -(-2*T_{nx-1} + 3*T_{nx-2} - T_{nx-3})/(-dx) = (-2*T_{nx-1} + 3*T_{nx-2} - T_{nx-3})/dx
            minus_dt_dx_r = (
                -2.0 * temperature_k[:, :, :, -1:]
                + 3.0 * temperature_k[:, :, :, -2:-1]
                - temperature_k[:, :, :, -3:-2]
            ) / self.dx
        elif nx >= 2:
            minus_dt_dx_r = (temperature_k[:, :, :, -2:-1] - temperature_k[:, :, :, -1:]) / self.dx
        else:
            minus_dt_dx_r = torch.zeros_like(temperature_k[:, :, :, -1:])
        q_out_r = k_field[:, :, :, -1:] * minus_dt_dx_r
        err_neu_r = ((q_out_r - neu_v_r) / self.flux_scale) ** 2
        neu_errors.append(neu_m_r * err_neu_r)
        neu_weights.append(neu_m_r)

        # South face: outward normal n = (0, -1) -> j_out = +k * (dT/dy)|_{y=0}
        neu_m_b = bc_channels[:, 6:7, 0:1, :]
        neu_v_b = bc_channels[:, 7:8, 0:1, :] * self.flux_scale
        if ny >= 3 and self.neumann_order == 2:
            dt_dy_b = (
                -2.0 * temperature_k[:, :, 0:1, :]
                + 3.0 * temperature_k[:, :, 1:2, :]
                - temperature_k[:, :, 2:3, :]
            ) / self.dy
        elif ny >= 2:
            dt_dy_b = (temperature_k[:, :, 1:2, :] - temperature_k[:, :, 0:1, :]) / self.dy
        else:
            dt_dy_b = torch.zeros_like(temperature_k[:, :, 0:1, :])
        q_out_b = k_field[:, :, 0:1, :] * dt_dy_b
        err_neu_b = ((q_out_b - neu_v_b) / self.flux_scale) ** 2
        neu_errors.append(neu_m_b * err_neu_b)
        neu_weights.append(neu_m_b)

        # North face: outward normal n = (0, +1) -> j_out = -k * (dT/dy)|_{y=Ly}
        neu_m_t = bc_channels[:, 6:7, -1:, :]
        neu_v_t = bc_channels[:, 7:8, -1:, :] * self.flux_scale
        if ny >= 3 and self.neumann_order == 2:
            minus_dt_dy_t = (
                -2.0 * temperature_k[:, :, -1:, :]
                + 3.0 * temperature_k[:, :, -2:-1, :]
                - temperature_k[:, :, -3:-2, :]
            ) / self.dy
        elif ny >= 2:
            minus_dt_dy_t = (temperature_k[:, :, -2:-1, :] - temperature_k[:, :, -1:, :]) / self.dy
        else:
            minus_dt_dy_t = torch.zeros_like(temperature_k[:, :, -1:, :])
        q_out_t = k_field[:, :, -1:, :] * minus_dt_dy_t
        err_neu_t = ((q_out_t - neu_v_t) / self.flux_scale) ** 2
        neu_errors.append(neu_m_t * err_neu_t)
        neu_weights.append(neu_m_t)

        total_neu_err = sum(torch.sum(e) for e in neu_errors)
        total_neu_w = sum(torch.sum(w) for w in neu_weights)
        loss_neumann = total_neu_err / torch.clamp(total_neu_w, min=1.0)

        # ------------------------------------------------------------------
        # 3. Global Heat Balance Loss
        # ------------------------------------------------------------------
        loss_flux_balance = torch.tensor(0.0, device=device)
        q_gen_val = 0.0
        q_out_val = 0.0
        imbalance_val = 0.0

        if self.weight_flux_balance > 0.0 and q_field is not None:
            # Reconstruct boundary heat flows matching FVM solver diagnostics
            bnd_fluxes = self.compute_boundary_fluxes(
                temperature_k=temperature_k,
                k_field=k_field,
                bc_channels=bc_channels,
                t_mean=t_mean,
                t_std=t_std,
            )
            q_out_total = (
                bnd_fluxes["left"]["q_flow"]
                + bnd_fluxes["right"]["q_flow"]
                + bnd_fluxes["bottom"]["q_flow"]
                + bnd_fluxes["top"]["q_flow"]
            )  # (B,)

            # Total volumetric heat generation in domain: (B,)
            q_gen_total = torch.sum(q_field * self.vol, dim=(-3, -2, -1))

            # Dimensionless balance error: |Q_out - Q_gen| / (Q_gen + Q_floor)
            char_q = torch.clamp(torch.abs(q_gen_total), min=self.q_floor)
            balance_error = ((q_out_total - q_gen_total) / char_q) ** 2
            loss_flux_balance = torch.mean(balance_error)

            q_gen_val = float(torch.mean(q_gen_total).detach().item())
            q_out_val = float(torch.mean(q_out_total).detach().item())
            imbalance_val = float(torch.mean(torch.abs(q_out_total - q_gen_total)).detach().item())

        # Total combined loss
        total_loss = (
            self.weight_dirichlet * loss_dirichlet
            + self.weight_neumann * loss_neumann
            + self.weight_flux_balance * loss_flux_balance
        )

        metrics = {
            "loss_bc_total": float(total_loss.detach().item()),
            "loss_dirichlet": float(loss_dirichlet.detach().item()),
            "loss_neumann": float(loss_neumann.detach().item()),
            "loss_flux_balance": float(loss_flux_balance.detach().item()),
            "q_gen_mean_w": q_gen_val,
            "q_out_mean_w": q_out_val,
            "flux_imbalance_mean_w": imbalance_val,
        }

        return total_loss, metrics


class FaceConsistentPhysicsLoss(nn.Module):
    """Compound objective coordinating supervised data loss and face-consistent boundary losses.

    Computes:
        loss_total = w_data * loss_data + w_bc * loss_bc
    where loss_bc enforces face-consistent Dirichlet temperature extrapolation,
    outward Neumann flux matching, and global heat balance.
    """

    is_physics_aware: bool = True

    def __init__(
        self,
        grid: Grid2D,
        normalizer: Any | None = None,
        data_weight: float = 1.0,
        bc_weight: float = 0.05,
        weight_dirichlet: float = 1.0,
        weight_neumann: float = 1.0,
        weight_flux_balance: float = 0.1,
        delta_t_char: float = 50.0,
        flux_scale: float = 1000.0,
        q_floor: float = 1.0,
    ) -> None:
        """Initialize FaceConsistentPhysicsLoss."""
        super().__init__()
        from neural_process_twin.training.loss import RelativeL2Loss

        self.grid = grid
        self.normalizer = normalizer
        self.data_weight = float(data_weight)
        self.bc_weight = float(bc_weight)
        self.data_loss_fn = RelativeL2Loss()

        self.bc_loss_fn = FaceConsistentBoundaryLoss(
            grid=grid,
            delta_t_char=delta_t_char,
            flux_scale=flux_scale,
            q_floor=q_floor,
            weight_dirichlet=weight_dirichlet,
            weight_neumann=weight_neumann,
            weight_flux_balance=weight_flux_balance,
        )

    def _get_t_scales(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self.normalizer is not None and getattr(self.normalizer, "t_stats", None) is not None:
            t_mean = torch.tensor(self.normalizer.t_stats.mean, dtype=torch.float32, device=device)
            t_std = torch.tensor(self.normalizer.t_stats.std, dtype=torch.float32, device=device)
        else:
            t_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
            t_std = torch.tensor(1.0, dtype=torch.float32, device=device)
        return t_mean, t_std

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute compound loss combining supervised relative L2 and face-consistent BC loss."""
        device = y_pred.device
        t_mean, t_std = self._get_t_scales(device)

        loss_data = self.data_loss_fn(y_pred, y_true)
        loss_total = self.data_weight * loss_data

        metrics = {
            "loss_total": float(loss_total.detach().item()),
            "loss_data": float(loss_data.detach().item()),
            "loss_bc": 0.0,
        }

        if self.bc_weight > 0.0 and batch is not None:
            x = batch["x"].to(device)
            if "k_raw" in batch:
                k_raw = batch["k_raw"].to(device)
            elif (
                self.normalizer is not None
                and getattr(self.normalizer, "k_stats", None) is not None
            ):
                k_raw = x[:, 0:1] * self.normalizer.k_stats.std + self.normalizer.k_stats.mean
            else:
                k_raw = x[:, 0:1]

            if "q_raw" in batch:
                q_raw = batch["q_raw"].to(device)
            elif (
                self.normalizer is not None
                and getattr(self.normalizer, "q_stats", None) is not None
            ):
                q_raw = x[:, 1:2] * self.normalizer.q_stats.std + self.normalizer.q_stats.mean
            else:
                q_raw = x[:, 1:2]

            bc_channels = (
                x[:, 4:12]
                if x.shape[1] >= 12
                else torch.zeros((x.shape[0], 8, x.shape[2], x.shape[3]), device=device)
            )
            t_pred_k = y_pred * t_std + t_mean

            loss_bc, bc_metrics = self.bc_loss_fn(
                temperature_k=t_pred_k,
                k_field=k_raw,
                bc_channels=bc_channels,
                q_field=q_raw,
                t_mean=t_mean,
                t_std=t_std,
            )

            loss_total = loss_total + self.bc_weight * loss_bc
            metrics["loss_total"] = float(loss_total.detach().item())
            metrics["loss_bc"] = float(loss_bc.detach().item())
            metrics.update(bc_metrics)

        return loss_total, metrics
