"""Boundary-Aware Fourier Neural Operator (BoundaryAwareFNO2D).

Addresses non-periodic boundary condition enforcement and Gibbs ringing on cell-centred grids:
1. Boundary-Conditioned Input Encoding: Integrates physical coordinates, normalized boundary
   conditions, and boundary distance transforms.
2. Local Boundary Residual Bypass: Complements periodic Fourier layers with local non-periodic
   convolutions with replicate padding to resolve steep boundary gradients without ringing.
3. Discrete Boundary Consistency Layer: Maps decoded temperatures through exact discrete
   half-cell FVM boundary projections:
   - Dirichlet faces: T_0 = (2/3)*T_dir + (1/3)*T_1 (exact linear face extrapolation).
   - Neumann faces: T_0 = 1.5*T_1 - 0.5*T_2 - (dx / (2*k))*q_out (exact 2nd-order flux gradient).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_process_twin.models.base import SurrogateModel
from neural_process_twin.models.fno2d import SpectralConv2D
from neural_process_twin.solvers.grid import Grid2D


class DiscreteBoundaryCorrection(nn.Module):
    """Exact discrete cell-centred finite-volume boundary reconstruction layer.

    Enforces face-consistent boundary conditions on cell-centred grids without violating
    interior continuity or crushing boundary conductances.
    """

    def __init__(
        self,
        grid: Grid2D,
        flux_scale: float = 1000.0,
        correction_mode: str = "hard",
        learnable_gate: bool = False,
    ) -> None:
        """Initialize DiscreteBoundaryCorrection.

        Args:
            grid: 2D computational domain grid.
            flux_scale: Characteristic scale [W/m^2] for Neumann flux decoding.
            correction_mode: 'hard' (strict discrete enforcement), 'soft' (gated residual),
                or 'none' (identity bypass).
            learnable_gate: If True, uses a learnable scalar parameter to control correction blend.
        """
        super().__init__()
        self.grid = grid
        self.dx = float(grid.dx)
        self.dy = float(grid.dy)
        self.nx = int(grid.nx)
        self.ny = int(grid.ny)
        self.flux_scale = float(flux_scale)
        self.correction_mode = correction_mode.lower()

        if learnable_gate:
            self.gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0.0) = 0.5
        else:
            self.register_parameter("gate", None)

    def _apply_fvm_balance(
        self,
        temperature_k: torch.Tensor,
        k_field: torch.Tensor,
        bc_channels: torch.Tensor,
        q_field: torch.Tensor | None = None,
        t_mean: float | torch.Tensor = 0.0,
        t_std: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Apply exact discrete single-cell finite volume balance to boundary perimeter cells."""
        dx, dy = self.dx, self.dy
        thick = float(self.grid.thickness)
        area_x = dy * thick
        area_y = dx * thick
        vol = dx * dy * thick
        eps = 1e-12

        # Unpack boundary channels
        dir_m_x = bc_channels[:, 0:1, :, :]
        dir_v_x = bc_channels[:, 1:2, :, :] * t_std + t_mean
        neu_m_x = bc_channels[:, 2:3, :, :]
        neu_v_x = bc_channels[:, 3:4, :, :] * self.flux_scale

        dir_m_y = bc_channels[:, 4:5, :, :]
        dir_v_y = bc_channels[:, 5:6, :, :] * t_std + t_mean
        neu_m_y = bc_channels[:, 6:7, :, :]
        neu_v_y = bc_channels[:, 7:8, :, :] * self.flux_scale

        q_src = q_field * vol if q_field is not None else torch.zeros_like(temperature_k)
        t_corr = temperature_k.clone()

        # 1. West boundary (i = 0)
        g_west = dir_m_x[:, :, :, 0:1] * (2.0 * k_field[:, :, :, 0:1] * area_x / dx)
        b_west = (
            dir_m_x[:, :, :, 0:1] * g_west * dir_v_x[:, :, :, 0:1]
            - neu_m_x[:, :, :, 0:1] * neu_v_x[:, :, :, 0:1] * area_x
        )

        k_w = k_field[:, :, :, 0:1]
        k_e = k_field[:, :, :, 1:2]
        k_east_face = 2.0 * k_w * k_e / (k_w + k_e + eps)
        g_east = k_east_face * (area_x / dx)
        s_east = g_east * temperature_k[:, :, :, 1:2]

        k_curr_w = k_field[:, :, :, 0:1]
        k_south_face_w = (
            2.0
            * k_curr_w[:, :, 1:-1, :]
            * k_curr_w[:, :, 0:-2, :]
            / (k_curr_w[:, :, 1:-1, :] + k_curr_w[:, :, 0:-2, :] + eps)
        )
        g_south_w = k_south_face_w * (area_y / dy)
        s_south_w = g_south_w * temperature_k[:, :, 0:-2, 0:1]

        k_north_face_w = (
            2.0
            * k_curr_w[:, :, 1:-1, :]
            * k_curr_w[:, :, 2:, :]
            / (k_curr_w[:, :, 1:-1, :] + k_curr_w[:, :, 2:, :] + eps)
        )
        g_north_w = k_north_face_w * (area_y / dy)
        s_north_w = g_north_w * temperature_k[:, :, 2:, 0:1]

        num_w = (
            b_west[:, :, 1:-1, :]
            + q_src[:, :, 1:-1, 0:1]
            + s_east[:, :, 1:-1, :]
            + s_south_w
            + s_north_w
        )
        den_w = g_west[:, :, 1:-1, :] + g_east[:, :, 1:-1, :] + g_south_w + g_north_w
        t_west_int = num_w / den_w

        # 2. East boundary (i = nx - 1)
        g_east_bnd = dir_m_x[:, :, :, -1:] * (2.0 * k_field[:, :, :, -1:] * area_x / dx)
        b_east_bnd = (
            dir_m_x[:, :, :, -1:] * g_east_bnd * dir_v_x[:, :, :, -1:]
            - neu_m_x[:, :, :, -1:] * neu_v_x[:, :, :, -1:] * area_x
        )

        k_curr_e = k_field[:, :, :, -1:]
        k_penult_e = k_field[:, :, :, -2:-1]
        k_west_face_e = 2.0 * k_curr_e * k_penult_e / (k_curr_e + k_penult_e + eps)
        g_west_int = k_west_face_e * (area_x / dx)
        s_west_int = g_west_int * temperature_k[:, :, :, -2:-1]

        k_s_face_e = (
            2.0
            * k_curr_e[:, :, 1:-1, :]
            * k_curr_e[:, :, 0:-2, :]
            / (k_curr_e[:, :, 1:-1, :] + k_curr_e[:, :, 0:-2, :] + eps)
        )
        g_south_e = k_s_face_e * (area_y / dy)
        s_south_e = g_south_e * temperature_k[:, :, 0:-2, -1:]

        k_n_face_e = (
            2.0
            * k_curr_e[:, :, 1:-1, :]
            * k_curr_e[:, :, 2:, :]
            / (k_curr_e[:, :, 1:-1, :] + k_curr_e[:, :, 2:, :] + eps)
        )
        g_north_e = k_n_face_e * (area_y / dy)
        s_north_e = g_north_e * temperature_k[:, :, 2:, -1:]

        num_e = (
            b_east_bnd[:, :, 1:-1, :]
            + q_src[:, :, 1:-1, -1:]
            + s_west_int[:, :, 1:-1, :]
            + s_south_e
            + s_north_e
        )
        den_e = g_east_bnd[:, :, 1:-1, :] + g_west_int[:, :, 1:-1, :] + g_south_e + g_north_e
        t_east_int = num_e / den_e

        # 3. South boundary (j = 0)
        g_south_bnd = dir_m_y[:, :, 0:1, :] * (2.0 * k_field[:, :, 0:1, :] * area_y / dy)
        b_south_bnd = (
            dir_m_y[:, :, 0:1, :] * g_south_bnd * dir_v_y[:, :, 0:1, :]
            - neu_m_y[:, :, 0:1, :] * neu_v_y[:, :, 0:1, :] * area_y
        )

        k_curr_s = k_field[:, :, 0:1, :]
        k_row1_s = k_field[:, :, 1:2, :]
        k_n_face_s = 2.0 * k_curr_s * k_row1_s / (k_curr_s + k_row1_s + eps)
        g_north_s = k_n_face_s * (area_y / dy)
        s_north_s = g_north_s * temperature_k[:, :, 1:2, :]

        k_w_face_s = (
            2.0
            * k_curr_s[:, :, :, 1:-1]
            * k_curr_s[:, :, :, 0:-2]
            / (k_curr_s[:, :, :, 1:-1] + k_curr_s[:, :, :, 0:-2] + eps)
        )
        g_west_s = k_w_face_s * (area_x / dx)
        s_west_s = g_west_s * temperature_k[:, :, 0:1, 0:-2]

        k_e_face_s = (
            2.0
            * k_curr_s[:, :, :, 1:-1]
            * k_curr_s[:, :, :, 2:]
            / (k_curr_s[:, :, :, 1:-1] + k_curr_s[:, :, :, 2:] + eps)
        )
        g_east_s = k_e_face_s * (area_x / dx)
        s_east_s = g_east_s * temperature_k[:, :, 0:1, 2:]

        num_s = (
            b_south_bnd[:, :, :, 1:-1]
            + q_src[:, :, 0:1, 1:-1]
            + s_north_s[:, :, :, 1:-1]
            + s_west_s
            + s_east_s
        )
        den_s = g_south_bnd[:, :, :, 1:-1] + g_north_s[:, :, :, 1:-1] + g_west_s + g_east_s
        t_south_int = num_s / den_s

        # 4. North boundary (j = ny - 1)
        g_north_bnd = dir_m_y[:, :, -1:, :] * (2.0 * k_field[:, :, -1:, :] * area_y / dy)
        b_north_bnd = (
            dir_m_y[:, :, -1:, :] * g_north_bnd * dir_v_y[:, :, -1:, :]
            - neu_m_y[:, :, -1:, :] * neu_v_y[:, :, -1:, :] * area_y
        )

        k_curr_n = k_field[:, :, -1:, :]
        k_penult_n = k_field[:, :, -2:-1, :]
        k_s_face_n = 2.0 * k_curr_n * k_penult_n / (k_curr_n + k_penult_n + eps)
        g_south_n = k_s_face_n * (area_y / dy)
        s_south_n = g_south_n * temperature_k[:, :, -2:-1, :]

        k_w_face_n = (
            2.0
            * k_curr_n[:, :, :, 1:-1]
            * k_curr_n[:, :, :, 0:-2]
            / (k_curr_n[:, :, :, 1:-1] + k_curr_n[:, :, :, 0:-2] + eps)
        )
        g_west_n = k_w_face_n * (area_x / dx)
        s_west_n = g_west_n * temperature_k[:, :, -1:, 0:-2]

        k_e_face_n = (
            2.0
            * k_curr_n[:, :, :, 1:-1]
            * k_curr_n[:, :, :, 2:]
            / (k_curr_n[:, :, :, 1:-1] + k_curr_n[:, :, :, 2:] + eps)
        )
        g_east_n = k_e_face_n * (area_x / dx)
        s_east_n = g_east_n * temperature_k[:, :, -1:, 2:]

        num_n = (
            b_north_bnd[:, :, :, 1:-1]
            + q_src[:, :, -1:, 1:-1]
            + s_south_n[:, :, :, 1:-1]
            + s_west_n
            + s_east_n
        )
        den_n = g_north_bnd[:, :, :, 1:-1] + g_south_n[:, :, :, 1:-1] + g_west_n + g_east_n
        t_north_int = num_n / den_n

        # Update edges
        t_corr[:, :, 1:-1, 0:1] = t_west_int
        t_corr[:, :, 1:-1, -1:] = t_east_int
        t_corr[:, :, 0:1, 1:-1] = t_south_int
        t_corr[:, :, -1:, 1:-1] = t_north_int

        # 5. Corners
        # SW corner (0, 0)
        c_sw_num = (
            b_west[:, :, 0:1, 0:1]
            + b_south_bnd[:, :, 0:1, 0:1]
            + q_src[:, :, 0:1, 0:1]
            + g_east[:, :, 0:1, 0:1] * temperature_k[:, :, 0:1, 1:2]
            + g_north_s[:, :, 0:1, 0:1] * temperature_k[:, :, 1:2, 0:1]
        )
        c_sw_den = (
            g_west[:, :, 0:1, 0:1]
            + g_south_bnd[:, :, 0:1, 0:1]
            + g_east[:, :, 0:1, 0:1]
            + g_north_s[:, :, 0:1, 0:1]
        )
        t_corr[:, :, 0:1, 0:1] = c_sw_num / c_sw_den

        # SE corner (nx-1, 0)
        c_se_num = (
            b_east_bnd[:, :, 0:1, 0:1]
            + b_south_bnd[:, :, 0:1, -1:]
            + q_src[:, :, 0:1, -1:]
            + g_west_int[:, :, 0:1, 0:1] * temperature_k[:, :, 0:1, -2:-1]
            + g_north_s[:, :, 0:1, -1:] * temperature_k[:, :, 1:2, -1:]
        )
        c_se_den = (
            g_east_bnd[:, :, 0:1, 0:1]
            + g_south_bnd[:, :, 0:1, -1:]
            + g_west_int[:, :, 0:1, 0:1]
            + g_north_s[:, :, 0:1, -1:]
        )
        t_corr[:, :, 0:1, -1:] = c_se_num / c_se_den

        # NW corner (0, ny-1)
        c_nw_num = (
            b_west[:, :, -1:, 0:1]
            + b_north_bnd[:, :, 0:1, 0:1]
            + q_src[:, :, -1:, 0:1]
            + g_east[:, :, -1:, 0:1] * temperature_k[:, :, -1:, 1:2]
            + g_south_n[:, :, 0:1, 0:1] * temperature_k[:, :, -2:-1, 0:1]
        )
        c_nw_den = (
            g_west[:, :, -1:, 0:1]
            + g_north_bnd[:, :, 0:1, 0:1]
            + g_east[:, :, -1:, 0:1]
            + g_south_n[:, :, 0:1, 0:1]
        )
        t_corr[:, :, -1:, 0:1] = c_nw_num / c_nw_den

        # NE corner (nx-1, ny-1)
        c_ne_num = (
            b_east_bnd[:, :, -1:, 0:1]
            + b_north_bnd[:, :, 0:1, -1:]
            + q_src[:, :, -1:, -1:]
            + g_west_int[:, :, -1:, 0:1] * temperature_k[:, :, -1:, -2:-1]
            + g_south_n[:, :, 0:1, -1:] * temperature_k[:, :, -2:-1, -1:]
        )
        c_ne_den = (
            g_east_bnd[:, :, -1:, 0:1]
            + g_north_bnd[:, :, 0:1, -1:]
            + g_west_int[:, :, -1:, 0:1]
            + g_south_n[:, :, 0:1, -1:]
        )
        t_corr[:, :, -1:, -1:] = c_ne_num / c_ne_den

        return t_corr

    def forward(
        self,
        temperature_k: torch.Tensor,
        k_field: torch.Tensor,
        bc_channels: torch.Tensor,
        q_field: torch.Tensor | None = None,
        t_mean: float | torch.Tensor = 0.0,
        t_std: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Apply discrete boundary correction to predicted temperature field.

        Args:
            temperature_k: Predicted temperature tensor in physical Kelvin [K] (B, 1, ny, nx).
            k_field: Physical thermal conductivity [W/(m K)] (B, 1, ny, nx).
            bc_channels: Encoded 8-channel boundary tensor (B, 8, ny, nx).
            q_field: Optional physical volumetric heat source [W/m^3] (B, 1, ny, nx).
            t_mean: Temperature normalizer mean [K].
            t_std: Temperature normalizer std [K].

        Returns:
            Boundary-corrected temperature field in Kelvin [K] of shape (B, 1, ny, nx).
        """
        if self.correction_mode == "none":
            return temperature_k

        if temperature_k.shape[3] < 3 or temperature_k.shape[2] < 3:
            return temperature_k

        if self.correction_mode == "fvm_balance":
            return self._apply_fvm_balance(
                temperature_k=temperature_k,
                k_field=k_field,
                bc_channels=bc_channels,
                q_field=q_field,
                t_mean=t_mean,
                t_std=t_std,
            )

        # Unpack boundary channels
        dir_m_x = bc_channels[:, 0:1, :, :]
        dir_v_x = bc_channels[:, 1:2, :, :] * t_std + t_mean
        neu_m_x = bc_channels[:, 2:3, :, :]
        neu_v_x = bc_channels[:, 3:4, :, :] * self.flux_scale

        dir_m_y = bc_channels[:, 4:5, :, :]
        dir_v_y = bc_channels[:, 5:6, :, :] * t_std + t_mean
        neu_m_y = bc_channels[:, 6:7, :, :]
        neu_v_y = bc_channels[:, 7:8, :, :] * self.flux_scale

        # Clone input to avoid in-place autograd corruption
        t_corr = temperature_k.clone()

        # 1. West boundary (i = 0):
        # Dirichlet: T_0 = (2/3)*T_dir + (1/3)*T_1 -> T_face = 1.5*T_0 - 0.5*T_1 = T_dir
        t_west_dir = (2.0 / 3.0) * dir_v_x[:, :, :, 0:1] + (1.0 / 3.0) * temperature_k[:, :, :, 1:2]
        # Neumann: Outward flux j_out = +k*(dT/dx) -> T_0 = 1.5*T_1 - 0.5*T_2 - (dx / (2*k))*j_out
        k_w = k_field[:, :, :, 0:1]
        t_west_neu = (
            1.5 * temperature_k[:, :, :, 1:2]
            - 0.5 * temperature_k[:, :, :, 2:3]
            - (self.dx / (2.0 * k_w)) * neu_v_x[:, :, :, 0:1]
        )
        t_west = dir_m_x[:, :, :, 0:1] * t_west_dir + neu_m_x[:, :, :, 0:1] * t_west_neu

        # 2. East boundary (i = nx - 1):
        t_east_dir = (2.0 / 3.0) * dir_v_x[:, :, :, -1:] + (1.0 / 3.0) * temperature_k[
            :, :, :, -2:-1
        ]
        k_e = k_field[:, :, :, -1:]
        t_east_neu = (
            1.5 * temperature_k[:, :, :, -2:-1]
            - 0.5 * temperature_k[:, :, :, -3:-2]
            - (self.dx / (2.0 * k_e)) * neu_v_x[:, :, :, -1:]
        )
        t_east = dir_m_x[:, :, :, -1:] * t_east_dir + neu_m_x[:, :, :, -1:] * t_east_neu

        # 3. South boundary (j = 0):
        t_south_dir = (2.0 / 3.0) * dir_v_y[:, :, 0:1, :] + (1.0 / 3.0) * temperature_k[
            :, :, 1:2, :
        ]
        k_s = k_field[:, :, 0:1, :]
        t_south_neu = (
            1.5 * temperature_k[:, :, 1:2, :]
            - 0.5 * temperature_k[:, :, 2:3, :]
            - (self.dy / (2.0 * k_s)) * neu_v_y[:, :, 0:1, :]
        )
        t_south = dir_m_y[:, :, 0:1, :] * t_south_dir + neu_m_y[:, :, 0:1, :] * t_south_neu

        # 4. North boundary (j = ny - 1):
        t_north_dir = (2.0 / 3.0) * dir_v_y[:, :, -1:, :] + (1.0 / 3.0) * temperature_k[
            :, :, -2:-1, :
        ]
        k_n = k_field[:, :, -1:, :]
        t_north_neu = (
            1.5 * temperature_k[:, :, -2:-1, :]
            - 0.5 * temperature_k[:, :, -3:-2, :]
            - (self.dy / (2.0 * k_n)) * neu_v_y[:, :, -1:, :]
        )
        t_north = dir_m_y[:, :, -1:, :] * t_north_dir + neu_m_y[:, :, -1:, :] * t_north_neu

        # Blending factor
        if self.gate is not None:
            alpha = torch.sigmoid(self.gate)
        elif self.correction_mode == "soft":
            alpha = 0.5
        else:
            alpha = 1.0

        # Apply reconstructions to West/East columns (excluding corners first)
        t_corr[:, :, 1:-1, 0:1] = (1.0 - alpha) * temperature_k[:, :, 1:-1, 0:1] + alpha * t_west[
            :, :, 1:-1, :
        ]
        t_corr[:, :, 1:-1, -1:] = (1.0 - alpha) * temperature_k[:, :, 1:-1, -1:] + alpha * t_east[
            :, :, 1:-1, :
        ]

        # Apply reconstructions to South/North rows (excluding corners first)
        t_corr[:, :, 0:1, 1:-1] = (1.0 - alpha) * temperature_k[:, :, 0:1, 1:-1] + alpha * t_south[
            :, :, :, 1:-1
        ]
        t_corr[:, :, -1:, 1:-1] = (1.0 - alpha) * temperature_k[:, :, -1:, 1:-1] + alpha * t_north[
            :, :, :, 1:-1
        ]

        # Corner cells: harmonic/average resolution between x-normal and y-normal boundaries
        c_sw = 0.5 * (t_west[:, :, 0:1, :] + t_south[:, :, :, 0:1])
        c_se = 0.5 * (t_east[:, :, 0:1, :] + t_south[:, :, :, -1:])
        c_nw = 0.5 * (t_west[:, :, -1:, :] + t_north[:, :, :, 0:1])
        c_ne = 0.5 * (t_east[:, :, -1:, :] + t_north[:, :, :, -1:])

        t_corr[:, :, 0:1, 0:1] = (1.0 - alpha) * temperature_k[:, :, 0:1, 0:1] + alpha * c_sw
        t_corr[:, :, 0:1, -1:] = (1.0 - alpha) * temperature_k[:, :, 0:1, -1:] + alpha * c_se
        t_corr[:, :, -1:, 0:1] = (1.0 - alpha) * temperature_k[:, :, -1:, 0:1] + alpha * c_nw
        t_corr[:, :, -1:, -1:] = (1.0 - alpha) * temperature_k[:, :, -1:, -1:] + alpha * c_ne

        return t_corr


class BoundaryAwareFNO2D(SurrogateModel):
    """Boundary-Aware Fourier Neural Operator (BA-FNO-2D).

    Combines:
    - Fourier Neural Operator spectral trunk for resolution-invariant continuous operator learning.
    - Local non-periodic residual CNN bypass to eliminate boundary ringing.
    - Discrete FVM boundary correction layer for face-consistent boundary flux matching.
    """

    def __init__(
        self,
        grid: Grid2D | None = None,
        in_channels: int = 12,
        out_channels: int = 1,
        modes_x: int = 12,
        modes_y: int = 12,
        width: int = 32,
        depth: int = 4,
        activation: str = "gelu",
        use_boundary_bypass: bool = True,
        correction_mode: str = "hard",
        learnable_gate: bool = False,
        flux_scale: float = 1000.0,
    ) -> None:
        """Initialize BoundaryAwareFNO2D.

        Args:
            grid: Grid2D domain geometry specification.
            in_channels: Input channels (default 12 for canonical representation).
            out_channels: Output channels (default 1 for temperature).
            modes_x: Retained Fourier modes along x.
            modes_y: Retained Fourier modes along y.
            width: Hidden latent channel dimension.
            depth: Number of Fourier neural operator layers.
            activation: Activation function ('gelu' or 'relu').
            use_boundary_bypass: If True, adds local non-periodic CNN residual branch.
            correction_mode: Discrete correction mode ('hard', 'soft', or 'none').
            learnable_gate: If True, learns correction gating parameter.
            flux_scale: Characteristic scale for boundary flux decoding.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.grid = grid or Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=32, ny=32)
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.depth = depth
        self.use_boundary_bypass = use_boundary_bypass
        self.correction_mode = correction_mode

        # 1. Lifting projection
        self.p = nn.Conv2d(in_channels, width, kernel_size=1)
        nn.init.kaiming_normal_(self.p.weight, nonlinearity="relu")
        if self.p.bias is not None:
            nn.init.zeros_(self.p.bias)

        # 2. Spectral Fourier layers and local linear bypass
        self.spectral_layers = nn.ModuleList()
        self.bypass_layers = nn.ModuleList()
        for _ in range(depth):
            self.spectral_layers.append(
                SpectralConv2D(
                    in_channels=width,
                    out_channels=width,
                    modes_x=modes_x,
                    modes_y=modes_y,
                )
            )
            bypass = nn.Conv2d(width, width, kernel_size=1)
            nn.init.kaiming_normal_(bypass.weight, nonlinearity="relu")
            if bypass.bias is not None:
                nn.init.zeros_(bypass.bias)
            self.bypass_layers.append(bypass)

        # 3. Local boundary-localized CNN residual branch (replicate padding to preserve boundary values)
        if use_boundary_bypass:
            self.boundary_branch = nn.Sequential(
                nn.Conv2d(width, width, kernel_size=3, padding=1, padding_mode="replicate"),
                nn.GELU() if activation.lower() == "gelu" else nn.ReLU(inplace=True),
                nn.Conv2d(width, width, kernel_size=3, padding=1, padding_mode="replicate"),
            )
            nn.init.zeros_(self.boundary_branch[2].weight)
            nn.init.zeros_(self.boundary_branch[2].bias)
        else:
            self.boundary_branch = None

        # Activation
        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

        # 4. Projection decoder: Conv2D 1x1 -> Activation -> Conv2D 1x1
        self.q0 = nn.Conv2d(width, 128, kernel_size=1)
        self.q1 = nn.Conv2d(128, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.q0.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.q1.weight)
        if self.q0.bias is not None:
            nn.init.zeros_(self.q0.bias)
        if self.q1.bias is not None:
            nn.init.zeros_(self.q1.bias)

        # 5. Discrete Boundary Correction layer
        self.boundary_correction = DiscreteBoundaryCorrection(
            grid=self.grid,
            flux_scale=flux_scale,
            correction_mode=correction_mode,
            learnable_gate=learnable_gate,
        )

    def forward(
        self,
        x: torch.Tensor,
        k_raw: torch.Tensor | None = None,
        q_raw: torch.Tensor | None = None,
        t_mean: float | torch.Tensor = 0.0,
        t_std: float | torch.Tensor = 1.0,
        apply_boundary_correction: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, ny, nx).
            k_raw: Optional physical conductivity tensor [W/(m K)] (B, 1, ny, nx).
            q_raw: Optional physical heat source tensor [W/m^3] (B, 1, ny, nx).
            t_mean: Normalizer temperature mean [K].
            t_std: Normalizer temperature std [K].
            apply_boundary_correction: If True and physical metadata available, applies
                discrete boundary correction layer in normalized space.

        Returns:
            Normalized predicted temperature field of shape (batch, 1, ny, nx).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (batch, C, ny, nx), got {x.dim()}D.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.in_channels} channels, got {x.shape[1]}."
            )

        # 1. Lifting
        h = self.p(x)

        # 2. Fourier trunk
        h_fourier = h
        for spec_conv, bypass in zip(self.spectral_layers, self.bypass_layers, strict=True):
            h_spec = spec_conv(h_fourier)
            h_lin = bypass(h_fourier)
            h_fourier = self.act(h_spec + h_lin)

        # 3. Add boundary residual branch if enabled
        if self.boundary_branch is not None:
            h_bnd = self.boundary_branch(h)
            h_out = h_fourier + h_bnd
        else:
            h_out = h_fourier

        # 4. Project to output
        h_out = self.act(self.q0(h_out))
        y_pred = self.q1(h_out)  # Normalized output

        # 5. Optional discrete boundary correction in physical space
        if apply_boundary_correction and self.correction_mode != "none" and x.shape[1] >= 12:
            # Extract raw conductivity and source or reconstruct from normalized channels
            k_est = x[:, 0:1] if k_raw is None else k_raw
            q_est = x[:, 1:2] if q_raw is None else q_raw
            bc_channels = x[:, 4:12]

            # Reversibly project to Kelvin
            t_pred_k = y_pred * t_std + t_mean

            # Apply discrete correction
            t_corr_k = self.boundary_correction(
                temperature_k=t_pred_k,
                k_field=k_est,
                bc_channels=bc_channels,
                q_field=q_est,
                t_mean=t_mean,
                t_std=t_std,
            )

            # Re-normalize back to model output space
            y_pred = (t_corr_k - t_mean) / t_std

        return y_pred
