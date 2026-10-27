"""Unit and regression tests verifying Neumann boundary flux signs across all 4 domain edges.

Milestone 08B.1 Phase 2: Neumann Sign Correction Audit.
Verifies:
1. West (x = 0, normal n = (-1, 0)): j_out = +k * (dT/dx)|_{x=0}
2. East (x = Lx, normal n = (+1, 0)): j_out = -k * (dT/dx)|_{x=Lx}
3. South (y = 0, normal n = (0, -1)): j_out = +k * (dT/dy)|_{y=0}
4. North (y = Ly, normal n = (0, +1)): j_out = -k * (dT/dy)|_{y=Ly}
5. Explicitly demonstrates that historical BoundaryConditionLoss has inverted signs on all four edges.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_process_twin.data.boundary_encoding import encode_boundary_conditions
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.training.face_consistent_loss import FaceConsistentBoundaryLoss
from neural_process_twin.training.physics_loss import BoundaryConditionLoss


class TestNeumannSignConventions:
    """Rigorous verification of Neumann outward flux signs against analytical solutions."""

    @classmethod
    def setup_class(cls) -> None:
        cls.grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=16, ny=16, thickness=1.0)
        cls.k_val = 2.0
        cls.k_tensor = torch.full((1, 1, 16, 16), cls.k_val, dtype=torch.float32)

    def test_west_boundary_outward_heat_flux_sign(self) -> None:
        """West (x=0, n=(-1, 0)): T(x) = T_0 + G*x -> j_out = +k * G."""
        gx = 50.0  # Temperature increases to the right -> heat flows left (outwards across West)
        j_expected_west = self.k_val * gx  # +100 W/m^2

        xc = torch.from_numpy(self.grid.x_mesh).float().unsqueeze(0).unsqueeze(0)
        temp = 300.0 + gx * xc

        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.NEUMANN, j_expected_west),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, float(300.0 + gx)),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
        )
        bc_channels = (
            torch.from_numpy(encode_boundary_conditions(bcs, self.grid)).float().unsqueeze(0)
        )

        # 1. FaceConsistentBoundaryLoss must have near-zero Neumann error
        loss_fn = FaceConsistentBoundaryLoss(grid=self.grid, neumann_order=1, flux_scale=1000.0)
        _, metrics = loss_fn(temperature_k=temp, k_field=self.k_tensor, bc_channels=bc_channels)
        assert metrics["loss_neumann"] < 1e-6

        # 2. Historical BoundaryConditionLoss computes k * (T_0 - T_1)/dx = -k * G = -100 W/m^2
        # Expected error under historical loss is (( -100 - 100 ) / 1000)^2 = (-0.2)^2 = 0.04
        hist_loss_fn = BoundaryConditionLoss(grid=self.grid, flux_scale=1000.0)
        hist_loss = hist_loss_fn(temperature_k=temp, k_field=self.k_tensor, bc_channels=bc_channels)
        assert hist_loss.item() > 0.01  # Demonstrates sign inversion error in historical module

    def test_east_boundary_outward_heat_flux_sign(self) -> None:
        """East (x=1, n=(+1, 0)): T(x) = T_0 - G*(x - 1) -> j_out = -k * (-G) = +k * G."""
        # Temperature drops to the right -> heat flows right (outwards across East)
        gx = 40.0
        j_expected_east = self.k_val * gx  # +80 W/m^2

        xc = torch.from_numpy(self.grid.x_mesh).float().unsqueeze(0).unsqueeze(0)
        temp = 300.0 - gx * (xc - 1.0)  # At x=1, temp=300; at x=0, temp=300+gx

        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, float(300.0 + gx)),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, j_expected_east),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
        )
        bc_channels = (
            torch.from_numpy(encode_boundary_conditions(bcs, self.grid)).float().unsqueeze(0)
        )

        loss_fn = FaceConsistentBoundaryLoss(grid=self.grid, neumann_order=1, flux_scale=1000.0)
        _, metrics = loss_fn(temperature_k=temp, k_field=self.k_tensor, bc_channels=bc_channels)
        assert metrics["loss_neumann"] < 1e-6

        # Historical module computes +k * (T_last - T_penult)/dx = -80 W/m^2
        hist_loss_fn = BoundaryConditionLoss(grid=self.grid, flux_scale=1000.0)
        hist_loss = hist_loss_fn(temperature_k=temp, k_field=self.k_tensor, bc_channels=bc_channels)
        assert hist_loss.item() > 0.005

    def test_south_boundary_outward_heat_flux_sign(self) -> None:
        """South (y=0, n=(0, -1)): T(y) = T_0 + Gy*y -> j_out = +k * Gy."""
        gy = 60.0
        j_expected_south = self.k_val * gy  # +120 W/m^2

        yc = torch.from_numpy(self.grid.y_mesh).float().unsqueeze(0).unsqueeze(0)
        temp = 290.0 + gy * yc

        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.NEUMANN, 0.0),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, 0.0),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, j_expected_south),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, float(290.0 + gy)),
        )
        bc_channels = (
            torch.from_numpy(encode_boundary_conditions(bcs, self.grid)).float().unsqueeze(0)
        )

        loss_fn = FaceConsistentBoundaryLoss(grid=self.grid, neumann_order=1, flux_scale=1000.0)
        _, metrics = loss_fn(temperature_k=temp, k_field=self.k_tensor, bc_channels=bc_channels)
        assert metrics["loss_neumann"] < 1e-6

        hist_loss_fn = BoundaryConditionLoss(grid=self.grid, flux_scale=1000.0)
        hist_loss = hist_loss_fn(temperature_k=temp, k_field=self.k_tensor, bc_channels=bc_channels)
        assert hist_loss.item() > 0.005

    def test_north_boundary_outward_heat_flux_sign(self) -> None:
        """North (y=1, n=(0, +1)): T(y) = T_0 - Gy*(y - 1) -> j_out = -k * (-Gy) = +k * Gy."""
        gy = 45.0
        j_expected_north = self.k_val * gy  # +90 W/m^2

        yc = torch.from_numpy(self.grid.y_mesh).float().unsqueeze(0).unsqueeze(0)
        temp = 290.0 - gy * (yc - 1.0)

        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.NEUMANN, 0.0),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, 0.0),
            bottom=BoundaryCondition(
                BoundarySide.BOTTOM, BoundaryType.DIRICHLET, float(290.0 + gy)
            ),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, j_expected_north),
        )
        bc_channels = (
            torch.from_numpy(encode_boundary_conditions(bcs, self.grid)).float().unsqueeze(0)
        )

        loss_fn = FaceConsistentBoundaryLoss(grid=self.grid, neumann_order=1, flux_scale=1000.0)
        _, metrics = loss_fn(temperature_k=temp, k_field=self.k_tensor, bc_channels=bc_channels)
        assert metrics["loss_neumann"] < 1e-6

        hist_loss_fn = BoundaryConditionLoss(grid=self.grid, flux_scale=1000.0)
        hist_loss = hist_loss_fn(temperature_k=temp, k_field=self.k_tensor, bc_channels=bc_channels)
        assert hist_loss.item() > 0.005

    def test_reference_solver_diagnostics_flux_consistency(self) -> None:
        """Verify that diagnostics.py outward heat flow matches FaceConsistentBoundaryLoss."""
        # Constant gradient along x: T(x) = 300 + 50*x
        gx = 50.0
        j_west = self.k_val * gx  # +100 W/m^2 leaving West
        j_east = -self.k_val * gx  # -100 W/m^2 leaving East (inward)
        temp_np = 300.0 + gx * self.grid.x_mesh
        k_np = np.full(self.grid.shape, self.k_val, dtype=np.float64)
        q_np = np.zeros(self.grid.shape, dtype=np.float64)

        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.NEUMANN, j_west),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, j_east),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
        )

        eb = compute_heat_balance(self.grid, temp_np, k_np, q_np, bcs)
        total_area_x = self.grid.face_area_x * self.grid.ny
        assert pytest.approx(eb.boundary_heat_flow_out["left"], rel=1e-5) == j_west * total_area_x
        assert pytest.approx(eb.boundary_heat_flow_out["right"], rel=1e-5) == j_east * total_area_x
        assert pytest.approx(eb.flux_imbalance, abs=1e-10) == 0.0
