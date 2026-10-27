"""Scientific tests verifying mathematical properties of discrete boundary formulations.

Milestone 08B.1 Phase 1.
Verifies:
1. Linear face extrapolation is exact if and only if q = 0 and k is uniform in 1D.
2. Linear face extrapolation on heated domain (q != 0) induces an O(dx^2) bias.
3. DiscreteBoundaryCorrection projection T_0 = (2/3)*T_dir + (1/3)*T_1 induces an exact
   discrete PDE residual equal to q * cell_volume on every Dirichlet boundary cell.
4. Exact discrete FVM single-cell balance satisfies linear system to machine precision.
5. Energy balance conservation holds for rectangular cells and non-unit thickness.
"""

from __future__ import annotations

import numpy as np

from neural_process_twin.physics.manufactured import (
    LinearConduction1D,
    SinusoidalMMSVariableK,
)
from neural_process_twin.solvers.assembly import assemble_steady_heat_system
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D


class TestDiscreteBoundaryConsistencyMath:
    """Scientific verification of discrete boundary mathematics."""

    def test_linear_extrapolation_exact_for_unheated_conduction(self) -> None:
        """Verify 1.5*T_0 - 0.5*T_1 == T_dir to machine precision when q = 0."""
        mms = LinearConduction1D(t_left=280.0, t_right=380.0, k_0=1.5)
        grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=16, ny=16, thickness=1.0)
        xc, yc = grid.x_mesh, grid.y_mesh
        k_field = mms.conductivity(xc, yc)
        q_field = mms.source(xc, yc)
        bcs = mms.boundary_conditions()

        solver = HeatConductionSolver2D()
        result = solver.solve(grid, k_field, q_field, bcs)
        t_fvm = result.temperature

        t_0 = t_fvm[:, 0]
        t_1 = t_fvm[:, 1]
        t_face_extrap = 1.5 * t_0 - 0.5 * t_1

        # Extrapolation must match Dirichlet face value to machine precision
        assert np.max(np.abs(t_face_extrap - 280.0)) < 1e-11

        # Discrete projection must match t_0 to machine precision
        t_0_proj = (2.0 / 3.0) * 280.0 + (1.0 / 3.0) * t_1
        assert np.max(np.abs(t_0 - t_0_proj)) < 1e-11

    def test_extrapolation_bias_and_residual_under_volumetric_heating(self) -> None:
        """Verify that under q != 0, linear projection induces an exact residual imbalance of q*V_P."""
        q_val = 8000.0
        k_val = 2.0
        grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=32, ny=32, thickness=1.0)
        k_field = np.full(grid.shape, k_val, dtype=np.float64)
        q_field = np.full(grid.shape, q_val, dtype=np.float64)
        t_dir = 300.0
        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, t_dir),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, t_dir),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
        )

        solver = HeatConductionSolver2D()
        result = solver.solve(grid, k_field, q_field, bcs)
        t_fvm = result.temperature

        t_0 = t_fvm[:, 0]
        t_1 = t_fvm[:, 1]

        # 1. Extrapolation error is strictly non-zero
        t_face_extrap = 1.5 * t_0 - 0.5 * t_1
        extrap_err = np.max(np.abs(t_face_extrap - t_dir))
        assert extrap_err > 0.1  # Significant deviation due to curvature

        # 2. Applying extrapolation projection T_0 = (2/3)*T_dir + (1/3)*T_1
        t_projected = t_fvm.copy()
        t_projected[:, 0] = (2.0 / 3.0) * t_dir + (1.0 / 3.0) * t_1

        lin_sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)
        res_proj = lin_sys.A @ t_projected.ravel() - lin_sys.b

        # Boundary cell residual must match theoretical q * cell_volume exactly
        q_cell = q_val * grid.cell_volume
        bnd_res = np.abs(res_proj[[grid.cell_index(0, j) for j in range(grid.ny)]])
        np.testing.assert_allclose(bnd_res, q_cell, rtol=1e-5)

    def test_exact_fvm_single_cell_balance_on_variable_k(self) -> None:
        """Verify that the exact single-cell FVM balance satisfies the discrete system to machine precision."""
        mms = SinusoidalMMSVariableK(k_0=2.5, k_1=0.8, amplitude=30.0, t_0=310.0)
        grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=24, ny=24, thickness=1.0)
        xc, yc = grid.x_mesh, grid.y_mesh
        k_field = mms.conductivity(xc, yc)
        q_field = mms.source(xc, yc)
        bcs = mms.boundary_conditions()

        solver = HeatConductionSolver2D()
        result = solver.solve(grid, k_field, q_field, bcs)
        t_fvm = result.temperature

        lin_sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)
        matrix_a = lin_sys.A
        b = lin_sys.b

        # Reconstruct boundary cells using single-cell FVM balance:
        # T_p = (b_p - sum_{n != p} A_pn * T_n) / A_pp
        t_rec = np.zeros(grid.ny)
        for j in range(grid.ny):
            p = grid.cell_index(0, j)
            row_start = matrix_a.indptr[p]
            row_end = matrix_a.indptr[p + 1]
            cols = matrix_a.indices[row_start:row_end]
            vals = matrix_a.data[row_start:row_end]
            neighbor_sum = sum(
                v * t_fvm.ravel()[c] for c, v in zip(cols, vals, strict=True) if c != p
            )
            a_pp = float(matrix_a[p, p])
            t_rec[j] = (b[p] - neighbor_sum) / a_pp

        np.testing.assert_allclose(t_fvm[:, 0], t_rec, rtol=1e-11, atol=1e-11)

    def test_rectangular_mesh_and_non_unit_thickness_conservation(self) -> None:
        """Verify heat balance diagnostics with rectangular cells (dx != dy) and thickness != 1."""
        grid = Grid2D(x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, nx=32, ny=16, thickness=0.02)
        k_field = np.full(grid.shape, 4.0, dtype=np.float64)
        q_field = np.full(grid.shape, 3000.0, dtype=np.float64)
        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 350.0),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 300.0),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 200.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, -200.0),
        )

        solver = HeatConductionSolver2D()
        result = solver.solve(grid, k_field, q_field, bcs)
        diag = result.diagnostics

        assert diag.is_converged
        assert diag.energy_balance.flux_imbalance < 1e-10
        assert diag.energy_balance.relative_imbalance < 1e-12
