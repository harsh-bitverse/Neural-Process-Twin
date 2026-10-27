"""Audit script comparing discrete boundary formulations against reference FVM solver.

Milestone 08B.1 Phase 1: Discrete Boundary Consistency Analysis.
Compares:
1. Exact FVM boundary face flux: j_out = (2 * k / dx) * (T_0 - T_dir)
2. Linear face extrapolation: T_face = 1.5 * T_0 - 0.5 * T_1
3. DiscreteBoundaryCorrection projection: T_0 = (2/3) * T_dir + (1/3) * T_1
4. Exact discrete FVM cell balance: (A * T - b)_0 = 0
"""

from __future__ import annotations

import json
from pathlib import Path

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


def audit_linear_conduction_1d() -> dict:
    """Case 1: 1D linear heat conduction (q = 0, k = const)."""
    mms = LinearConduction1D(t_left=300.0, t_right=400.0, k_0=2.0)
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=16, ny=16, thickness=1.0)
    xc, yc = grid.x_mesh, grid.y_mesh
    k_field = mms.conductivity(xc, yc)
    q_field = mms.source(xc, yc)
    bcs = mms.boundary_conditions()

    solver = HeatConductionSolver2D()
    result = solver.solve(grid, k_field, q_field, bcs)
    t_fvm = result.temperature

    dx = grid.dx
    t_0 = t_fvm[:, 0]
    t_1 = t_fvm[:, 1]
    t_dir = 300.0

    # 1. Linear face extrapolation
    t_face_extrap = 1.5 * t_0 - 0.5 * t_1
    err_face_extrap = np.max(np.abs(t_face_extrap - t_dir))

    # 2. Discrete projection
    t_0_proj = (2.0 / 3.0) * t_dir + (1.0 / 3.0) * t_1
    err_t0_proj = np.max(np.abs(t_0 - t_0_proj))

    # 3. FVM West face flux
    j_fvm_west = (2.0 * k_field[:, 0] / dx) * (t_0 - t_dir)
    # Outward normal at West is n = (-1, 0) -> j_out = q . n = (-k dT/dx) * (-1) = +k * (T_0 - T_dir) / (dx / 2) = +200 W/m^2
    j_exact_west = k_field[:, 0] * (400.0 - 300.0) / 1.0
    err_flux_west = np.max(np.abs(j_fvm_west - j_exact_west))

    # 4. Residual check
    lin_sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)
    res = lin_sys.A @ t_fvm.ravel() - lin_sys.b
    max_res = float(np.max(np.abs(res)))

    return {
        "case": "1D Linear (q=0, k=const)",
        "err_face_extrap_k": float(err_face_extrap),
        "err_t0_proj_k": float(err_t0_proj),
        "err_flux_west_w_per_m2": float(err_flux_west),
        "max_fvm_residual_w": max_res,
        "is_extrapolation_exact": bool(err_face_extrap < 1e-10),
    }


def audit_quadratic_conduction_1d(q_val: float = 10000.0, k_val: float = 2.0) -> dict:
    """Case 2: 1D conduction with constant heat generation (q != 0, k = const).

    T''(x) = -q/k -> T(x) = T_0 + (T_1 - T_0)*x + (q/(2k)) * x * (1 - x).
    """
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=32, ny=32, thickness=1.0)
    k_field = np.full(grid.shape, k_val, dtype=np.float64)
    q_field = np.full(grid.shape, q_val, dtype=np.float64)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 300.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )

    solver = HeatConductionSolver2D()
    result = solver.solve(grid, k_field, q_field, bcs)
    t_fvm = result.temperature

    dx = grid.dx
    t_0 = t_fvm[:, 0]
    t_1 = t_fvm[:, 1]
    t_dir = 300.0

    # 1. Linear face extrapolation error
    t_face_extrap = 1.5 * t_0 - 0.5 * t_1
    err_face_extrap = np.max(np.abs(t_face_extrap - t_dir))
    # Theoretical prediction: 3 * q * dx^2 / (8 * k)
    theoretical_extrap_bias = (3.0 * q_val * (dx**2)) / (8.0 * k_val)

    # 2. Discrete projection error: T_0 - ((2/3)*T_dir + (1/3)*T_1)
    t_0_proj = (2.0 / 3.0) * t_dir + (1.0 / 3.0) * t_1
    err_t0_proj = np.max(np.abs(t_0 - t_0_proj))
    theoretical_t0_bias = (q_val * (dx**2)) / (4.0 * k_val)

    # 3. Residual when forcing projection T_0_proj into FVM matrix
    t_projected = t_fvm.copy()
    t_projected[:, 0] = t_0_proj
    lin_sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)
    res_orig = lin_sys.A @ t_fvm.ravel() - lin_sys.b
    res_proj = lin_sys.A @ t_projected.ravel() - lin_sys.b

    # Boundary cell residual on West edge (i=0):
    bnd_cells = [grid.cell_index(0, j) for j in range(grid.ny)]
    bnd_res_proj = np.abs(res_proj[bnd_cells])

    # Theoretical boundary cell residual = q * cell_volume
    q_cell = q_val * grid.cell_volume

    return {
        "case": "1D Quadratic (q!=0, k=const)",
        "q_val_w_m3": q_val,
        "k_val_w_mk": k_val,
        "dx": float(dx),
        "err_face_extrap_measured_k": float(err_face_extrap),
        "err_face_extrap_theory_k": float(theoretical_extrap_bias),
        "err_t0_proj_measured_k": float(err_t0_proj),
        "err_t0_proj_theory_k": float(theoretical_t0_bias),
        "original_max_residual_w": float(np.max(np.abs(res_orig))),
        "projected_boundary_cell_residual_mean_w": float(np.mean(bnd_res_proj)),
        "theoretical_projected_residual_w": float(q_cell),
        "is_extrapolation_exact": False,
    }


def audit_variable_k_2d_mms() -> dict:
    """Case 3: 2D manufactured solution with spatially varying conductivity."""
    mms = SinusoidalMMSVariableK(k_0=2.0, k_1=0.5, amplitude=25.0, t_0=300.0)
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=32, ny=32, thickness=1.0)
    xc, yc = grid.x_mesh, grid.y_mesh
    k_field = mms.conductivity(xc, yc)
    q_field = mms.source(xc, yc)
    bcs = mms.boundary_conditions()

    solver = HeatConductionSolver2D()
    result = solver.solve(grid, k_field, q_field, bcs)
    t_fvm = result.temperature

    # Reconstruct West Dirichlet face
    t_0 = t_fvm[:, 0]
    t_1 = t_fvm[:, 1]
    t_dir = 300.0

    t_face_extrap = 1.5 * t_0 - 0.5 * t_1
    err_face_extrap = np.max(np.abs(t_face_extrap - t_dir))

    t_0_proj = (2.0 / 3.0) * t_dir + (1.0 / 3.0) * t_1
    err_t0_proj = np.max(np.abs(t_0 - t_0_proj))

    # Exact discrete FVM boundary update formula
    # T_0,j = (G_west * T_dir + q * V_P + G_east * T_1 + G_south * T_0,j-1 + G_north * T_0,j+1) / sum(G)
    lin_sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)
    matrix_a = lin_sys.A
    b = lin_sys.b

    # Check that exact FVM cell satisfies A * T - b = 0 down to machine precision
    res = matrix_a @ t_fvm.ravel() - b

    # For each boundary cell, compute its exact single-cell FVM balance:
    # (A*T - b)_p = A_pp * T_p + sum_{n!=p} A_pn * T_n - b_p = 0
    # => T_p_exact = (b_p - sum_{n!=p} A_pn * T_n) / A_pp
    t_exact_balance = np.zeros(grid.ny)
    for j in range(grid.ny):
        p = grid.cell_index(0, j)
        row_start = matrix_a.indptr[p]
        row_end = matrix_a.indptr[p + 1]
        col_indices = matrix_a.indices[row_start:row_end]
        row_data = matrix_a.data[row_start:row_end]

        neighbor_sum = 0.0
        a_pp = 0.0
        for c, v in zip(col_indices, row_data, strict=True):
            if c == p:
                a_pp = v
            else:
                neighbor_sum += v * t_fvm.ravel()[c]
        t_exact_balance[j] = (b[p] - neighbor_sum) / a_pp

    err_exact_balance = np.max(np.abs(t_0 - t_exact_balance))

    return {
        "case": "2D Sinusoidal MMS Variable K",
        "err_face_extrap_max_k": float(err_face_extrap),
        "err_t0_proj_max_k": float(err_t0_proj),
        "err_exact_fvm_balance_k": float(err_exact_balance),
        "max_fvm_residual_w": float(np.max(np.abs(res))),
        "is_extrapolation_exact": False,
        "is_fvm_balance_exact": bool(err_exact_balance < 1e-12),
    }


def audit_rectangular_cells_and_thickness() -> dict:
    """Case 4: Rectangular cells (dx != dy) and non-unit slab thickness (t != 1.0)."""
    grid = Grid2D(x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, nx=32, ny=32, thickness=0.05)
    k_field = np.full(grid.shape, 3.5, dtype=np.float64)
    q_field = np.full(grid.shape, 5000.0, dtype=np.float64)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 320.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 290.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 150.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, -150.0),
    )

    solver = HeatConductionSolver2D()
    result = solver.solve(grid, k_field, q_field, bcs)
    diag = result.diagnostics

    # Verify global conservation with rectangular cells and non-unit thickness
    eb = diag.energy_balance
    dx, dy, thick = grid.dx, grid.dy, grid.thickness
    area_x = dy * thick
    area_y = dx * thick

    return {
        "case": "Rectangular Grid & Non-Unit Thickness",
        "dx": float(dx),
        "dy": float(dy),
        "aspect_ratio_dx_dy": float(dx / dy),
        "thickness": float(thick),
        "area_x": float(area_x),
        "area_y": float(area_y),
        "q_gen_total_w": float(eb.q_generated_total),
        "q_bnd_out_total_w": float(eb.q_boundary_out_total),
        "flux_imbalance_w": float(eb.flux_imbalance),
        "relative_imbalance": float(eb.relative_imbalance),
        "solver_converged": bool(diag.is_converged),
    }


def run_full_audit() -> dict:
    print("Running Milestone 08B.1 Phase 1 Discrete Boundary Consistency Audit...")
    c1 = audit_linear_conduction_1d()
    print("  Case 1 (1D Linear, q=0):", c1)
    c2 = audit_quadratic_conduction_1d()
    print("  Case 2 (1D Quadratic, q!=0):", c2)
    c3 = audit_variable_k_2d_mms()
    print("  Case 3 (2D MMS Variable K):", c3)
    c4 = audit_rectangular_cells_and_thickness()
    print("  Case 4 (Rectangular & Thickness):", c4)

    summary = {
        "case_1_linear_1d": c1,
        "case_2_quadratic_1d": c2,
        "case_3_variable_k_2d": c3,
        "case_4_rectangular_thickness": c4,
    }

    out_path = Path("artifacts/reports/milestone_08b1_boundary_consistency_math.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")
    return summary


if __name__ == "__main__":
    run_full_audit()
