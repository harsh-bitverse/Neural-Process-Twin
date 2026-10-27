"""Independent discrete physical residual diagnostics for surrogate predictions.

Evaluates the discrete PDE balance -div_h(k grad_h(T_pred)) - q and boundary
conservation error using the conservative finite-volume stencil of the reference solver.
"""

from __future__ import annotations

import numpy as np

from neural_process_twin.solvers.assembly import assemble_steady_heat_system
from neural_process_twin.solvers.boundary import BoundaryConditions
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D


def compute_discrete_pde_residual(
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
    temperature_pred: np.ndarray,
) -> dict[str, float]:
    """Compute discrete cell-centred PDE residuals for a surrogate temperature prediction.

    Evaluates r = A @ T_pred.ravel() - b using the reference finite-volume discretization,
    where A and b represent the exact cell-integrated conservation stencils in Watts [W].

    Args:
        grid: Computational grid.
        k_field: 2D conductivity field (ny, nx) [W/(m K)].
        q_field: 2D source term field (ny, nx) [W/m^3].
        bcs: BoundaryConditions instance.
        temperature_pred: Predicted 2D temperature field (ny, nx) [K].

    Returns:
        Dictionary with residual norms in Watts [W] and relative residual.
    """
    sys = assemble_steady_heat_system(
        grid=grid,
        k_field=k_field,
        q_field=q_field,
        bcs=bcs,
    )

    t_flat = np.asarray(temperature_pred, dtype=np.float64).ravel()
    r = sys.A @ t_flat - sys.b

    r_l2 = float(np.linalg.norm(r))
    b_l2 = float(np.linalg.norm(sys.b))
    r_rel = float(r_l2 / max(b_l2, 1e-12))
    r_linf = float(np.max(np.abs(r)))

    return {
        "pde_residual_l2_w": r_l2,
        "pde_residual_relative": r_rel,
        "pde_residual_linf_w": r_linf,
    }


def compute_boundary_flux_diagnostics(
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
    temperature_pred: np.ndarray,
) -> dict[str, float]:
    """Compute global boundary heat flux balance and conservation error for a surrogate.

    Args:
        grid: Computational grid.
        k_field: 2D conductivity field (ny, nx) [W/(m K)].
        q_field: 2D source term field (ny, nx) [W/m^3].
        bcs: BoundaryConditions instance.
        temperature_pred: Predicted 2D temperature field (ny, nx) [K].

    Returns:
        Dictionary reporting total heat flows and conservation imbalance in Watts [W].
    """
    diag = compute_heat_balance(
        grid=grid,
        temperature=np.asarray(temperature_pred, dtype=np.float64),
        k_field=k_field,
        q_field=q_field,
        bcs=bcs,
    )

    return {
        "q_generated_total_w": diag.q_generated_total,
        "q_boundary_out_total_w": diag.q_boundary_out_total,
        "flux_imbalance_w": diag.flux_imbalance,
        "flux_imbalance_relative": diag.relative_imbalance,
    }


def evaluate_physical_residuals(
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
    temperature_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate combined discrete PDE residual and boundary flux conservation metrics.

    Args:
        grid: Computational grid.
        k_field: 2D conductivity field (ny, nx) [W/(m K)].
        q_field: 2D source term field (ny, nx) [W/m^3].
        bcs: BoundaryConditions instance.
        temperature_pred: Predicted 2D temperature field (ny, nx) [K].

    Returns:
        Dictionary combining discrete PDE residuals and boundary flux diagnostics.
    """
    pde_res = compute_discrete_pde_residual(grid, k_field, q_field, bcs, temperature_pred)
    flux_res = compute_boundary_flux_diagnostics(grid, k_field, q_field, bcs, temperature_pred)
    return {**pde_res, **flux_res}
