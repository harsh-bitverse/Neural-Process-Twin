"""Physical energy balance and numerical residual diagnostics for 2D heat conduction.

Provides independent physical conservation checks comparing total integrated
volumetric heat generation against reconstructed boundary heat flows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_process_twin.solvers.boundary import BoundaryConditions, BoundaryType
from neural_process_twin.solvers.grid import Grid2D


@dataclass(frozen=True)
class EnergyBalanceDiagnostics:
    """Quantitative diagnostics for global physical energy conservation.

    Attributes:
        q_generated_total: Total volumetric heat generation rate in domain [W].
        q_boundary_out_total: Total conductive heat flow rate leaving domain across boundaries [W].
        flux_imbalance: Absolute conservation error |q_boundary_out_total - q_generated_total| [W].
        relative_imbalance: Dimensionless relative error normalized by the characteristic heat flow.
        boundary_heat_flow_out: Net outward heat flow rate for each boundary [W].
        boundary_flux_densities: 1D array of outward heat flux densities j_out for each boundary [W/m^2].
    """

    q_generated_total: float
    q_boundary_out_total: float
    flux_imbalance: float
    relative_imbalance: float
    boundary_heat_flow_out: dict[str, float]
    boundary_flux_densities: dict[str, np.ndarray]


@dataclass(frozen=True)
class SolverDiagnostics:
    """Numerical convergence, sparse matrix residual, and physical conservation metrics.

    Attributes:
        linear_residual_l2: Euclidean L2 norm of the algebraic residual ||A * T - b||_2 [W].
        relative_residual: Normalized algebraic residual ||A * T - b||_2 / ||b||_2.
        is_converged: Boolean flag indicating successful linear solver termination.
        energy_balance: EnergyBalanceDiagnostics containing physical conservation checks.
    """

    linear_residual_l2: float
    relative_residual: float
    is_converged: bool
    energy_balance: EnergyBalanceDiagnostics


def compute_heat_balance(
    grid: Grid2D,
    temperature: np.ndarray,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
) -> EnergyBalanceDiagnostics:
    """Compute independent physical energy balance diagnostics from the solution field.

    Reconstructs the boundary heat flux densities from the cell-centre temperatures
    for Dirichlet faces and evaluates the imposed fluxes for Neumann faces.

    Args:
        grid: Computational grid.
        temperature: Computed 2D temperature field of shape (ny, nx) [K].
        k_field: 2D conductivity field of shape (ny, nx) [W/(m K)].
        q_field: 2D source term field of shape (ny, nx) [W/m^3].
        bcs: Prescribed boundary conditions.

    Returns:
        EnergyBalanceDiagnostics reporting total heat flows and conservation metrics.
    """
    dx, dy = grid.dx, grid.dy
    area_x, area_y = grid.face_area_x, grid.face_area_y
    bc_values = bcs.evaluate_all(grid)

    q_flows: dict[str, float] = {}
    flux_densities: dict[str, np.ndarray] = {}

    # 1. West boundary (x = x_min, normal = (-1, 0))
    if bcs.left.bc_type == BoundaryType.DIRICHLET:
        t_bnd = bc_values["left"]
        # Outward flux: j_out = -k * (dT/dn) = k * (dT/dx) = 2 * k * (T_P - T_bnd) / dx
        j_out = (2.0 * k_field[:, 0] / dx) * (temperature[:, 0] - t_bnd)
    else:
        # Imposed outward flux density
        j_out = bc_values["left"].copy()
    q_flows["left"] = float(np.sum(j_out * area_x))
    flux_densities["left"] = j_out

    # 2. East boundary (x = x_max, normal = (+1, 0))
    if bcs.right.bc_type == BoundaryType.DIRICHLET:
        t_bnd = bc_values["right"]
        j_out = (2.0 * k_field[:, -1] / dx) * (temperature[:, -1] - t_bnd)
    else:
        j_out = bc_values["right"].copy()
    q_flows["right"] = float(np.sum(j_out * area_x))
    flux_densities["right"] = j_out

    # 3. South boundary (y = y_min, normal = (0, -1))
    if bcs.bottom.bc_type == BoundaryType.DIRICHLET:
        t_bnd = bc_values["bottom"]
        j_out = (2.0 * k_field[0, :] / dy) * (temperature[0, :] - t_bnd)
    else:
        j_out = bc_values["bottom"].copy()
    q_flows["bottom"] = float(np.sum(j_out * area_y))
    flux_densities["bottom"] = j_out

    # 4. North boundary (y = y_max, normal = (0, +1))
    if bcs.top.bc_type == BoundaryType.DIRICHLET:
        t_bnd = bc_values["top"]
        j_out = (2.0 * k_field[-1, :] / dy) * (temperature[-1, :] - t_bnd)
    else:
        j_out = bc_values["top"].copy()
    q_flows["top"] = float(np.sum(j_out * area_y))
    flux_densities["top"] = j_out

    # Total heat flows
    q_boundary_out = float(sum(q_flows.values()))
    q_generated = float(np.sum(q_field * grid.cell_volume))
    imbalance = abs(q_boundary_out - q_generated)

    # Reference heat scale for dimensionless relative imbalance
    scale = max(abs(q_generated), abs(q_boundary_out), 1.0e-12)
    relative_imbalance = imbalance / scale

    return EnergyBalanceDiagnostics(
        q_generated_total=q_generated,
        q_boundary_out_total=q_boundary_out,
        flux_imbalance=imbalance,
        relative_imbalance=relative_imbalance,
        boundary_heat_flow_out=q_flows,
        boundary_flux_densities=flux_densities,
    )
