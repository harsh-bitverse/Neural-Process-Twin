"""Top-level reference solver API for 2D steady-state heat conduction.

Provides structured results including temperature fields, grid metadata,
sparse system residuals, and physical energy balance conservation diagnostics.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla

from neural_process_twin.solvers.assembly import (
    LinearSystem,
    assemble_steady_heat_system,
)
from neural_process_twin.solvers.boundary import BoundaryConditions
from neural_process_twin.solvers.diagnostics import (
    SolverDiagnostics,
    compute_heat_balance,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.utils.config import BoundaryConditionsConfig


@dataclass(frozen=True)
class SolverResult:
    """Structured result returned by the numerical reference heat solver.

    Attributes:
        temperature: 2D array of solution temperatures of shape (ny, nx) [K].
        grid: Underlying 2D computational grid.
        conductivity: 2D array of evaluated conductivities of shape (ny, nx) [W/(m K)].
        source: 2D array of evaluated volumetric heat source of shape (ny, nx) [W/m^3].
        boundary_conditions: Applied boundary conditions.
        linear_system: Assembled sparse linear system A * T = b.
        diagnostics: Algebraic residual and physical energy balance metrics.
        solve_time_seconds: Elapsed wall-clock time for matrix assembly and solution [s].
    """

    temperature: np.ndarray
    grid: Grid2D
    conductivity: np.ndarray
    source: np.ndarray
    boundary_conditions: BoundaryConditions
    linear_system: LinearSystem
    diagnostics: SolverDiagnostics
    solve_time_seconds: float


class HeatConductionSolver2D:
    """Finite-volume reference numerical solver for 2D steady-state heat conduction."""

    def __init__(self, tolerance: float = 1e-10) -> None:
        """Initialize solver.

        Args:
            tolerance: Convergence tolerance for residual checks.
        """
        self.tolerance = tolerance

    def solve(
        self,
        grid: Grid2D,
        conductivity: float | int | np.ndarray | Callable[[np.ndarray, np.ndarray], Any],
        source: float | int | np.ndarray | Callable[[np.ndarray, np.ndarray], Any],
        boundary_conditions: BoundaryConditions | BoundaryConditionsConfig,
        gauge_pin: tuple[int, int] | None = None,
        gauge_temperature: float | None = None,
    ) -> SolverResult:
        """Solve steady-state heat conduction -div(k grad(T)) = q on the specified grid.

        Args:
            grid: 2D computational grid defining spatial bounds and cell resolution.
            conductivity: Spatial conductivity k(x, y) as float, 2D array, or callable [W/(m K)].
            source: Volumetric heat source q(x, y) as float, 2D array, or callable [W/m^3].
            boundary_conditions: Boundary conditions across all four domain boundaries.
            gauge_pin: Optional (i, j) cell coordinates to pin for pure-Neumann problems.
            gauge_temperature: Optional temperature [K] to assign at gauge_pin.

        Returns:
            Structured SolverResult containing temperature field and diagnostics.

        Raises:
            PureNeumannSingularError: If all boundaries are Neumann and no gauge pin is provided.
            ValueError: If input dimensions, domain parameters, or physical values are invalid.
        """
        start_time = time.perf_counter()

        # 1. Normalize boundary conditions
        if isinstance(boundary_conditions, BoundaryConditionsConfig):
            bcs = BoundaryConditions.from_config(boundary_conditions)
        elif isinstance(boundary_conditions, BoundaryConditions):
            bcs = boundary_conditions
        else:
            raise TypeError(
                f"boundary_conditions must be BoundaryConditions or BoundaryConditionsConfig, "
                f"got {type(boundary_conditions).__name__}."
            )

        # 2. Evaluate fields on cell-centred grid
        k_field = grid.evaluate_field(conductivity, field_name="conductivity")
        q_field = grid.evaluate_field(source, field_name="source")

        if np.any(k_field <= 0.0):
            raise ValueError("Thermal conductivity must be strictly positive (k > 0) everywhere.")

        # 3. Assemble sparse linear system
        system = assemble_steady_heat_system(
            grid=grid,
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            gauge_pin=gauge_pin,
            gauge_temperature=gauge_temperature,
        )

        # 4. Solve sparse linear system A * T_flat = b
        # Using SuperLU direct solver from SciPy sparse
        t_flat = spla.spsolve(system.A, system.b)
        t_field = t_flat.reshape((grid.ny, grid.nx))

        # 5. Compute algebraic residual ||A * T - b||_2
        res_vec = system.A.dot(t_flat) - system.b
        res_l2 = float(np.linalg.norm(res_vec))
        b_norm = float(np.linalg.norm(system.b))
        rel_res = res_l2 / b_norm if b_norm > 0.0 else res_l2

        # 6. Compute independent physical energy balance diagnostics
        energy_diag = compute_heat_balance(
            grid=grid,
            temperature=t_field,
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
        )

        diagnostics = SolverDiagnostics(
            linear_residual_l2=res_l2,
            relative_residual=rel_res,
            is_converged=bool(rel_res < self.tolerance or res_l2 < self.tolerance),
            energy_balance=energy_diag,
        )

        elapsed = time.perf_counter() - start_time

        return SolverResult(
            temperature=t_field,
            grid=grid,
            conductivity=k_field,
            source=q_field,
            boundary_conditions=bcs,
            linear_system=system,
            diagnostics=diagnostics,
            solve_time_seconds=elapsed,
        )


def solve_steady_heat_2d(
    grid: Grid2D,
    conductivity: float | int | np.ndarray | Callable[[np.ndarray, np.ndarray], Any],
    source: float | int | np.ndarray | Callable[[np.ndarray, np.ndarray], Any],
    boundary_conditions: BoundaryConditions | BoundaryConditionsConfig,
    gauge_pin: tuple[int, int] | None = None,
    gauge_temperature: float | None = None,
    tolerance: float = 1e-10,
) -> SolverResult:
    """Convenience function to solve 2D steady-state heat conduction."""
    solver = HeatConductionSolver2D(tolerance=tolerance)
    return solver.solve(
        grid=grid,
        conductivity=conductivity,
        source=source,
        boundary_conditions=boundary_conditions,
        gauge_pin=gauge_pin,
        gauge_temperature=gauge_temperature,
    )
