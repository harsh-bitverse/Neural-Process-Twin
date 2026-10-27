"""Numerical reference solvers for Neural Process Twin.

This package provides a conservative, cell-centred finite-volume reference solver
for 2D steady-state heat conduction:
    -div(k(x, y) grad(T(x, y))) = q(x, y)

Exposes spatial grid representations, harmonic conductivity averaging, sparse
matrix assembly, physical energy balance diagnostics, and convergence analysis tools.
"""

from neural_process_twin.solvers.assembly import (
    LinearSystem,
    PureNeumannSingularError,
    assemble_steady_heat_system,
)
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.conductivity import (
    arithmetic_average,
    compute_internal_face_conductivities,
    harmonic_average,
)
from neural_process_twin.solvers.convergence import (
    calculate_gci,
    compute_linf_error,
    compute_relative_l2_error,
    estimate_convergence_rates,
)
from neural_process_twin.solvers.diagnostics import (
    EnergyBalanceDiagnostics,
    SolverDiagnostics,
    compute_heat_balance,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import (
    HeatConductionSolver2D,
    SolverResult,
    solve_steady_heat_2d,
)

__all__ = [
    "BoundaryCondition",
    "BoundaryConditions",
    "BoundarySide",
    "BoundaryType",
    "EnergyBalanceDiagnostics",
    "Grid2D",
    "HeatConductionSolver2D",
    "LinearSystem",
    "PureNeumannSingularError",
    "SolverDiagnostics",
    "SolverResult",
    "arithmetic_average",
    "assemble_steady_heat_system",
    "calculate_gci",
    "compute_heat_balance",
    "compute_internal_face_conductivities",
    "compute_linf_error",
    "compute_relative_l2_error",
    "estimate_convergence_rates",
    "harmonic_average",
    "solve_steady_heat_2d",
]
