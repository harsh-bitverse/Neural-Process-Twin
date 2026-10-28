"""Physics-constrained local smoothing and residual relaxation for Milestone 08F.

Provides fast, matrix-free Jacobi relaxation operators to damp high-frequency
localized residual spikes and smooth boundary layer errors without the full cost
of sparse matrix LU factorization.
"""

from __future__ import annotations

import numpy as np

from neural_process_twin.solvers.assembly import LinearSystem, assemble_steady_heat_system
from neural_process_twin.solvers.boundary import BoundaryConditions
from neural_process_twin.solvers.grid import Grid2D


class JacobiPhysicsSmoother:
    """Damped Jacobi relaxation operator for localized PDE residual smoothing.

    Performs:
        T^{(m+1)} = T^{(m)} - omega * D^{-1} (A * T^{(m)} - b)
    where D = diag(A) contains diagonal conductances and omega in (0, 1] is the damping factor.
    """

    def __init__(self, omega: float = 0.67, max_iters: int = 3) -> None:
        """Initialize JacobiPhysicsSmoother.

        Args:
            omega: Damping factor (default: 2/3 for optimal high-frequency error damping).
            max_iters: Number of Jacobi smoothing passes (default: 3).
        """
        if not (0.0 < omega <= 1.0):
            raise ValueError(f"omega must be in (0, 1], got {omega}")
        if max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {max_iters}")
        self.omega = float(omega)
        self.max_iters = int(max_iters)

    def smooth(
        self,
        t_init: np.ndarray,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
        num_iters: int | None = None,
        system: LinearSystem | None = None,
    ) -> np.ndarray:
        """Apply Jacobi smoothing passes to an initial temperature field.

        Args:
            t_init: Initial predicted temperature field [K] of shape (ny, nx).
            grid: 2D computational domain grid.
            k_field: Thermal conductivity field [W/(m K)].
            q_field: Heat source field [W/m^3].
            bcs: Prescribed boundary conditions.
            num_iters: Optional override for number of iterations.
            system: Optional pre-assembled LinearSystem to avoid re-assembly.

        Returns:
            Smoothed temperature field [K] of shape (ny, nx).
        """
        iters = num_iters if num_iters is not None else self.max_iters
        if iters <= 0:
            return t_init.copy()

        if system is None:
            system = assemble_steady_heat_system(grid, k_field, q_field, bcs)

        a_mat = system.A
        b_vec = system.b

        # Extract diagonal elements D of sparse matrix A
        diag_a = a_mat.diagonal()
        inv_diag = 1.0 / np.maximum(diag_a, 1e-12)

        t_vec = t_init.reshape(-1).astype(np.float64)

        for _ in range(iters):
            # Compute residual r = A * T - b
            residual = a_mat.dot(t_vec) - b_vec
            # Damped Jacobi update: T_new = T - omega * (r / D)
            t_vec -= self.omega * (inv_diag * residual)

        return t_vec.reshape((grid.ny, grid.nx)).astype(t_init.dtype)
