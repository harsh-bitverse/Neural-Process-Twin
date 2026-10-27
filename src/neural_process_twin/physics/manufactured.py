"""Analytical manufactured solutions for verification of numerical PDE solvers.

Provides closed-form paired solutions (T_exact, k_exact, q_exact) for steady-state
heat conduction:
    -div(k(x, y) grad(T(x, y))) = q(x, y)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)


@dataclass(frozen=True)
class SinusoidalMMSConstantK:
    """Manufactured 2D sinusoidal solution with constant thermal conductivity.

    Solution:
        T*(x, y) = T_0 + A * sin(pi * (x - x_min) / Lx) * sin(pi * (y - y_min) / Ly)

    Source:
        q*(x, y) = -k_0 * laplacian(T*)
                 = k_0 * pi^2 * (1 / Lx^2 + 1 / Ly^2) * A * sin(...) * sin(...)

    Boundary Conditions:
        T*(x, y) = T_0 along all four boundaries (homogeneous Dirichlet relative to T_0).
    """

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    k_0: float = 1.0
    t_0: float = 300.0
    amplitude: float = 25.0

    @property
    def lx(self) -> float:
        return self.x_max - self.x_min

    @property
    def ly(self) -> float:
        return self.y_max - self.y_min

    def temperature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact temperature field T*(x, y) [K]."""
        sx = np.sin(np.pi * (x - self.x_min) / self.lx)
        sy = np.sin(np.pi * (y - self.y_min) / self.ly)
        return self.t_0 + self.amplitude * sx * sy

    def conductivity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact thermal conductivity k*(x, y) [W/(m K)]."""
        return np.full_like(x, self.k_0, dtype=np.float64)

    def source(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact volumetric heat generation rate q*(x, y) [W/m^3]."""
        sx = np.sin(np.pi * (x - self.x_min) / self.lx)
        sy = np.sin(np.pi * (y - self.y_min) / self.ly)
        factor = self.k_0 * (np.pi**2) * (1.0 / (self.lx**2) + 1.0 / (self.ly**2)) * self.amplitude
        return factor * sx * sy

    def boundary_conditions(self) -> BoundaryConditions:
        """Return Dirichlet boundary conditions with value T_0 on all four sides."""
        return BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, self.t_0),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, self.t_0),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, self.t_0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, self.t_0),
        )


@dataclass(frozen=True)
class SinusoidalMMSVariableK:
    """Manufactured 2D sinusoidal solution with spatially varying conductivity.

    Temperature:
        T*(x, y) = T_0 + A * sin(pi * (x - x_min) / Lx) * sin(pi * (y - y_min) / Ly)

    Conductivity:
        k(x, y) = k_0 + k_1 * cos(pi * (x - x_min) / Lx) * cos(pi * (y - y_min) / Ly)

    Source:
        q*(x, y) = -div(k grad(T*))
                 = - (grad(k) . grad(T*) + k * laplacian(T*))
                 = k(x, y) * pi^2 * (1/Lx^2 + 1/Ly^2) * A * sin(...) * sin(...)
                   + A * k_1 * (pi^2 / 4) * (1/Lx^2 + 1/Ly^2) * sin(2*pi*(x-x_min)/Lx) * sin(2*pi*(y-y_min)/Ly)
    """

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    k_0: float = 2.0
    k_1: float = 0.5
    t_0: float = 300.0
    amplitude: float = 25.0

    def __post_init__(self) -> None:
        if self.k_0 <= self.k_1:
            raise ValueError(
                f"k_0 ({self.k_0}) must be > k_1 ({self.k_1}) to guarantee k(x,y) > 0."
            )

    @property
    def lx(self) -> float:
        return self.x_max - self.x_min

    @property
    def ly(self) -> float:
        return self.y_max - self.y_min

    def temperature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact temperature field T*(x, y) [K]."""
        sx = np.sin(np.pi * (x - self.x_min) / self.lx)
        sy = np.sin(np.pi * (y - self.y_min) / self.ly)
        return self.t_0 + self.amplitude * sx * sy

    def conductivity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact thermal conductivity field k*(x, y) [W/(m K)]."""
        cx = np.cos(np.pi * (x - self.x_min) / self.lx)
        cy = np.cos(np.pi * (y - self.y_min) / self.ly)
        return self.k_0 + self.k_1 * cx * cy

    def source(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact volumetric heat source q*(x, y) = -div(k grad(T*)) [W/m^3]."""
        xi = (x - self.x_min) / self.lx
        eta = (y - self.y_min) / self.ly
        inv_len_sq = 1.0 / (self.lx**2) + 1.0 / (self.ly**2)

        k_val = self.conductivity(x, y)
        term1 = (
            k_val
            * (np.pi**2)
            * inv_len_sq
            * self.amplitude
            * np.sin(np.pi * xi)
            * np.sin(np.pi * eta)
        )
        term2 = (
            self.amplitude
            * self.k_1
            * (np.pi**2 / 4.0)
            * inv_len_sq
            * np.sin(2.0 * np.pi * xi)
            * np.sin(2.0 * np.pi * eta)
        )
        return term1 + term2

    def boundary_conditions(self) -> BoundaryConditions:
        """Return Dirichlet boundary conditions with value T_0 on all four sides."""
        return BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, self.t_0),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, self.t_0),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, self.t_0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, self.t_0),
        )


@dataclass(frozen=True)
class LinearConduction1D:
    """One-dimensional steady linear heat conduction between two Dirichlet walls.

    Domain: [x_min, x_max] x [y_min, y_max]
    Temperature: T*(x, y) = T_left + ((T_right - T_left) / Lx) * (x - x_min)
    Conductivity: k_0 (constant)
    Source: q(x, y) = 0
    Boundaries: Dirichlet on left/right, Adiabatic Neumann (g=0) on bottom/top.
    """

    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    k_0: float = 1.0
    t_left: float = 300.0
    t_right: float = 400.0

    @property
    def lx(self) -> float:
        return self.x_max - self.x_min

    def temperature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.t_left + ((self.t_right - self.t_left) / self.lx) * (x - self.x_min)

    def conductivity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.full_like(x, self.k_0, dtype=np.float64)

    def source(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.zeros_like(x, dtype=np.float64)

    def boundary_conditions(self) -> BoundaryConditions:
        return BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, self.t_left),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, self.t_right),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
        )
