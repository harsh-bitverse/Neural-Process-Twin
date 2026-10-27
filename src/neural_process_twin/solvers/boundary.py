"""Boundary condition definitions, evaluation, and validation for 2D heat conduction.

Supported boundary conditions:
    - Dirichlet: Prescribed boundary temperature T_b [K].
    - Neumann: Prescribed outward conductive heat flux g_out = j . n = -k (dT/dn) [W/m^2].
      Positive g_out denotes heat leaving the domain; negative g_out denotes heat entering.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.utils.config import BoundaryConditionsConfig, BoundarySideConfig


class BoundarySide(StrEnum):
    """The four boundaries of the 2D rectangular domain."""

    LEFT = "left"  # x = x_min
    RIGHT = "right"  # x = x_max
    BOTTOM = "bottom"  # y = y_min
    TOP = "top"  # y = y_max


class BoundaryType(StrEnum):
    """Supported physical boundary condition types."""

    DIRICHLET = "dirichlet"  # Prescribed temperature T_b [K]
    NEUMANN = "neumann"  # Prescribed outward heat flux g_out = -k (dT/dn) [W/m^2]


@dataclass(frozen=True)
class BoundaryCondition:
    """Specification of a boundary condition along a single domain side.

    Attributes:
        side: The domain boundary side (left, right, bottom, top).
        bc_type: Boundary type (dirichlet or neumann).
        value: Prescribed value. Can be a constant float, 1D NumPy array matching
            the face count, or a callable function f(coord) of the face coordinate.
    """

    side: BoundarySide
    bc_type: BoundaryType
    value: float | int | np.ndarray | Callable[[np.ndarray], Any]

    def evaluate(self, coords: np.ndarray) -> np.ndarray:
        """Evaluate the boundary condition at the specified face coordinates.

        Args:
            coords: 1D array of face coordinates (y-coordinates for left/right,
                x-coordinates for bottom/top).

        Returns:
            1D array of boundary values matching coords.shape, dtype float64.
        """
        n_pts = len(coords)
        if isinstance(self.value, (int, float, np.floating, np.integer)):
            val = float(self.value)
            if not np.isfinite(val):
                raise ValueError(f"Boundary value for {self.side.value} must be finite, got {val}.")
            return np.full(n_pts, val, dtype=np.float64)

        if callable(self.value):
            arr = np.asarray(self.value(coords), dtype=np.float64)
        elif isinstance(self.value, np.ndarray):
            arr = np.asarray(self.value, dtype=np.float64)
        else:
            raise TypeError(
                f"Boundary value for {self.side.value} must be float, 1D array, or callable; "
                f"got {type(self.value).__name__}."
            )

        if arr.shape != (n_pts,):
            raise ValueError(
                f"Boundary array for {self.side.value} has shape {arr.shape}, expected ({n_pts},)."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Boundary array for {self.side.value} contains non-finite values.")

        return arr


@dataclass(frozen=True)
class BoundaryConditions:
    """Aggregated boundary conditions for all four sides of the 2D domain."""

    left: BoundaryCondition
    right: BoundaryCondition
    bottom: BoundaryCondition
    top: BoundaryCondition

    @property
    def is_pure_neumann(self) -> bool:
        """Return True if all four boundaries are Neumann conditions."""
        return all(
            bc.bc_type == BoundaryType.NEUMANN
            for bc in (self.left, self.right, self.bottom, self.top)
        )

    def evaluate_all(self, grid: Grid2D) -> dict[str, np.ndarray]:
        """Evaluate boundary values along each boundary at the cell-centre projections.

        Returns:
            Dictionary mapping side names ('left', 'right', 'bottom', 'top') to 1D arrays:
                - 'left': shape (ny,) evaluated at y_centers
                - 'right': shape (ny,) evaluated at y_centers
                - 'bottom': shape (nx,) evaluated at x_centers
                - 'top': shape (nx,) evaluated at x_centers
        """
        return {
            "left": self.left.evaluate(grid.y_centers),
            "right": self.right.evaluate(grid.y_centers),
            "bottom": self.bottom.evaluate(grid.x_centers),
            "top": self.top.evaluate(grid.x_centers),
        }

    @classmethod
    def from_config(cls, config: BoundaryConditionsConfig) -> BoundaryConditions:
        """Construct BoundaryConditions from validated configuration dataclass."""

        def make_bc(side: BoundarySide, side_cfg: BoundarySideConfig) -> BoundaryCondition:
            bc_type = BoundaryType(side_cfg.type.lower())
            return BoundaryCondition(side=side, bc_type=bc_type, value=side_cfg.value)

        return cls(
            left=make_bc(BoundarySide.LEFT, config.left),
            right=make_bc(BoundarySide.RIGHT, config.right),
            bottom=make_bc(BoundarySide.BOTTOM, config.bottom),
            top=make_bc(BoundarySide.TOP, config.top),
        )
