"""Two-dimensional cell-centred Cartesian grid for finite-volume PDE discretizations.

This module defines the spatial geometry, cell coordinates, face areas, control
volume metrics, and field evaluation helpers for 2D steady-state heat conduction.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from neural_process_twin.utils.config import DomainConfig, GridConfig


@dataclass(frozen=True)
class Grid2D:
    """Cell-centred uniform Cartesian grid in two spatial dimensions.

    Attributes:
        x_min: Minimum x-coordinate of rectangular domain [m].
        x_max: Maximum x-coordinate of rectangular domain [m].
        y_min: Minimum y-coordinate of rectangular domain [m].
        y_max: Maximum y-coordinate of rectangular domain [m].
        nx: Number of control volumes (cells) along x-axis. Must be >= 2.
        ny: Number of control volumes (cells) along y-axis. Must be >= 2.
        thickness: Out-of-plane model thickness [m]. Defaults to 1.0 m (unit depth).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    thickness: float = 1.0

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ValueError(
                f"x_min ({self.x_min}) must be strictly less than x_max ({self.x_max})."
            )
        if self.y_min >= self.y_max:
            raise ValueError(
                f"y_min ({self.y_min}) must be strictly less than y_max ({self.y_max})."
            )
        if self.nx < 2:
            raise ValueError(f"nx ({self.nx}) must be >= 2.")
        if self.ny < 2:
            raise ValueError(f"ny ({self.ny}) must be >= 2.")
        if self.thickness <= 0.0:
            raise ValueError(f"thickness ({self.thickness}) must be strictly positive.")

    @property
    def lx(self) -> float:
        """Domain length in x-direction [m]."""
        return self.x_max - self.x_min

    @property
    def ly(self) -> float:
        """Domain length in y-direction [m]."""
        return self.y_max - self.y_min

    @property
    def dx(self) -> float:
        """Uniform cell width in x-direction [m]."""
        return self.lx / self.nx

    @property
    def dy(self) -> float:
        """Uniform cell height in y-direction [m]."""
        return self.ly / self.ny

    @property
    def cell_volume(self) -> float:
        """Control volume of a single cell [m^3]."""
        return self.dx * self.dy * self.thickness

    @property
    def face_area_x(self) -> float:
        """Area of internal or boundary face normal to x-axis [m^2]."""
        return self.dy * self.thickness

    @property
    def face_area_y(self) -> float:
        """Area of internal or boundary face normal to y-axis [m^2]."""
        return self.dx * self.thickness

    @property
    def num_cells(self) -> int:
        """Total number of control volumes N = nx * ny."""
        return self.nx * self.ny

    @property
    def shape(self) -> tuple[int, int]:
        """2D array shape convention (ny, nx)."""
        return (self.ny, self.nx)

    @property
    def x_centers(self) -> np.ndarray:
        """1D array of cell-centre x-coordinates of shape (nx,) [m]."""
        return self.x_min + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.dx

    @property
    def y_centers(self) -> np.ndarray:
        """1D array of cell-centre y-coordinates of shape (ny,) [m]."""
        return self.y_min + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.dy

    @property
    def x_mesh(self) -> np.ndarray:
        """2D array of cell-centre x-coordinates of shape (ny, nx) [m]."""
        x_mesh, _ = np.meshgrid(self.x_centers, self.y_centers, indexing="xy")
        return x_mesh

    @property
    def y_mesh(self) -> np.ndarray:
        """2D array of cell-centre y-coordinates of shape (ny, nx) [m]."""
        _, y_mesh = np.meshgrid(self.x_centers, self.y_centers, indexing="xy")
        return y_mesh

    def cell_index(self, i: int, j: int) -> int:
        """Convert 2D cell indices (i, j) to flattened linear index p in [0, N-1].

        Row-major (C-order) indexing: p = j * nx + i.
        """
        if not (0 <= i < self.nx):
            raise IndexError(f"Column index i={i} out of bounds for nx={self.nx}.")
        if not (0 <= j < self.ny):
            raise IndexError(f"Row index j={j} out of bounds for ny={self.ny}.")
        return j * self.nx + i

    def cell_coords(self, p: int) -> tuple[int, int]:
        """Convert flattened linear index p in [0, N-1] to 2D cell indices (i, j).

        Returns:
            Tuple of (i, j) where i is column index along x and j is row index along y.
        """
        if not (0 <= p < self.num_cells):
            raise IndexError(f"Linear index p={p} out of bounds for num_cells={self.num_cells}.")
        j = p // self.nx
        i = p % self.nx
        return i, j

    def evaluate_field(
        self,
        field_input: float | int | np.ndarray | Callable[[np.ndarray, np.ndarray], Any],
        field_name: str = "field",
    ) -> np.ndarray:
        """Evaluate or validate a scalar field on the cell-centred grid.

        Args:
            field_input: Scalar float, 2D NumPy array of shape (ny, nx), or
                callable function f(x, y) accepting 2D coordinate meshgrids.
            field_name: Descriptive name for error messages.

        Returns:
            2D NumPy array of shape (ny, nx) and dtype float64.

        Raises:
            TypeError: If input is neither a scalar, array, nor callable.
            ValueError: If array shape does not match (ny, nx) or contains NaN/Inf.
        """
        if isinstance(field_input, (int, float, np.floating, np.integer)):
            val = float(field_input)
            if not np.isfinite(val):
                raise ValueError(f"{field_name} constant value must be finite, got {val}.")
            return np.full((self.ny, self.nx), val, dtype=np.float64)

        if callable(field_input):
            arr = np.asarray(field_input(self.x_mesh, self.y_mesh), dtype=np.float64)
        elif isinstance(field_input, np.ndarray):
            arr = np.asarray(field_input, dtype=np.float64)
        else:
            raise TypeError(
                f"{field_name} must be a float, 2D ndarray of shape (ny, nx), or callable f(x, y); "
                f"got {type(field_input).__name__}."
            )

        if arr.shape != (self.ny, self.nx):
            raise ValueError(
                f"{field_name} array shape {arr.shape} does not match expected grid shape "
                f"(ny={self.ny}, nx={self.nx})."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{field_name} contains non-finite values (NaN or Inf).")

        return arr

    @classmethod
    def from_configs(
        cls,
        domain_cfg: DomainConfig,
        grid_cfg: GridConfig,
        thickness: float = 1.0,
    ) -> Grid2D:
        """Construct Grid2D from validated DomainConfig and GridConfig dataclasses."""
        return cls(
            x_min=domain_cfg.x_min,
            x_max=domain_cfg.x_max,
            y_min=domain_cfg.y_min,
            y_max=domain_cfg.y_max,
            nx=grid_cfg.nx,
            ny=grid_cfg.ny,
            thickness=thickness,
        )
