"""Sparse linear system assembly for 2D cell-centred finite-volume heat conduction.

Discretizes the steady-state heat equation:
    -div(k grad(T)) = q
over a 2D uniform Cartesian grid with conservative flux balance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from neural_process_twin.solvers.boundary import BoundaryConditions, BoundaryType
from neural_process_twin.solvers.conductivity import compute_internal_face_conductivities
from neural_process_twin.solvers.grid import Grid2D


class PureNeumannSingularError(ValueError):
    """Raised when a steady-state problem has purely Neumann boundaries without a gauge reference."""


@dataclass(frozen=True)
class LinearSystem:
    """Assembled sparse linear system A * T = b and associated face conductances.

    Attributes:
        A: Sparse coefficient matrix in CSR format of size (N, N), where N = nx * ny.
        b: Right-hand side load vector of shape (N,) [W].
        grid: Underlying 2D computational grid.
        conductance_east: Conductance across vertical internal faces, shape (ny, nx - 1) [W/K].
        conductance_north: Conductance across horizontal internal faces, shape (ny - 1, nx) [W/K].
        boundary_conductances: Dictionary of Dirichlet boundary face conductances [W/K].
    """

    A: sp.csr_matrix
    b: np.ndarray
    grid: Grid2D
    conductance_east: np.ndarray
    conductance_north: np.ndarray
    boundary_conductances: dict[str, np.ndarray]


def assemble_steady_heat_system(
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
    gauge_pin: tuple[int, int] | None = None,
    gauge_temperature: float | None = None,
) -> LinearSystem:
    """Assemble the conservative finite-volume linear system A * T = b.

    Args:
        grid: Grid2D instance defining geometry and resolution.
        k_field: 2D array of cell conductivities, shape (ny, nx) [W/(m K)].
        q_field: 2D array of volumetric heat source, shape (ny, nx) [W/m^3].
        bcs: Boundary conditions across left, right, bottom, and top.
        gauge_pin: Optional (i, j) cell coordinates to pin for pure-Neumann problems.
        gauge_temperature: Optional temperature [K] to assign at gauge_pin.

    Returns:
        LinearSystem containing sparse CSR matrix A, RHS vector b, and conductances.

    Raises:
        PureNeumannSingularError: If all boundaries are Neumann and no gauge pin is provided.
        ValueError: If conductivity is non-positive or field dimensions mismatch.
    """
    if k_field.shape != grid.shape:
        raise ValueError(f"k_field shape {k_field.shape} does not match grid shape {grid.shape}.")
    if q_field.shape != grid.shape:
        raise ValueError(f"q_field shape {q_field.shape} does not match grid shape {grid.shape}.")
    if np.any(k_field <= 0.0):
        raise ValueError("Conductivity k_field must be strictly positive everywhere.")

    # Pure Neumann singularity check
    if bcs.is_pure_neumann and (gauge_pin is None or gauge_temperature is None):
        raise PureNeumannSingularError(
            "All four boundaries are Neumann. Steady-state heat conduction is singular without "
            "a Dirichlet boundary or reference gauge constraint. Provide gauge_pin=(i, j) and "
            "gauge_temperature to anchor the solution."
        )

    nx, ny = grid.nx, grid.ny
    n_total = grid.num_cells
    dx, dy = grid.dx, grid.dy
    area_x, area_y = grid.face_area_x, grid.face_area_y

    # Internal face conductivities and conductances
    k_east, k_north = compute_internal_face_conductivities(k_field)
    g_east = (k_east * area_x) / dx  # shape (ny, nx - 1)
    g_north = (k_north * area_y) / dy  # shape (ny - 1, nx)

    # Initialize COO triplet buffers and right-hand side vector
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    diag = np.zeros(n_total, dtype=np.float64)
    b = np.zeros(n_total, dtype=np.float64)

    # 1. Internal East-West faces (normal to x)
    for j in range(ny):
        for i in range(nx - 1):
            p = grid.cell_index(i, j)
            e = grid.cell_index(i + 1, j)
            g = float(g_east[j, i])

            diag[p] += g
            diag[e] += g
            rows.extend([p, e])
            cols.extend([e, p])
            vals.extend([-g, -g])

    # 2. Internal North-South faces (normal to y)
    for j in range(ny - 1):
        for i in range(nx):
            p = grid.cell_index(i, j)
            n = grid.cell_index(i, j + 1)
            g = float(g_north[j, i])

            diag[p] += g
            diag[n] += g
            rows.extend([p, n])
            cols.extend([n, p])
            vals.extend([-g, -g])

    # 3. Boundary conditions
    bc_values = bcs.evaluate_all(grid)
    boundary_conductances: dict[str, np.ndarray] = {}

    # West boundary (x = x_min, i = 0)
    left_vals = bc_values["left"]
    if bcs.left.bc_type == BoundaryType.DIRICHLET:
        # Distance from face to cell centre is dx / 2: G_b = k * area_x / (dx / 2) = 2 * k * area_x / dx
        g_left = (2.0 * k_field[:, 0] * area_x) / dx
        boundary_conductances["left"] = g_left
        for j in range(ny):
            p = grid.cell_index(0, j)
            g_b = float(g_left[j])
            diag[p] += g_b
            b[p] += g_b * float(left_vals[j])
    else:
        # Neumann: left_vals represents outward heat flux g_out [W/m^2]
        # Heat leaving domain across west face is Q_out = g_out * area_x
        # Moving to RHS: b[p] -= Q_out
        for j in range(ny):
            p = grid.cell_index(0, j)
            b[p] -= float(left_vals[j]) * area_x

    # East boundary (x = x_max, i = nx - 1)
    right_vals = bc_values["right"]
    if bcs.right.bc_type == BoundaryType.DIRICHLET:
        g_right = (2.0 * k_field[:, nx - 1] * area_x) / dx
        boundary_conductances["right"] = g_right
        for j in range(ny):
            p = grid.cell_index(nx - 1, j)
            g_b = float(g_right[j])
            diag[p] += g_b
            b[p] += g_b * float(right_vals[j])
    else:
        for j in range(ny):
            p = grid.cell_index(nx - 1, j)
            b[p] -= float(right_vals[j]) * area_x

    # South boundary (y = y_min, j = 0)
    bottom_vals = bc_values["bottom"]
    if bcs.bottom.bc_type == BoundaryType.DIRICHLET:
        g_bottom = (2.0 * k_field[0, :] * area_y) / dy
        boundary_conductances["bottom"] = g_bottom
        for i in range(nx):
            p = grid.cell_index(i, 0)
            g_b = float(g_bottom[i])
            diag[p] += g_b
            b[p] += g_b * float(bottom_vals[i])
    else:
        for i in range(nx):
            p = grid.cell_index(i, 0)
            b[p] -= float(bottom_vals[i]) * area_y

    # North boundary (y = y_max, j = ny - 1)
    top_vals = bc_values["top"]
    if bcs.top.bc_type == BoundaryType.DIRICHLET:
        g_top = (2.0 * k_field[ny - 1, :] * area_y) / dy
        boundary_conductances["top"] = g_top
        for i in range(nx):
            p = grid.cell_index(i, ny - 1)
            g_b = float(g_top[i])
            diag[p] += g_b
            b[p] += g_b * float(top_vals[i])
    else:
        for i in range(nx):
            p = grid.cell_index(i, ny - 1)
            b[p] -= float(top_vals[i]) * area_y

    # 4. Volumetric heat generation source term: S_P = q_P * cell_volume
    q_vol = q_field * grid.cell_volume
    b += q_vol.ravel()

    # 5. Add diagonal entries to COO triplet list
    for p in range(n_total):
        rows.append(p)
        cols.append(p)
        vals.append(diag[p])

    # 6. Apply gauge pin if requested (for pure Neumann problems)
    if gauge_pin is not None and gauge_temperature is not None:
        pin_i, pin_j = gauge_pin
        pin_p = grid.cell_index(pin_i, pin_j)
        # Filter out all non-diagonal entries for row pin_p
        filtered_rows = []
        filtered_cols = []
        filtered_vals = []
        for r_idx, c_idx, v_val in zip(rows, cols, vals, strict=True):
            if r_idx == pin_p:
                if c_idx == pin_p:
                    filtered_rows.append(r_idx)
                    filtered_cols.append(c_idx)
                    filtered_vals.append(1.0)
            else:
                filtered_rows.append(r_idx)
                filtered_cols.append(c_idx)
                filtered_vals.append(v_val)
        rows, cols, vals = filtered_rows, filtered_cols, filtered_vals
        b[pin_p] = float(gauge_temperature)

    a_coo = sp.coo_matrix((vals, (rows, cols)), shape=(n_total, n_total), dtype=np.float64)
    a_csr = a_coo.tocsr()

    return LinearSystem(
        A=a_csr,
        b=b,
        grid=grid,
        conductance_east=g_east,
        conductance_north=g_north,
        boundary_conductances=boundary_conductances,
    )
