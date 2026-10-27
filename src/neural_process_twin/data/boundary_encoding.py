"""Deterministic spatial boundary condition encoding for 2D heat conduction.

Provides exact, reversible spatial channel encodings for all boundary condition
configurations supported by Neural Process Twin.

Resolves cell-centred face conventions and corner ambiguities by separating
normal-to-x boundaries (Left at i=0, Right at i=nx-1) and normal-to-y boundaries
(Bottom at j=0, Top at j=ny-1).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D

# Canonical 12-channel model input specification
CANONICAL_CHANNEL_NAMES: list[str] = [
    "conductivity",  # Ch 0: k(x, y) [W/(m K)]
    "heat_source",  # Ch 1: q(x, y) [W/m^3]
    "coord_x",  # Ch 2: x in [0, 1] [m]
    "coord_y",  # Ch 3: y in [0, 1] [m]
    "bc_dirichlet_mask_x",  # Ch 4: Dirichlet mask on x-normal faces (left/right)
    "bc_dirichlet_val_x",  # Ch 5: Prescribed Dirichlet temperature on x-normal faces [K]
    "bc_neumann_mask_x",  # Ch 6: Neumann mask on x-normal faces (left/right)
    "bc_neumann_val_x",  # Ch 7: Prescribed outward Neumann heat flux on x-normal faces [W/m^2]
    "bc_dirichlet_mask_y",  # Ch 8: Dirichlet mask on y-normal faces (bottom/top)
    "bc_dirichlet_val_y",  # Ch 9: Prescribed Dirichlet temperature on y-normal faces [K]
    "bc_neumann_mask_y",  # Ch 10: Neumann mask on y-normal faces (bottom/top)
    "bc_neumann_val_y",  # Ch 11: Prescribed outward Neumann heat flux on y-normal faces [W/m^2]
]

NUM_CANONICAL_CHANNELS = len(CANONICAL_CHANNEL_NAMES)  # 12


def encode_boundary_conditions(
    bc_source: BoundaryConditions | dict[str, Any],
    grid: Grid2D,
    normalizer: FieldNormalizer | None = None,
    flux_scale: float = 1000.0,
) -> np.ndarray:
    """Encode physical boundary conditions into an 8-channel 2D spatial representation.

    Channels:
        0: bc_dirichlet_mask_x (ny, nx) - 1.0 on left/right Dirichlet boundary cells
        1: bc_dirichlet_val_x  (ny, nx) - Prescribed Dirichlet temperature (normalized)
        2: bc_neumann_mask_x   (ny, nx) - 1.0 on left/right Neumann boundary cells
        3: bc_neumann_val_x    (ny, nx) - Prescribed outward heat flux (normalized)
        4: bc_dirichlet_mask_y (ny, nx) - 1.0 on bottom/top Dirichlet boundary cells
        5: bc_dirichlet_val_y  (ny, nx) - Prescribed Dirichlet temperature (normalized)
        6: bc_neumann_mask_y   (ny, nx) - 1.0 on bottom/top Neumann boundary cells
        7: bc_neumann_val_y    (ny, nx) - Prescribed outward heat flux (normalized)

    Args:
        bc_source: Either a BoundaryConditions instance or dictionary of evaluated values
            with keys 'left_type', 'left_value', 'right_type', etc.
        grid: 2D computational domain grid.
        normalizer: Optional FieldNormalizer used to scale Dirichlet temperatures into
            the same normalized space as solution temperatures.
        flux_scale: Characteristic scale [W/m^2] used to normalize outward heat fluxes.

    Returns:
        Array of shape (8, ny, nx), dtype float32.
    """
    ny, nx = grid.ny, grid.nx
    bc_tensor = np.zeros((8, ny, nx), dtype=np.float32)

    # Extract evaluated boundary arrays and types
    if isinstance(bc_source, BoundaryConditions):
        evals = bc_source.evaluate_all(grid)
        l_type = bc_source.left.bc_type.value
        r_type = bc_source.right.bc_type.value
        b_type = bc_source.bottom.bc_type.value
        t_type = bc_source.top.bc_type.value
        l_val = evals["left"].astype(np.float32)
        r_val = evals["right"].astype(np.float32)
        b_val = evals["bottom"].astype(np.float32)
        t_val = evals["top"].astype(np.float32)
    else:

        def get_type_str(val: Any) -> str:
            if isinstance(val, bytes):
                return val.decode("utf-8").lower()
            return str(val).lower()

        l_type = get_type_str(bc_source["left_type"])
        r_type = get_type_str(bc_source["right_type"])
        b_type = get_type_str(bc_source["bottom_type"])
        t_type = get_type_str(bc_source["top_type"])
        l_val = np.asarray(bc_source["left_value"], dtype=np.float32)
        r_val = np.asarray(bc_source["right_value"], dtype=np.float32)
        b_val = np.asarray(bc_source["bottom_value"], dtype=np.float32)
        t_val = np.asarray(bc_source["top_value"], dtype=np.float32)

    # Normalize Dirichlet temperature values if normalizer is fitted
    def norm_temp(val: np.ndarray) -> np.ndarray:
        if normalizer is not None and normalizer.is_fitted:
            return normalizer.transform_temperature(val).astype(np.float32)
        return val

    # Normalize outward Neumann fluxes by characteristic flux scale
    def norm_flux(val: np.ndarray) -> np.ndarray:
        scale = flux_scale if flux_scale > 1e-12 else 1.0
        return (val / scale).astype(np.float32)

    # 1. Left boundary (x = x_min, column i = 0)
    if l_type == "dirichlet":
        bc_tensor[0, :, 0] = 1.0
        bc_tensor[1, :, 0] = norm_temp(l_val)
    else:
        bc_tensor[2, :, 0] = 1.0
        bc_tensor[3, :, 0] = norm_flux(l_val)

    # 2. Right boundary (x = x_max, column i = nx - 1)
    if r_type == "dirichlet":
        bc_tensor[0, :, nx - 1] = 1.0
        bc_tensor[1, :, nx - 1] = norm_temp(r_val)
    else:
        bc_tensor[2, :, nx - 1] = 1.0
        bc_tensor[3, :, nx - 1] = norm_flux(r_val)

    # 3. Bottom boundary (y = y_min, row j = 0)
    if b_type == "dirichlet":
        bc_tensor[4, 0, :] = 1.0
        bc_tensor[5, 0, :] = norm_temp(b_val)
    else:
        bc_tensor[6, 0, :] = 1.0
        bc_tensor[7, 0, :] = norm_flux(b_val)

    # 4. Top boundary (y = y_max, row j = ny - 1)
    if t_type == "dirichlet":
        bc_tensor[4, ny - 1, :] = 1.0
        bc_tensor[5, ny - 1, :] = norm_temp(t_val)
    else:
        bc_tensor[6, ny - 1, :] = 1.0
        bc_tensor[7, ny - 1, :] = norm_flux(t_val)

    return bc_tensor


def decode_boundary_conditions(
    bc_tensor: np.ndarray,
    grid: Grid2D,
    normalizer: FieldNormalizer | None = None,
    flux_scale: float = 1000.0,
) -> BoundaryConditions:
    """Reconstruct physical BoundaryConditions object from an 8-channel BC tensor.

    Args:
        bc_tensor: Array of shape (8, ny, nx) or (12, ny, nx). If 12 channels, the last 8
            channels are used.
        grid: 2D computational grid.
        normalizer: Optional FieldNormalizer to invert normalized Dirichlet temperatures back to Kelvin.
        flux_scale: Characteristic scale [W/m^2] to invert normalized heat fluxes.

    Returns:
        Exact reconstructed BoundaryConditions instance.
    """
    if bc_tensor.shape[0] == 12:
        bc_data = bc_tensor[4:12]
    elif bc_tensor.shape[0] == 8:
        bc_data = bc_tensor
    else:
        raise ValueError(f"Expected 8 or 12 channels in bc_tensor, got shape {bc_tensor.shape}.")

    ny, nx = grid.ny, grid.nx

    def inv_temp(val: np.ndarray) -> np.ndarray:
        if normalizer is not None and normalizer.is_fitted:
            return normalizer.inverse_transform_temperature(val).astype(np.float64)
        return val.astype(np.float64)

    def inv_flux(val: np.ndarray) -> np.ndarray:
        return (val * flux_scale).astype(np.float64)

    # 1. Left boundary (i = 0)
    if float(np.mean(bc_data[0, :, 0])) > 0.5:
        l_type = BoundaryType.DIRICHLET
        l_val = inv_temp(bc_data[1, :, 0])
    else:
        l_type = BoundaryType.NEUMANN
        l_val = inv_flux(bc_data[3, :, 0])

    # 2. Right boundary (i = nx - 1)
    if float(np.mean(bc_data[0, :, nx - 1])) > 0.5:
        r_type = BoundaryType.DIRICHLET
        r_val = inv_temp(bc_data[1, :, nx - 1])
    else:
        r_type = BoundaryType.NEUMANN
        r_val = inv_flux(bc_data[3, :, nx - 1])

    # 3. Bottom boundary (j = 0)
    if float(np.mean(bc_data[4, 0, :])) > 0.5:
        b_type = BoundaryType.DIRICHLET
        b_val = inv_temp(bc_data[5, 0, :])
    else:
        b_type = BoundaryType.NEUMANN
        b_val = inv_flux(bc_data[7, 0, :])

    # 4. Top boundary (j = ny - 1)
    if float(np.mean(bc_data[4, ny - 1, :])) > 0.5:
        t_type = BoundaryType.DIRICHLET
        t_val = inv_temp(bc_data[5, ny - 1, :])
    else:
        t_type = BoundaryType.NEUMANN
        t_val = inv_flux(bc_data[7, ny - 1, :])

    return BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, l_type, l_val),
        right=BoundaryCondition(BoundarySide.RIGHT, r_type, r_val),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, b_type, b_val),
        top=BoundaryCondition(BoundarySide.TOP, t_type, t_val),
    )
