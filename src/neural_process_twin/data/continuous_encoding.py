"""Resolution-consistent continuous physical input encoding for 2D neural operators.

Milestone 08C Phase 3: Continuous Physical Input Encoding.
Replaces mesh-dependent single-cell boundary masks with continuous physical fields:
- Smooth coordinate distance weighting across the physical domain
- Continuous representation of boundary temperatures and heat fluxes
- Support for both linear conductivity and logarithmic conductivity
- Guaranteed discretization invariance across 32x32, 64x64, 128x128, and 256x256 grids
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

CONTINUOUS_CHANNEL_NAMES: list[str] = [
    "conductivity",  # Ch 0: Normalized k(x, y) or ln(k(x, y))
    "heat_source",  # Ch 1: Normalized q(x, y)
    "coord_x",  # Ch 2: Normalized x in [0, 1]
    "coord_y",  # Ch 3: Normalized y in [0, 1]
    "cont_bc_dirichlet_west",  # Ch 4: (1 - x/L_x) * I(West is Dirichlet) * T_west_norm(y)
    "cont_bc_dirichlet_east",  # Ch 5: (x/L_x) * I(East is Dirichlet) * T_east_norm(y)
    "cont_bc_neumann_west",  # Ch 6: (1 - x/L_x) * I(West is Neumann) * j_west_norm(y)
    "cont_bc_neumann_east",  # Ch 7: (x/L_x) * I(East is Neumann) * j_east_norm(y)
    "cont_bc_dirichlet_south",  # Ch 8: (1 - y/L_y) * I(South is Dirichlet) * T_south_norm(x)
    "cont_bc_dirichlet_north",  # Ch 9: (y/L_y) * I(North is Dirichlet) * T_north_norm(x)
    "cont_bc_neumann_south",  # Ch 10: (1 - y/L_y) * I(South is Neumann) * j_south_norm(x)
    "cont_bc_neumann_north",  # Ch 11: (y/L_y) * I(North is Neumann) * j_north_norm(x)
]

NUM_CONTINUOUS_CHANNELS = len(CONTINUOUS_CHANNEL_NAMES)  # 12


def encode_continuous_physical_inputs(
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions | dict[str, Any],
    grid: Grid2D,
    normalizer: FieldNormalizer | None = None,
    use_log_k: bool = False,
    flux_scale: float = 1000.0,
    log_k_mean: float = 0.5,
    log_k_std: float = 1.0,
) -> np.ndarray:
    """Encode physical problem fields into a 12-channel continuous input tensor.

    Args:
        k_field: 2D conductivity field [W/(m K)] of shape (ny, nx).
        q_field: 2D heat source field [W/m^3] of shape (ny, nx).
        bcs: Prescribed boundary conditions across all 4 domain edges.
        grid: Computational grid.
        normalizer: FieldNormalizer fitted to training data.
        use_log_k: If True, encodes ln(k) in channel 0 instead of linear k.
        flux_scale: Scale for normalizing Neumann outward fluxes [W/m^2].
        log_k_mean: Training set mean of ln(k) for log-conductivity normalization.
        log_k_std: Training set std of ln(k) for log-conductivity normalization.

    Returns:
        Tensor of shape (12, ny, nx), dtype float32.
    """
    ny, nx = grid.ny, grid.nx
    tensor = np.zeros((12, ny, nx), dtype=np.float32)

    # 1. Conductivity channel (Ch 0)
    if use_log_k:
        log_k = np.log(np.maximum(k_field, 1e-12))
        k_norm = (log_k - log_k_mean) / (log_k_std if log_k_std > 1e-12 else 1.0)
        tensor[0] = k_norm.astype(np.float32)
    else:
        if normalizer is not None and normalizer.is_fitted and normalizer.k_stats is not None:
            tensor[0] = normalizer.transform_k(k_field).astype(np.float32)
        else:
            tensor[0] = k_field.astype(np.float32)

    # 2. Heat source channel (Ch 1)
    if normalizer is not None and normalizer.is_fitted and normalizer.q_stats is not None:
        tensor[1] = normalizer.transform_q(q_field).astype(np.float32)
    else:
        tensor[1] = q_field.astype(np.float32)

    # 3. Normalized spatial coordinates in [0, 1] (Ch 2, Ch 3)
    x_norm = (grid.x_mesh - grid.x_min) / grid.lx
    y_norm = (grid.y_mesh - grid.y_min) / grid.ly
    tensor[2] = x_norm.astype(np.float32)
    tensor[3] = y_norm.astype(np.float32)

    # Extract boundary values
    if isinstance(bcs, BoundaryConditions):
        evals = bcs.evaluate_all(grid)
        l_type = bcs.left.bc_type.value
        r_type = bcs.right.bc_type.value
        b_type = bcs.bottom.bc_type.value
        t_type = bcs.top.bc_type.value
        l_val = evals["left"]
        r_val = evals["right"]
        b_val = evals["bottom"]
        t_val = evals["top"]
    else:

        def get_type_str(val: Any) -> str:
            if isinstance(val, bytes):
                return val.decode("utf-8").lower()
            return str(val).lower()

        l_type = get_type_str(bcs["left_type"])
        r_type = get_type_str(bcs["right_type"])
        b_type = get_type_str(bcs["bottom_type"])
        t_type = get_type_str(bcs["top_type"])
        l_val = np.asarray(bcs["left_value"], dtype=np.float64)
        r_val = np.asarray(bcs["right_value"], dtype=np.float64)
        b_val = np.asarray(bcs["bottom_value"], dtype=np.float64)
        t_val = np.asarray(bcs["top_value"], dtype=np.float64)

    # Normalization helpers
    def norm_t(val: np.ndarray) -> np.ndarray:
        if normalizer is not None and normalizer.is_fitted and normalizer.t_stats is not None:
            return normalizer.transform_temperature(val)
        return val

    def norm_j(val: np.ndarray) -> np.ndarray:
        return val / (flux_scale if flux_scale > 1e-12 else 1.0)

    # 4. Continuous West boundary (x = 0, weight = 1 - x_norm)
    # Broadcast (ny,) values across columns along x-axis
    l_val_2d = np.repeat(l_val[:, np.newaxis], nx, axis=1)
    weight_west = 1.0 - x_norm
    if l_type == "dirichlet":
        tensor[4] = (weight_west * norm_t(l_val_2d)).astype(np.float32)
    else:
        tensor[6] = (weight_west * norm_j(l_val_2d)).astype(np.float32)

    # 5. Continuous East boundary (x = 1, weight = x_norm)
    r_val_2d = np.repeat(r_val[:, np.newaxis], nx, axis=1)
    weight_east = x_norm
    if r_type == "dirichlet":
        tensor[5] = (weight_east * norm_t(r_val_2d)).astype(np.float32)
    else:
        tensor[7] = (weight_east * norm_j(r_val_2d)).astype(np.float32)

    # 6. Continuous South boundary (y = 0, weight = 1 - y_norm)
    # Broadcast (nx,) values across rows along y-axis
    b_val_2d = np.repeat(b_val[np.newaxis, :], ny, axis=0)
    weight_south = 1.0 - y_norm
    if b_type == "dirichlet":
        tensor[8] = (weight_south * norm_t(b_val_2d)).astype(np.float32)
    else:
        tensor[10] = (weight_south * norm_j(b_val_2d)).astype(np.float32)

    # 7. Continuous North boundary (y = 1, weight = y_norm)
    t_val_2d = np.repeat(t_val[np.newaxis, :], ny, axis=0)
    weight_north = y_norm
    if t_type == "dirichlet":
        tensor[9] = (weight_north * norm_t(t_val_2d)).astype(np.float32)
    else:
        tensor[11] = (weight_north * norm_j(t_val_2d)).astype(np.float32)

    return tensor


def decode_continuous_boundary_conditions(
    tensor: np.ndarray,
    grid: Grid2D,
    normalizer: FieldNormalizer | None = None,
    flux_scale: float = 1000.0,
) -> BoundaryConditions:
    """Decode physical boundary conditions from a continuous 12-channel input tensor.

    Args:
        tensor: Array of shape (12, ny, nx) or (8, ny, nx).
        grid: Computational grid.
        normalizer: FieldNormalizer instance for inverting temperature scaling.
        flux_scale: Flux normalization scale [W/m^2].

    Returns:
        Reconstructed BoundaryConditions object.
    """
    if tensor.shape[0] == 12:
        bc_channels = tensor[4:12]
    elif tensor.shape[0] == 8:
        bc_channels = tensor
    else:
        raise ValueError(f"Expected 8 or 12 channels, got shape {tensor.shape}")

    # Channels:
    # 0: dirichlet_west, 1: dirichlet_east, 2: neumann_west, 3: neumann_east
    # 4: dirichlet_south, 5: dirichlet_north, 6: neumann_south, 7: neumann_north

    # West (column 0): weight = 1 - x_norm = 1.0 - 0.5/nx approx 1.0
    w_dir = bc_channels[0, :, 0]
    w_neu = bc_channels[2, :, 0]
    w_factor = 1.0 - (grid.x_centers[0] - grid.x_min) / grid.lx
    if np.max(np.abs(w_dir)) >= np.max(np.abs(w_neu)):
        l_type = BoundaryType.DIRICHLET
        raw_val = w_dir / w_factor
        l_val = (
            normalizer.inverse_transform_temperature(raw_val)
            if normalizer and normalizer.is_fitted and normalizer.t_stats
            else raw_val
        )
    else:
        l_type = BoundaryType.NEUMANN
        l_val = (w_neu / w_factor) * flux_scale

    # East (column -1): weight = x_norm approx 1.0
    e_dir = bc_channels[1, :, -1]
    e_neu = bc_channels[3, :, -1]
    e_factor = (grid.x_centers[-1] - grid.x_min) / grid.lx
    if np.max(np.abs(e_dir)) >= np.max(np.abs(e_neu)):
        r_type = BoundaryType.DIRICHLET
        raw_val = e_dir / e_factor
        r_val = (
            normalizer.inverse_transform_temperature(raw_val)
            if normalizer and normalizer.is_fitted and normalizer.t_stats
            else raw_val
        )
    else:
        r_type = BoundaryType.NEUMANN
        r_val = (e_neu / e_factor) * flux_scale

    # South (row 0): weight = 1 - y_norm approx 1.0
    s_dir = bc_channels[4, 0, :]
    s_neu = bc_channels[6, 0, :]
    s_factor = 1.0 - (grid.y_centers[0] - grid.y_min) / grid.ly
    if np.max(np.abs(s_dir)) >= np.max(np.abs(s_neu)):
        b_type = BoundaryType.DIRICHLET
        raw_val = s_dir / s_factor
        b_val = (
            normalizer.inverse_transform_temperature(raw_val)
            if normalizer and normalizer.is_fitted and normalizer.t_stats
            else raw_val
        )
    else:
        b_type = BoundaryType.NEUMANN
        b_val = (s_neu / s_factor) * flux_scale

    # North (row -1): weight = y_norm approx 1.0
    n_dir = bc_channels[5, -1, :]
    n_neu = bc_channels[7, -1, :]
    n_factor = (grid.y_centers[-1] - grid.y_min) / grid.ly
    if np.max(np.abs(n_dir)) >= np.max(np.abs(n_neu)):
        t_type = BoundaryType.DIRICHLET
        raw_val = n_dir / n_factor
        t_val = (
            normalizer.inverse_transform_temperature(raw_val)
            if normalizer and normalizer.is_fitted and normalizer.t_stats
            else raw_val
        )
    else:
        t_type = BoundaryType.NEUMANN
        t_val = (n_neu / n_factor) * flux_scale

    return BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, l_type, l_val),
        right=BoundaryCondition(BoundarySide.RIGHT, r_type, r_val),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, b_type, b_val),
        top=BoundaryCondition(BoundarySide.TOP, t_type, t_val),
    )
