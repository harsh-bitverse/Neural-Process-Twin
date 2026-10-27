"""Harmonic conductivity averaging for cell-centred finite-volume discretization.

In steady-state heat conduction across heterogeneous media, the conductive heat flux
j = -k * dT/dn must be continuous across material interfaces. Placing the interface
between adjacent control volumes P and N, the two half-cell thermal resistances act
in series:
    R_total = R_P + R_N = (dx / (2 * k_P * A)) + (dx / (2 * k_N * A))
                        = (dx / A) * ((k_P + k_N) / (2 * k_P * k_N))

Setting R_total equal to dx / (k_f * A) yields the interface harmonic mean:
    k_f = 2 * k_P * k_N / (k_P + k_N)

Using an arithmetic average (k_P + k_N) / 2 violates flux continuity and produces
unphysically high conductance when an insulator adjoins a conductor.
"""

from __future__ import annotations

import numpy as np


def harmonic_average(
    k1: float | np.ndarray,
    k2: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the harmonic average of two thermal conductivities.

    Args:
        k1: Conductivity at cell P [W/(m K)]. Must be strictly positive and finite.
        k2: Conductivity at neighbor N [W/(m K)]. Must be strictly positive and finite.

    Returns:
        Harmonic average 2 * k1 * k2 / (k1 + k2).

    Raises:
        ValueError: If any conductivity is non-positive, NaN, or infinite.
    """
    arr1 = np.asarray(k1, dtype=np.float64)
    arr2 = np.asarray(k2, dtype=np.float64)

    if not np.all(np.isfinite(arr1)) or not np.all(np.isfinite(arr2)):
        raise ValueError("Conductivity values must be finite (no NaN or Inf).")

    if np.any(arr1 <= 0.0) or np.any(arr2 <= 0.0):
        raise ValueError(
            "Thermal conductivity must be strictly positive (k > 0) to ensure uniform ellipticity."
        )

    result = (2.0 * arr1 * arr2) / (arr1 + arr2)
    if isinstance(k1, (int, float, np.floating)) and isinstance(k2, (int, float, np.floating)):
        return float(result)
    return result


def arithmetic_average(
    k1: float | np.ndarray,
    k2: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate the arithmetic average of two conductivities (provided for comparison).

    Note:
        The arithmetic average should NOT be used for interface conductivities in
        heterogeneous media because it overestimates interface conductance across
        thermal resistance jumps.
    """
    arr1 = np.asarray(k1, dtype=np.float64)
    arr2 = np.asarray(k2, dtype=np.float64)
    result = 0.5 * (arr1 + arr2)
    if isinstance(k1, (int, float, np.floating)) and isinstance(k2, (int, float, np.floating)):
        return float(result)
    return result


def compute_internal_face_conductivities(
    k_field: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute harmonic interface conductivities across all internal cell faces.

    Args:
        k_field: 2D array of cell-centred conductivities of shape (ny, nx) [W/(m K)].

    Returns:
        Tuple of (k_east, k_north):
            - k_east: Array of shape (ny, nx - 1) for vertical internal faces normal to x.
            - k_north: Array of shape (ny - 1, nx) for horizontal internal faces normal to y.

    Raises:
        ValueError: If k_field contains non-positive or non-finite values.
    """
    if k_field.ndim != 2:
        raise ValueError(f"k_field must be 2D, got shape {k_field.shape}.")

    ny, nx = k_field.shape
    if nx < 2 or ny < 2:
        raise ValueError(f"k_field shape {k_field.shape} too small for internal faces.")

    if not np.all(np.isfinite(k_field)):
        raise ValueError("k_field contains non-finite values.")

    if np.any(k_field <= 0.0):
        min_k = float(np.min(k_field))
        raise ValueError(f"k_field must be strictly positive; minimum value was {min_k}.")

    # East internal faces between column i and i+1
    k_p_east = k_field[:, :-1]
    k_n_east = k_field[:, 1:]
    k_east = harmonic_average(k_p_east, k_n_east)

    # North internal faces between row j and j+1
    k_p_north = k_field[:-1, :]
    k_n_north = k_field[1:, :]
    k_north = harmonic_average(k_p_north, k_n_north)

    return np.asarray(k_east, dtype=np.float64), np.asarray(k_north, dtype=np.float64)
