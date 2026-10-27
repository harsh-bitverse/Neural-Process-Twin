"""Numerical convergence and Grid Convergence Index (GCI) tools.

Implements Roache's ASME-standard Grid Convergence Index (Roache, 1994, 1998)
and error norm evaluations for verification studies.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def compute_relative_l2_error(u_num: np.ndarray, u_exact: np.ndarray) -> float:
    """Calculate the discrete relative L2 error norm.

    Args:
        u_num: Numerical solution array.
        u_exact: Analytical / exact reference array of matching shape.

    Returns:
        Relative L2 error ||u_num - u_exact||_2 / ||u_exact||_2.
    """
    diff = u_num - u_exact
    norm_diff = float(np.sqrt(np.sum(diff**2)))
    norm_exact = float(np.sqrt(np.sum(u_exact**2)))
    if norm_exact == 0.0:
        return norm_diff
    return norm_diff / norm_exact


def compute_linf_error(u_num: np.ndarray, u_exact: np.ndarray) -> float:
    """Calculate the maximum pointwise absolute error (L_infinity norm).

    Args:
        u_num: Numerical solution array.
        u_exact: Analytical reference array of matching shape.

    Returns:
        max |u_num - u_exact|.
    """
    return float(np.max(np.abs(u_num - u_exact)))


def estimate_convergence_rates(
    errors: Sequence[float],
    resolutions: Sequence[int],
) -> list[float]:
    """Estimate observed convergence order p across a sequence of refined grids.

    For grid sequence with refinement ratio r = N_{k+1} / N_k:
        p = log(E_k / E_{k+1}) / log(r)

    Args:
        errors: Sequence of error norms [E_0, E_1, ..., E_{M-1}].
        resolutions: Corresponding grid resolutions [N_0, N_1, ..., N_{M-1}].

    Returns:
        List of observed convergence orders of length M - 1.
    """
    if len(errors) != len(resolutions):
        raise ValueError("errors and resolutions sequences must have identical lengths.")
    if len(errors) < 2:
        raise ValueError(
            "At least two grid resolutions are required to calculate convergence rate."
        )

    rates: list[float] = []
    for k in range(len(errors) - 1):
        e1, e2 = errors[k], errors[k + 1]
        n1, n2 = resolutions[k], resolutions[k + 1]
        r = n2 / n1
        if e1 <= 0.0 or e2 <= 0.0:
            rates.append(0.0)
        else:
            p = float(np.log(e1 / e2) / np.log(r))
            rates.append(p)
    return rates


def calculate_gci(
    f1: float,
    f2: float,
    f3: float,
    r: float = 2.0,
    safety_factor: float = 1.25,
) -> dict[str, float]:
    """Compute the Grid Convergence Index (GCI) across three grid solutions.

    Follows Roache (1994, 1998) / ASME V&V 20 standards:
        - f1: Fine grid solution (finest resolution h1).
        - f2: Medium grid solution (h2 = r * h1).
        - f3: Coarse grid solution (h3 = r * h2 = r^2 * h1).
        - r: Grid refinement ratio (defaults to 2.0).
        - safety_factor: Fs = 1.25 for three-grid study with observed order.

    Returns:
        Dictionary containing:
            - 'order_p': Observed apparent order of convergence.
            - 'extrapolated_value': Richardson extrapolated value on infinitely fine grid.
            - 'relative_error_21': Relative step difference |f1 - f2| / |f1|.
            - 'gci_fine': Fine-grid GCI (percentage uncertainty in f1).
            - 'gci_medium': Medium-grid GCI (percentage uncertainty in f2).
            - 'asymptotic_ratio': GCI_medium / (r^p * GCI_fine), should be ~1.0 if in asymptotic range.
    """
    if r <= 1.0:
        raise ValueError(f"Refinement ratio r must be > 1.0, got {r}.")

    eps21 = f2 - f1
    eps32 = f3 - f2

    # Check monotonic convergence
    apparent_order = 0.0 if eps21 * eps32 <= 0.0 else float(np.log(abs(eps32 / eps21)) / np.log(r))

    denom = (r**apparent_order - 1.0) if apparent_order > 0.0 else 1.0

    extrapolated = f1 + (f1 - f2) / denom if denom != 0.0 else f1

    scale1 = abs(f1) if abs(f1) > 1e-12 else 1.0
    scale2 = abs(f2) if abs(f2) > 1e-12 else 1.0

    rel_err21 = abs(eps21) / scale1
    rel_err32 = abs(eps32) / scale2

    gci_fine = (safety_factor * rel_err21 / abs(denom)) * 100.0 if denom != 0.0 else 0.0
    gci_medium = (safety_factor * rel_err32 / abs(denom)) * 100.0 if denom != 0.0 else 0.0

    asymp_denom = (r**apparent_order * gci_fine) if gci_fine > 0.0 else 1.0
    asymptotic_ratio = gci_medium / asymp_denom if asymp_denom != 0.0 else 1.0

    return {
        "order_p": apparent_order,
        "extrapolated_value": float(extrapolated),
        "relative_error_21": float(rel_err21),
        "gci_fine": float(gci_fine),
        "gci_medium": float(gci_medium),
        "asymptotic_ratio": float(asymptotic_ratio),
    }
