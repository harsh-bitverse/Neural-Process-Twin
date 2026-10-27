"""Method of Manufactured Solutions (MMS) convergence verification tests.

Verifies theoretical second-order asymptotic convergence (O(h^2)) of the cell-centred
finite-volume reference solver on uniform Cartesian grid refinements.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_process_twin.physics.manufactured import (
    SinusoidalMMSConstantK,
    SinusoidalMMSVariableK,
)
from neural_process_twin.solvers.convergence import (
    calculate_gci,
    compute_linf_error,
    compute_relative_l2_error,
    estimate_convergence_rates,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D


@pytest.mark.scientific
def test_mms_constant_conductivity_convergence() -> None:
    """Verify second-order convergence on manufactured solution with constant conductivity."""
    mms = SinusoidalMMSConstantK(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        k_0=1.5,
        t_0=300.0,
        amplitude=25.0,
    )
    bcs = mms.boundary_conditions()
    solver = HeatConductionSolver2D(tolerance=1e-12)

    resolutions = [16, 32, 64, 128]
    l2_errors: list[float] = []
    linf_errors: list[float] = []

    for n in resolutions:
        grid = Grid2D(
            x_min=mms.x_min,
            x_max=mms.x_max,
            y_min=mms.y_min,
            y_max=mms.y_max,
            nx=n,
            ny=n,
        )
        res = solver.solve(
            grid=grid,
            conductivity=mms.conductivity,
            source=mms.source,
            boundary_conditions=bcs,
        )
        t_exact = mms.temperature(grid.x_mesh, grid.y_mesh)

        err_l2 = compute_relative_l2_error(res.temperature, t_exact)
        err_linf = compute_linf_error(res.temperature, t_exact)

        l2_errors.append(err_l2)
        linf_errors.append(err_linf)

    # Compute observed convergence orders
    rates_l2 = estimate_convergence_rates(l2_errors, resolutions)
    rates_linf = estimate_convergence_rates(linf_errors, resolutions)

    # Verify that errors strictly decrease monotonically
    for i in range(len(l2_errors) - 1):
        assert l2_errors[i + 1] < l2_errors[i] / 3.8, (
            f"Error failed to decrease by factor ~4 at N={resolutions[i + 1]}"
        )

    # Verify asymptotic convergence rate is close to theoretical order 2.0
    for p in rates_l2:
        assert 1.95 <= p <= 2.05, f"Observed L2 convergence rate {p:.4f} outside [1.95, 2.05]."

    for p in rates_linf:
        assert 1.95 <= p <= 2.05, f"Observed Linf convergence rate {p:.4f} outside [1.95, 2.05]."


@pytest.mark.scientific
def test_mms_variable_conductivity_convergence() -> None:
    """Verify second-order convergence on manufactured solution with spatially varying k(x,y)."""
    mms = SinusoidalMMSVariableK(
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        k_0=2.0,
        k_1=0.5,
        t_0=300.0,
        amplitude=25.0,
    )
    bcs = mms.boundary_conditions()
    solver = HeatConductionSolver2D(tolerance=1e-12)

    resolutions = [16, 32, 64, 128]
    l2_errors: list[float] = []

    for n in resolutions:
        grid = Grid2D(
            x_min=mms.x_min,
            x_max=mms.x_max,
            y_min=mms.y_min,
            y_max=mms.y_max,
            nx=n,
            ny=n,
        )
        res = solver.solve(
            grid=grid,
            conductivity=mms.conductivity,
            source=mms.source,
            boundary_conditions=bcs,
        )
        t_exact = mms.temperature(grid.x_mesh, grid.y_mesh)
        err_l2 = compute_relative_l2_error(res.temperature, t_exact)
        l2_errors.append(err_l2)

    rates = estimate_convergence_rates(l2_errors, resolutions)

    # All observed rates should be in [1.95, 2.05]
    for p in rates:
        assert 1.95 <= p <= 2.05, (
            f"Observed variable-k convergence rate {p:.4f} outside [1.95, 2.05]."
        )


@pytest.mark.scientific
def test_grid_convergence_index_mms() -> None:
    """Verify Grid Convergence Index (GCI) on three refined grid solutions."""
    mms = SinusoidalMMSConstantK(amplitude=30.0)
    bcs = mms.boundary_conditions()
    solver = HeatConductionSolver2D()

    # Solutions on N = 32 (coarse), 64 (medium), 128 (fine)
    resolutions = [32, 64, 128]
    center_temps: list[float] = []

    for n in resolutions:
        grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=n, ny=n)
        res = solver.solve(
            grid=grid,
            conductivity=mms.conductivity,
            source=mms.source,
            boundary_conditions=bcs,
        )
        # Sample solution value at the physical centre (0.5, 0.5)
        # For even nx, ny, average the 4 cells surrounding centre
        c_i, c_j = n // 2, n // 2
        t_mid = float(np.mean(res.temperature[c_j - 1 : c_j + 1, c_i - 1 : c_i + 1]))
        center_temps.append(t_mid)

    f3, f2, f1 = center_temps[0], center_temps[1], center_temps[2]
    gci = calculate_gci(f1=f1, f2=f2, f3=f3, r=2.0, safety_factor=1.25)

    assert 1.90 <= gci["order_p"] <= 2.10, f"Apparent order {gci['order_p']} outside [1.90, 2.10]."
    assert gci["gci_fine"] < 0.05, f"Fine GCI {gci['gci_fine']}% should be small on 128x128 grid."
    assert 0.95 <= gci["asymptotic_ratio"] <= 1.05, (
        f"Asymptotic ratio {gci['asymptotic_ratio']} should be approximately 1.0."
    )
