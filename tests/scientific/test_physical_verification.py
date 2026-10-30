"""Independent physical verification tests for 2D steady-state heat conduction.

Verifies conservation laws, boundary treatments, symmetry, positivity, and physical limits.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_process_twin.physics.manufactured import LinearConduction1D
from neural_process_twin.solvers.assembly import PureNeumannSingularError
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.conductivity import arithmetic_average, harmonic_average
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D


@pytest.mark.scientific
def test_constant_solution() -> None:
    """Test 1: Zero source with matching constant Dirichlet temperatures produces a constant field."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=20, ny=20)
    t_const = 325.0

    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, t_const),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, t_const),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, t_const),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, t_const),
    )
    solver = HeatConductionSolver2D()
    res = solver.solve(grid=grid, conductivity=2.5, source=0.0, boundary_conditions=bcs)

    max_err = float(np.max(np.abs(res.temperature - t_const)))
    assert max_err < 1.0e-12, f"Constant solution error {max_err} exceeds 1e-12."


@pytest.mark.scientific
def test_linear_temperature_field() -> None:
    """Test 2: Constant k, zero source, and compatible Dirichlet boundaries reproduce 1D linear profile."""
    linear = LinearConduction1D(
        x_min=0.0, x_max=2.0, y_min=0.0, y_max=1.0, k_0=1.5, t_left=300.0, t_right=450.0
    )
    grid = Grid2D(
        x_min=linear.x_min, x_max=linear.x_max, y_min=linear.y_min, y_max=linear.y_max, nx=24, ny=12
    )
    solver = HeatConductionSolver2D()

    res = solver.solve(
        grid=grid,
        conductivity=linear.conductivity,
        source=linear.source,
        boundary_conditions=linear.boundary_conditions(),
    )
    t_exact = linear.temperature(grid.x_mesh, grid.y_mesh)
    max_err = float(np.max(np.abs(res.temperature - t_exact)))
    assert max_err < 1.0e-11, f"Linear temperature error {max_err} exceeds 1e-11."


@pytest.mark.scientific
def test_spatial_symmetry() -> None:
    """Test 3: A spatially symmetric physical problem yields a numerically symmetric field."""
    grid = Grid2D(x_min=-1.0, x_max=1.0, y_min=-1.0, y_max=1.0, nx=32, ny=32)

    # Symmetric Gaussian heat source at origin
    def symmetric_source(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1000.0 * np.exp(-10.0 * (x**2 + y**2))

    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 300.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, 300.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, 300.0),
    )
    solver = HeatConductionSolver2D()
    res = solver.solve(
        grid=grid, conductivity=1.0, source=symmetric_source, boundary_conditions=bcs
    )

    t = res.temperature
    # Check left-right symmetry: t(x, y) == t(-x, y) -> flip horizontally
    diff_lr = float(np.max(np.abs(t - np.fliplr(t))))
    # Check bottom-top symmetry: t(x, y) == t(x, -y) -> flip vertically
    diff_bt = float(np.max(np.abs(t - np.flipud(t))))
    # Check diagonal symmetry: t(x, y) == t(y, x) -> transpose
    diff_diag = float(np.max(np.abs(t - t.T)))

    assert diff_lr < 1.0e-12, f"Left-right symmetry error {diff_lr} exceeds 1e-12."
    assert diff_bt < 1.0e-12, f"Bottom-top symmetry error {diff_bt} exceeds 1e-12."
    assert diff_diag < 1.0e-12, f"Diagonal symmetry error {diff_diag} exceeds 1e-12."


@pytest.mark.scientific
def test_sparse_matrix_residual() -> None:
    """Test 4: Solution strictly satisfies assembled sparse matrix equation A * T = b."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=25, ny=25)

    def var_k(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1.0 + 0.5 * np.sin(np.pi * x) * np.cos(np.pi * y)

    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    solver = HeatConductionSolver2D()
    res = solver.solve(grid=grid, conductivity=var_k, source=100.0, boundary_conditions=bcs)

    assert res.diagnostics.is_converged
    assert res.diagnostics.linear_residual_l2 < 1.0e-10, (
        f"Sparse residual {res.diagnostics.linear_residual_l2} exceeds 1e-10."
    )
    assert res.diagnostics.relative_residual < 1.0e-11, (
        f"Relative residual {res.diagnostics.relative_residual} exceeds 1e-11."
    )


@pytest.mark.scientific
def test_conductivity_positivity_validation() -> None:
    """Test 5: Non-positive or non-finite thermal conductivity raises informative ValueError."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=10, ny=10)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 300.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, 300.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, 300.0),
    )
    solver = HeatConductionSolver2D()

    # Zero conductivity
    with pytest.raises(ValueError, match="strictly positive"):
        solver.solve(grid=grid, conductivity=0.0, source=0.0, boundary_conditions=bcs)

    # Negative conductivity
    with pytest.raises(ValueError, match="strictly positive"):
        solver.solve(grid=grid, conductivity=-1.5, source=0.0, boundary_conditions=bcs)

    # Direct harmonic average checks
    with pytest.raises(ValueError, match="strictly positive"):
        harmonic_average(1.0, -0.5)

    with pytest.raises(ValueError, match="strictly positive"):
        harmonic_average(0.0, 5.0)


@pytest.mark.scientific
def test_grid_geometry_validation() -> None:
    """Test 6: Invalid grid resolution, bounds, or thickness raise ValueError."""
    with pytest.raises(ValueError, match=r"nx .* must be >= 2"):
        Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=1, ny=10)

    with pytest.raises(ValueError, match=r"x_min .* must be strictly less than"):
        Grid2D(x_min=1.0, x_max=0.0, y_min=0.0, y_max=1.0, nx=10, ny=10)

    with pytest.raises(ValueError, match=r"thickness .* must be strictly positive"):
        Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=10, ny=10, thickness=-0.1)


@pytest.mark.scientific
def test_boundary_conditions_dirichlet_and_neumann() -> None:
    """Test 7: Support both positive and negative prescribed outward heat fluxes."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=20, ny=10)
    k_0 = 1.0
    t_left = 300.0
    solver = HeatConductionSolver2D()

    # Case A: Positive outward flux at right (heat leaves right wall, temperature drops)
    g_out_pos = 50.0
    bcs_pos = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, t_left),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, g_out_pos),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    res_pos = solver.solve(grid=grid, conductivity=k_0, source=0.0, boundary_conditions=bcs_pos)
    # Expected: T(x) = T_left - (g_out / k) * x
    t_expected_pos = t_left - (g_out_pos / k_0) * grid.x_mesh
    err_pos = float(np.max(np.abs(res_pos.temperature - t_expected_pos)))
    assert err_pos < 1.0e-11, f"Positive flux error {err_pos} exceeds 1e-11."

    # Case B: Negative outward flux at right (heat enters right wall, temperature rises)
    g_out_neg = -50.0
    bcs_neg = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, t_left),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, g_out_neg),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    res_neg = solver.solve(grid=grid, conductivity=k_0, source=0.0, boundary_conditions=bcs_neg)
    t_expected_neg = t_left - (g_out_neg / k_0) * grid.x_mesh
    err_neg = float(np.max(np.abs(res_neg.temperature - t_expected_neg)))
    assert err_neg < 1.0e-11, f"Negative flux error {err_neg} exceeds 1e-11."


@pytest.mark.scientific
def test_pure_neumann_singularity_and_gauge() -> None:
    """Test 8: Pure Neumann boundary condition raises PureNeumannSingularError unless gauge pinned."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=10, ny=10)
    pure_neumann_bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.NEUMANN, 0.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, 0.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    solver = HeatConductionSolver2D()

    # Reject unsupported singular pure Neumann
    with pytest.raises(PureNeumannSingularError, match="All four boundaries are Neumann"):
        solver.solve(grid=grid, conductivity=1.0, source=0.0, boundary_conditions=pure_neumann_bcs)

    # With gauge pin, solve succeeds and sets reference temperature
    res = solver.solve(
        grid=grid,
        conductivity=1.0,
        source=0.0,
        boundary_conditions=pure_neumann_bcs,
        gauge_pin=(0, 0),
        gauge_temperature=293.15,
    )
    assert abs(res.temperature[0, 0] - 293.15) < 1.0e-12
    # In insulated domain with zero source, entire field should be gauge temperature within machine precision
    assert float(np.max(np.abs(res.temperature - 293.15))) < 1.0e-11


@pytest.mark.scientific
def test_interface_harmonic_conductivity_averaging() -> None:
    """Test 9: Harmonic averaging correctly models discontinuous material interfaces."""
    # Composite bar: x in [0, 1]
    # x in [0, 0.5]: k1 = 1.0 W/(m K)
    # x in [0.5, 1]: k2 = 10.0 W/(m K)
    # Interface at x = 0.5
    # T(0) = 300 K, T(1) = 400 K, q = 0
    nx = 20
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=nx, ny=4)

    # Conductivity field
    k_field = np.where(grid.x_mesh < 0.5, 1.0, 10.0)

    # Analytical series resistance:
    # R_total = L1 / k1 + L2 / k2 = 0.5 / 1.0 + 0.5 / 10.0 = 0.55 m^2 K / W
    r_exact = 0.5 / 1.0 + 0.5 / 10.0
    q_exact = (400.0 - 300.0) / r_exact  # 181.81818... W/m^2 from x=1 to x=0

    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 400.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    solver = HeatConductionSolver2D()
    res = solver.solve(grid=grid, conductivity=k_field, source=0.0, boundary_conditions=bcs)

    # 1. Harmonic average interface value:
    # Cell i = nx//2 - 1 has k = 1.0, cell i = nx//2 has k = 10.0
    k_harm = harmonic_average(1.0, 10.0)
    k_arith = arithmetic_average(1.0, 10.0)
    assert abs(k_harm - (2.0 * 1.0 * 10.0 / 11.0)) < 1e-12
    assert abs(k_arith - 5.5) < 1e-12

    # 2. Check numerical heat flux:
    # Heat enters at right (outward flux is negative) and leaves at left (outward flux is positive)
    diag = res.diagnostics.energy_balance
    q_left_num = diag.boundary_heat_flow_out["left"] / (grid.ly * grid.thickness)
    q_right_num = diag.boundary_heat_flow_out["right"] / (grid.ly * grid.thickness)

    assert abs(q_left_num - (-q_exact)) < 1.0e-10 or abs(abs(q_left_num) - q_exact) < 1.0e-10
    assert abs(q_right_num - (-q_exact)) < 1.0e-10, (
        f"Harmonic solver right heat flux {q_right_num} deviates from expected {-q_exact}."
    )
    assert abs(q_left_num + q_right_num) < 1.0e-10, "Left and right heat flows must balance."

    # 3. If arithmetic mean were used, the equivalent total resistance would be wrong
    dx = 1.0 / nx
    r_arith = (0.5 - 0.5 * dx) / 1.0 + (0.5 - 0.5 * dx) / 10.0 + dx / k_arith
    q_arith = (400.0 - 300.0) / r_arith
    flux_discrepancy = abs(q_arith - q_exact)
    assert flux_discrepancy > 5.0, (
        "Arithmetic mean should cause noticeable error in composite flux."
    )


@pytest.mark.scientific
def test_global_physical_heat_balance() -> None:
    """Test 10: Verify global conservation between volumetric source and boundary heat flow."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=30, ny=30)

    # Non-uniform source and variable conductivity
    def var_source(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 500.0 * np.sin(np.pi * x) * np.sin(np.pi * y) + 200.0

    def var_cond(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1.5 + 0.5 * np.cos(np.pi * x) * np.cos(np.pi * y)

    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, 25.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, 320.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    solver = HeatConductionSolver2D()
    res = solver.solve(grid=grid, conductivity=var_cond, source=var_source, boundary_conditions=bcs)

    eb = res.diagnostics.energy_balance
    # Assert independent conservation: Q_out == Q_generated
    assert eb.relative_imbalance < 1.0e-12, (
        f"Relative heat balance imbalance {eb.relative_imbalance} exceeds 1e-12."
    )
    assert abs(eb.q_boundary_out_total - eb.q_generated_total) < 1.0e-10, (
        f"Heat flow imbalance {eb.flux_imbalance} W exceeds 1e-10 W."
    )


@pytest.mark.scientific
def test_signed_heat_source_and_heat_sink_conservation() -> None:
    """Test 11: Verify that endothermic heat sinks (q < 0) and mixed signed sources solve correctly.

    The governing elliptic PDE -div(k grad T) = q is linear with respect to the source term.
    Heat sinks (negative q) remove thermal energy, causing interior temperature to decrease
    below Dirichlet boundary temperatures while strictly preserving global energy conservation.
    """
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=32, ny=32)
    k_val = 2.0
    q_sink = -150.0  # Endothermic sink [W/m^3]
    t_boundary = 350.0

    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, t_boundary),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, t_boundary),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    solver = HeatConductionSolver2D()
    res = solver.solve(grid=grid, conductivity=k_val, source=q_sink, boundary_conditions=bcs)

    # 1. Physical temperature drop: interior must be cooler than boundary
    min_temp = float(np.min(res.temperature))
    assert min_temp < t_boundary, (
        f"Heat sink should cause temperature to drop below boundary ({t_boundary} K), got {min_temp} K."
    )

    # 2. Analytical comparison: 1D steady conduction with uniform sink:
    # d^2 T / dx^2 = -q/k = 150/2 = 75 => T(x) = T_boundary - (|q| / (2k)) * x * (1 - x)
    x_mesh, _ = np.meshgrid(grid.x_centers, grid.y_centers)
    t_analytical = t_boundary + (q_sink / (2.0 * k_val)) * x_mesh * (1.0 - x_mesh)
    max_disc_err = float(np.max(np.abs(res.temperature - t_analytical)))
    assert max_disc_err < 1.0e-2, (
        f"Discrete FVM solution error vs analytical 1D sink ({max_disc_err} K) exceeds 1e-2 K."
    )

    # 3. Energy conservation with negative source
    eb = res.diagnostics.energy_balance
    assert eb.relative_imbalance < 1.0e-12, (
        f"Energy balance relative imbalance ({eb.relative_imbalance}) exceeds 1e-12."
    )
    expected_q_total = q_sink * (grid.cell_volume * grid.num_cells)  # -150 W
    assert abs(eb.q_generated_total - expected_q_total) < 1.0e-10, (
        f"Total generated heat {eb.q_generated_total} W does not match expected {expected_q_total} W."
    )
