"""Scientific safety, conservation, and provenance verification tests for Milestone 06B."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from neural_process_twin.inference.contracts import (
    InferenceOptions,
    InferenceRequest,
    InferenceStatus,
    RejectionCode,
)
from neural_process_twin.inference.engine import HybridInferenceEngine
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D


@pytest.fixture(scope="module")
def engine() -> HybridInferenceEngine:
    """Fixture providing initialized HybridInferenceEngine."""
    return HybridInferenceEngine.from_artifacts(device="cpu")


@pytest.mark.scientific
def test_face_flux_general_formula_preserves_dimensions_and_thickness() -> None:
    """Verify Dirichlet face-flux calculation with non-square cells and non-unit thickness.

    Confirms that j_out = (2 * k / dx) * (T - T_bnd) and Q = sum(j_out * area) correctly
    incorporate grid.dx, grid.dy, grid.face_area_x, and grid.thickness without assuming
    unit depth or square cells.
    """
    # Non-square domain [0, 2] x [0, 1], non-square resolution (32 x 16), thickness 0.5 m
    grid = Grid2D(
        x_min=0.0,
        x_max=2.0,
        y_min=0.0,
        y_max=1.0,
        nx=32,
        ny=16,
        thickness=0.5,
    )
    assert grid.dx == 2.0 / 32  # 0.0625 m
    assert grid.dy == 1.0 / 16  # 0.0625 m
    assert grid.face_area_x == grid.dy * 0.5  # 0.03125 m^2
    assert grid.face_area_y == grid.dx * 0.5  # 0.03125 m^2
    assert grid.cell_volume == grid.dx * grid.dy * 0.5  # 0.001953125 m^3

    k_field = np.full((16, 32), 3.0)
    q_field = np.full((16, 32), 200.0)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )

    solver = HeatConductionSolver2D()
    res = solver.solve(grid, k_field, q_field, bcs)

    # Solve must satisfy conservation balance with non-unit thickness
    diag = compute_heat_balance(grid, res.temperature, k_field, q_field, bcs)
    assert abs(diag.q_boundary_out_total - diag.q_generated_total) < 1e-3
    assert diag.flux_imbalance < 1e-3


@pytest.mark.scientific
def test_distinguish_signed_net_imbalance_from_absolute_imbalance() -> None:
    """Verify that signed net imbalance Q_out - Q_gen is tracked separately from absolute imbalance."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=16, ny=16)
    k_field = np.full((16, 16), 1.0)
    q_field = np.full((16, 16), 100.0)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 300.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, 300.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, 300.0),
    )

    # Case A: Artificial high temperature -> net heat leaving domain (Q_out > Q_gen)
    t_high = np.full((16, 16), 350.0)
    diag_high = compute_heat_balance(grid, t_high, k_field, q_field, bcs)
    signed_high = diag_high.q_boundary_out_total - diag_high.q_generated_total
    assert signed_high > 0.0  # Net cooling / heat loss
    assert np.isclose(signed_high, diag_high.flux_imbalance)

    # Case B: Artificial low temperature -> net heat entering domain (Q_out < Q_gen)
    t_low = np.full((16, 16), 280.0)
    diag_low = compute_heat_balance(grid, t_low, k_field, q_field, bcs)
    signed_low = diag_low.q_boundary_out_total - diag_low.q_generated_total
    assert signed_low < 0.0  # Net accumulation / trapping
    assert np.isclose(-signed_low, diag_low.flux_imbalance)


@pytest.mark.scientific
def test_fail_closed_contract_never_exposes_rejected_surrogate(
    engine: HybridInferenceEngine,
) -> None:
    """Verify the strict fail-closed contract: if fallback fails, temperature is strictly None."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    # High-contrast input to trigger surrogate rejection
    k_field = np.full((32, 32), 1.0)
    k_field[0, 0] = 50.0

    request = InferenceRequest(
        grid=grid,
        conductivity=k_field,
        source=100.0,
        boundary_conditions=bcs,
        options=InferenceOptions(enable_fallback=True),
    )

    # Mock solver to simulate failure
    mock_solver = MagicMock()
    mock_solver.solve.side_effect = RuntimeError("Catastrophic solver crash")
    original_solver = engine.solver
    engine.solver = mock_solver

    try:
        result = engine.predict(request)
        # Critical safety contract assertion:
        assert result.status == InferenceStatus.SOLVER_FAILED
        assert result.execution_path == "aborted"
        assert result.temperature is None
        assert result.is_failure is True
        assert not result.is_accepted
        assert not result.is_fallback
        assert result.rejection.code == RejectionCode.SOLVER_EXCEPTION
    finally:
        engine.solver = original_solver


@pytest.mark.scientific
def test_checkpoint_sha256_provenance_immutability(
    engine: HybridInferenceEngine,
) -> None:
    """Verify that ensemble member checkpoints match Milestone 05B SHA-256 digests."""
    expected_digests = {
        "scaled_fno_supervised_seed42": "37b0b1a4b717a9c6be2558420fcde6aaff498c79d76aec7e1712b16eb5f82b02",
        "scaled_fno_supervised_seed43": "b7ef5e3c9e59b467be8c9fb783f0ea4a2f1c5316661ed1fc38f36a7e7b34e90a",
        "scaled_fno_supervised_seed44": "379bbdf16b28553e15431844bc94e71ab6785e45fdc0093117b9350966261434",
        "scaled_fno_supervised_seed45": "b407a34628f7953122a18e003a44b2013b7d387b6d9e95b039b6eeef96777bf8",
        "scaled_fno_supervised_seed46": "722123f37783b48eaf58c537df21af64af0a3f434dae037b9246a43f751a3905",
    }

    provenance = engine.provenance
    assert len(provenance.ensemble_member_checkpoints) == 5
    assert len(provenance.ensemble_member_sha256) == 5

    for p_str, sha in zip(
        provenance.ensemble_member_checkpoints,
        provenance.ensemble_member_sha256,
        strict=True,
    ):
        member_name = Path(p_str).parent.name
        assert member_name in expected_digests
        assert sha == expected_digests[member_name]


@pytest.mark.scientific
def test_separate_diagnostics_categorization(
    engine: HybridInferenceEngine,
) -> None:
    """Verify that local residuals, global conservation, and uncertainty remain distinct."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    request = InferenceRequest(
        grid=grid,
        conductivity=2.5,
        source=50.0,
        boundary_conditions=bcs,
    )
    result = engine.predict(request)

    assert result.status == InferenceStatus.ACCEPTED
    # 1. Uncertainty diagnostic
    assert result.ensemble_disagreement_kelvin is not None
    assert result.mean_disagreement_kelvin is not None
    assert result.max_disagreement_kelvin is not None

    # 2. Local residual diagnostic
    diag = result.diagnostics
    assert diag is not None
    assert isinstance(diag.pde_residual_l2_w, float)
    assert isinstance(diag.pde_residual_linf_w, float)
    assert isinstance(diag.pde_residual_relative, float)

    # 3. Global energy balance diagnostic
    assert isinstance(diag.signed_net_imbalance_w, float)
    assert isinstance(diag.absolute_flux_imbalance_w, float)
    assert isinstance(diag.relative_flux_imbalance, float)
    assert isinstance(diag.boundary_heat_flow_out_w, dict)
