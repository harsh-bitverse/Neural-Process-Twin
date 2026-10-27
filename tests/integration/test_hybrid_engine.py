"""Integration tests for the Hybrid Digital Twin inference engine."""

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
from neural_process_twin.solvers.grid import Grid2D


@pytest.fixture(scope="module")
def engine() -> HybridInferenceEngine:
    """Fixture providing initialized HybridInferenceEngine from repository artifacts."""
    return HybridInferenceEngine.from_artifacts(
        checkpoint_paths=[
            Path(f"artifacts/checkpoints/scaled_fno_supervised_seed{s}/best_model.pt")
            for s in (42, 43, 44, 45, 46)
        ],
        calibration_results_path="artifacts/reports/milestone_05b_calibration_results.json",
        reference_dataset_path="artifacts/data/scaled_dataset.h5",
        device="cpu",
    )


@pytest.fixture
def iid_bcs() -> BoundaryConditions:
    """Standard mixed Dirichlet/Neumann boundary conditions."""
    return BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )


@pytest.mark.integration
def test_hybrid_engine_in_distribution_accepted(
    engine: HybridInferenceEngine, iid_bcs: BoundaryConditions
) -> None:
    """Test that an in-distribution sample with low uncertainty is accepted on the fast path."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    # Smooth constant conductivity in training range [0.5, 5.0]
    k_field = np.full((32, 32), 2.5)
    q_field = np.full((32, 32), 50.0)

    request = InferenceRequest(
        grid=grid,
        conductivity=k_field,
        source=q_field,
        boundary_conditions=iid_bcs,
    )

    result = engine.predict(request)

    assert result.status == InferenceStatus.ACCEPTED
    assert result.execution_path == "surrogate_fast_path"
    assert result.rejection.code == RejectionCode.NONE
    assert result.temperature is not None
    assert result.temperature.shape == (32, 32)
    assert np.all(result.temperature > 0.0)

    # Uncertainty fields
    assert result.ensemble_disagreement_kelvin is not None
    assert result.mean_disagreement_kelvin is not None
    assert result.mean_disagreement_kelvin < 0.9978  # Below screening threshold

    # Conformal intervals
    assert result.conformal_intervals is not None
    c_int = result.conformal_intervals
    assert c_int.pointwise_lower_kelvin.shape == (32, 32)
    assert np.all(c_int.pointwise_lower_kelvin <= result.temperature)
    assert np.all(result.temperature <= c_int.pointwise_upper_kelvin)

    # Physical diagnostics
    assert result.diagnostics is not None
    assert result.diagnostics.pde_residual_l2_w > 0.0
    assert (
        abs(result.diagnostics.signed_net_imbalance_w)
        == result.diagnostics.absolute_flux_imbalance_w
    )

    # Latencies
    assert result.latency.total_pipeline_ms > 0.0
    assert result.latency.solver_fallback_ms == 0.0  # Solver was not invoked


@pytest.mark.integration
def test_hybrid_engine_high_contrast_fallback_succeeded(
    engine: HybridInferenceEngine, iid_bcs: BoundaryConditions
) -> None:
    """Test that high conductivity contrast triggers fallback to numerical solver."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    # High contrast composite: 30:1 > 7.16:1 threshold
    k_field = np.full((32, 32), 1.0)
    k_field[10:22, 10:22] = 30.0
    q_field = np.full((32, 32), 50.0)

    request = InferenceRequest(
        grid=grid,
        conductivity=k_field,
        source=q_field,
        boundary_conditions=iid_bcs,
        options=InferenceOptions(enable_fallback=True),
    )

    result = engine.predict(request)

    assert result.status == InferenceStatus.FALLBACK_SUCCEEDED
    assert result.execution_path == "solver_fallback_path"
    assert result.rejection.code == RejectionCode.HIGH_INPUT_CONTRAST
    assert result.temperature is not None
    assert result.temperature.shape == (32, 32)

    # Conformal intervals are not attached to deterministic numerical solve
    assert result.conformal_intervals is None

    # Solver diagnostics should be near zero (machine precision)
    assert result.diagnostics is not None
    assert result.diagnostics.pde_residual_relative < 1e-6
    assert result.diagnostics.absolute_flux_imbalance_w < 1e-2

    # Solver latency must be recorded
    assert result.latency.solver_fallback_ms > 0.0


@pytest.mark.integration
def test_hybrid_engine_high_contrast_fallback_disabled_fails_closed(
    engine: HybridInferenceEngine, iid_bcs: BoundaryConditions
) -> None:
    """Test fail-closed behavior when fallback is disabled on rejected input."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    k_field = np.full((32, 32), 1.0)
    k_field[10:22, 10:22] = 30.0  # High contrast
    q_field = np.full((32, 32), 50.0)

    request = InferenceRequest(
        grid=grid,
        conductivity=k_field,
        source=q_field,
        boundary_conditions=iid_bcs,
        options=InferenceOptions(enable_fallback=False),
    )

    result = engine.predict(request)

    # Must fail closed: temperature field is None!
    assert result.status == InferenceStatus.FALLBACK_REQUIRED
    assert result.execution_path == "aborted"
    assert result.temperature is None
    assert result.rejection.code == RejectionCode.FALLBACK_DISABLED


@pytest.mark.integration
def test_hybrid_engine_incompatible_grid_fallback_succeeded(
    engine: HybridInferenceEngine, iid_bcs: BoundaryConditions
) -> None:
    """Test that a non-surrogate grid (e.g. 16x16) routes directly to numerical solver."""
    grid_16 = Grid2D(0.0, 1.0, 0.0, 1.0, nx=16, ny=16)
    request = InferenceRequest(
        grid=grid_16,
        conductivity=2.0,
        source=50.0,
        boundary_conditions=iid_bcs,
        options=InferenceOptions(enable_fallback=True),
    )

    result = engine.predict(request)

    assert result.status == InferenceStatus.FALLBACK_SUCCEEDED
    assert result.execution_path == "solver_fallback_path"
    assert result.rejection.code == RejectionCode.INCOMPATIBLE_GRID_DIMENSIONS
    assert result.temperature is not None
    assert result.temperature.shape == (16, 16)
    assert result.latency.surrogate_inference_ms == 0.0  # Surrogate bypassed


@pytest.mark.integration
def test_hybrid_engine_invalid_input_aborts(
    engine: HybridInferenceEngine, iid_bcs: BoundaryConditions
) -> None:
    """Test that invalid physical input (k <= 0) aborts immediately without running solver."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    request = InferenceRequest(
        grid=grid,
        conductivity=-1.0,  # Physically invalid
        source=50.0,
        boundary_conditions=iid_bcs,
    )

    result = engine.predict(request)

    assert result.status == InferenceStatus.INVALID_INPUT
    assert result.execution_path == "aborted"
    assert result.temperature is None
    assert result.rejection.code == RejectionCode.NON_POSITIVE_CONDUCTIVITY


@pytest.mark.integration
def test_hybrid_engine_solver_failure_fails_closed(
    engine: HybridInferenceEngine, iid_bcs: BoundaryConditions
) -> None:
    """Test that if the numerical fallback solver fails, the engine fails closed."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    # Input that will trigger fallback (high contrast)
    k_field = np.full((32, 32), 1.0)
    k_field[0, 0] = 50.0

    request = InferenceRequest(
        grid=grid,
        conductivity=k_field,
        source=50.0,
        boundary_conditions=iid_bcs,
        options=InferenceOptions(enable_fallback=True),
    )

    # Mock solver to raise an exception
    mock_solver = MagicMock()
    mock_solver.solve.side_effect = RuntimeError(
        "Mocked SuperLU sparse solver factorization failure"
    )
    original_solver = engine.solver
    engine.solver = mock_solver

    try:
        result = engine.predict(request)
        assert result.status == InferenceStatus.SOLVER_FAILED
        assert result.execution_path == "aborted"
        assert result.temperature is None  # Never expose rejected surrogate prediction!
        assert result.rejection.code == RejectionCode.SOLVER_EXCEPTION
    finally:
        engine.solver = original_solver
