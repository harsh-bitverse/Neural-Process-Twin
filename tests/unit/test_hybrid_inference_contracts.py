"""Unit tests for Hybrid Digital Twin contracts and serialization."""

from __future__ import annotations

import numpy as np
import pytest

from neural_process_twin.inference.contracts import (
    ExecutionLatency,
    HybridInferenceResult,
    InferenceProvenance,
    InferenceRequest,
    InferenceStatus,
    RejectionCode,
    RejectionRecord,
)
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D


@pytest.mark.unit
def test_inference_status_enums() -> None:
    """Verify inference status codes and terminal representations."""
    assert InferenceStatus.ACCEPTED == "accepted"
    assert InferenceStatus.FALLBACK_SUCCEEDED == "fallback_succeeded"
    assert InferenceStatus.INVALID_INPUT == "invalid_input"
    assert InferenceStatus.SOLVER_FAILED == "solver_failed"
    assert InferenceStatus.FALLBACK_REQUIRED == "fallback_required"


@pytest.mark.unit
def test_rejection_code_enums() -> None:
    """Verify machine-readable rejection codes."""
    assert RejectionCode.NONE == "none"
    assert RejectionCode.NON_POSITIVE_CONDUCTIVITY == "non_positive_conductivity"
    assert RejectionCode.NON_FINITE_VALUES == "non_finite_values"
    assert RejectionCode.INCOMPATIBLE_GRID_DIMENSIONS == "incompatible_grid_dimensions"
    assert RejectionCode.HIGH_ENSEMBLE_DISAGREEMENT == "high_ensemble_disagreement"
    assert RejectionCode.SOLVER_DIVERGENCE == "solver_divergence"
    assert RejectionCode.SOLVER_TIMEOUT == "solver_timeout"


@pytest.mark.unit
def test_fail_closed_contract_property() -> None:
    """Verify fail-closed properties on HybridInferenceResult."""
    provenance = InferenceProvenance(
        ensemble_member_checkpoints=["m1", "m2"],
        ensemble_member_sha256=["h1", "h2"],
        normalizer_method="standard",
        temperature_mean_kelvin=300.0,
        temperature_std_kelvin=20.0,
        calibration_dataset_id="test_cal",
        calibration_confidence_level=0.95,
        solver_name="HeatConductionSolver2D",
        solver_tolerance=1e-10,
    )
    latency = ExecutionLatency(0.1, 10.0, 2.0, 0.0, 12.1)

    # 1. Accepted prediction
    res_accepted = HybridInferenceResult(
        temperature=np.full((32, 32), 300.0),
        status=InferenceStatus.ACCEPTED,
        execution_path="surrogate_fast_path",
        rejection=RejectionRecord(code=RejectionCode.NONE, message="Passed"),
        ensemble_disagreement_kelvin=np.full((32, 32), 0.5),
        mean_disagreement_kelvin=0.5,
        max_disagreement_kelvin=0.8,
        conformal_intervals=None,
        diagnostics=None,
        latency=latency,
        provenance=provenance,
    )
    assert res_accepted.is_accepted
    assert not res_accepted.is_fallback
    assert not res_accepted.is_failure
    assert res_accepted.temperature is not None

    # 2. Failed solver: temperature must be None
    res_failed = HybridInferenceResult(
        temperature=None,
        status=InferenceStatus.SOLVER_FAILED,
        execution_path="aborted",
        rejection=RejectionRecord(code=RejectionCode.SOLVER_DIVERGENCE, message="Diverged"),
        ensemble_disagreement_kelvin=None,
        mean_disagreement_kelvin=None,
        max_disagreement_kelvin=None,
        conformal_intervals=None,
        diagnostics=None,
        latency=latency,
        provenance=provenance,
    )
    assert not res_failed.is_accepted
    assert not res_failed.is_fallback
    assert res_failed.is_failure
    assert res_failed.temperature is None


@pytest.mark.unit
def test_inference_request_defaults() -> None:
    """Verify InferenceRequest and default options."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    req = InferenceRequest(grid=grid, conductivity=2.0, source=0.0, boundary_conditions=bcs)
    assert req.options.enable_fallback is True
    assert req.options.provisional_mean_disagreement_threshold_k == 0.9978
    assert req.options.provisional_max_disagreement_threshold_k == 4.4526
    assert req.options.pde_residual_threshold_w is None
    assert req.options.boundary_flux_imbalance_threshold_w is None
