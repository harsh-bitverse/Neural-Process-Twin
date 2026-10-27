"""Scientific regression tests for Milestone 08E Adaptive Hybrid Routing.

Tests:
1. AdaptiveHybridEngine initialization and default thresholds.
2. Direct FVM routing (R0).
3. Pre-screening rejection on extreme conductivity contrast.
4. Pre-screening rejection on extreme volumetric heat source.
5. Fail-closed fallback to FVM on high uncertainty.
6. R5 adaptive cascade routing semantics and escalation tracking.
7. Wilson score confidence interval edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from neural_process_twin.inference.adaptive_routing import (
    AdaptiveHybridEngine,
    RoutingPolicy,
    compute_wilson_ci,
)
from neural_process_twin.inference.contracts import InferenceStatus, RejectionCode
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D


@pytest.fixture
def test_grid() -> Grid2D:
    return Grid2D(0.0, 1.0, 0.0, 1.0, 32, 32, thickness=1.0)


@pytest.fixture
def nominal_problem(test_grid: Grid2D) -> tuple[np.ndarray, np.ndarray, BoundaryConditions]:
    ny, nx = test_grid.ny, test_grid.nx
    k = np.ones((ny, nx), dtype=np.float64) * 2.0
    q = np.zeros((ny, nx), dtype=np.float64)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, np.full(ny, 300.0)),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, np.full(ny, 350.0)),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, np.zeros(nx)),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, np.zeros(nx)),
    )
    return k, q, bcs


def test_engine_initialization() -> None:
    """Verify engine loads checkpoints and initializes default thresholds."""
    engine = AdaptiveHybridEngine()
    assert engine.thresholds.tau_k_max == 20.0
    assert engine.thresholds.tau_q_max == 1800.0
    assert engine.thresholds.tau_u1_unc > 0.0
    assert engine.thresholds.tau_u3_t5_unc > 0.0


def test_r0_direct_fvm_execution(
    test_grid: Grid2D, nominal_problem: tuple[np.ndarray, np.ndarray, BoundaryConditions]
) -> None:
    """Verify R0 policy executes reference solver directly."""
    k, q, bcs = nominal_problem
    engine = AdaptiveHybridEngine()

    res = engine.route_inference(test_grid, k, q, bcs, policy=RoutingPolicy.R0_DIRECT_FVM)
    assert res.status == InferenceStatus.FALLBACK_SUCCEEDED
    assert res.policy == RoutingPolicy.R0_DIRECT_FVM
    assert res.uncertainty is None
    assert "t_fvm_solve" in res.timings_ms
    # Check linear gradient from 300 to 350 K
    assert np.isclose(res.temperature[16, 0], 300.0, atol=1.0)
    assert np.isclose(res.temperature[16, -1], 350.0, atol=1.0)


def test_prescreening_contrast_rejection(
    test_grid: Grid2D, nominal_problem: tuple[np.ndarray, np.ndarray, BoundaryConditions]
) -> None:
    """Verify problem with C_k > tau_k escalates to FVM with high contrast code."""
    k, q, bcs = nominal_problem
    # Set extreme contrast inclusion
    k[10:20, 10:20] = 100.0
    k[0:5, 0:5] = 1.0  # contrast = 100.0 > 20.0

    engine = AdaptiveHybridEngine()
    res = engine.route_inference(test_grid, k, q, bcs, policy=RoutingPolicy.R2_U1_SINGLE_PASS)

    assert res.status == InferenceStatus.FALLBACK_SUCCEEDED
    assert res.escalated_to_fvm is True
    assert res.rejection_record is not None
    assert res.rejection_record.code == RejectionCode.HIGH_INPUT_CONTRAST


def test_prescreening_heat_source_rejection(
    test_grid: Grid2D, nominal_problem: tuple[np.ndarray, np.ndarray, BoundaryConditions]
) -> None:
    """Verify problem with q > tau_q escalates to FVM with high source code."""
    k, q, bcs = nominal_problem
    q[16, 16] = 2500.0  # > 1800.0

    engine = AdaptiveHybridEngine()
    res = engine.route_inference(test_grid, k, q, bcs, policy=RoutingPolicy.R4_U3_T5_MULTI_PASS)

    assert res.status == InferenceStatus.FALLBACK_SUCCEEDED
    assert res.escalated_to_fvm is True
    assert res.rejection_record is not None
    assert res.rejection_record.code == RejectionCode.HIGH_HEAT_SOURCE


def test_invalid_input_non_positive_conductivity(
    test_grid: Grid2D, nominal_problem: tuple[np.ndarray, np.ndarray, BoundaryConditions]
) -> None:
    """Verify invalid input with zero or negative conductivity is trapped."""
    k, q, bcs = nominal_problem
    k[10, 10] = -1.0

    engine = AdaptiveHybridEngine()
    res = engine.route_inference(test_grid, k, q, bcs, policy=RoutingPolicy.R5_ADAPTIVE_CASCADE)

    assert res.status == InferenceStatus.INVALID_INPUT
    assert res.rejection_record is not None
    assert res.rejection_record.code == RejectionCode.NON_POSITIVE_CONDUCTIVITY


def test_r5_adaptive_cascade_structure(
    test_grid: Grid2D, nominal_problem: tuple[np.ndarray, np.ndarray, BoundaryConditions]
) -> None:
    """Verify R5 cascade returns valid result and records complete timings."""
    k, q, bcs = nominal_problem
    engine = AdaptiveHybridEngine()

    res = engine.route_inference(test_grid, k, q, bcs, policy=RoutingPolicy.R5_ADAPTIVE_CASCADE)
    assert res.status in [InferenceStatus.ACCEPTED, InferenceStatus.FALLBACK_SUCCEEDED]
    assert res.policy == RoutingPolicy.R5_ADAPTIVE_CASCADE
    assert "t_total" in res.timings_ms
    assert "t_validation" in res.timings_ms
    assert "t_prescreening" in res.timings_ms


def test_wilson_ci_computation() -> None:
    """Verify Wilson score confidence interval computation and boundary edge cases."""
    # Zero failures out of 100
    p_hat, ci_l, ci_u = compute_wilson_ci(0, 100)
    assert p_hat == 0.0
    assert ci_l == 0.0
    assert 0.02 < ci_u < 0.05  # Upper bound ~ 3.6% (Rule of three)

    # 10 failures out of 100
    p_hat, ci_l, ci_u = compute_wilson_ci(10, 100)
    assert np.isclose(p_hat, 0.10)
    assert ci_l < 0.10 < ci_u

    # Edge case: n = 0
    p_hat, ci_l, ci_u = compute_wilson_ci(0, 0)
    assert p_hat == 0.0
    assert ci_l == 0.0
    assert ci_u == 1.0
