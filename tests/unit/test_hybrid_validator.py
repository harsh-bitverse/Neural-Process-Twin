"""Unit tests for InputValidator physical and dimensional checks."""

from __future__ import annotations

import numpy as np
import pytest

from neural_process_twin.inference.contracts import (
    InferenceOptions,
    InferenceRequest,
    RejectionCode,
)
from neural_process_twin.inference.validator import InputValidator
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D


@pytest.fixture
def base_request() -> InferenceRequest:
    """Fixture providing a standard valid 32x32 request."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    return InferenceRequest(
        grid=grid,
        conductivity=2.0,
        source=100.0,
        boundary_conditions=bcs,
    )


@pytest.mark.unit
def test_validator_valid_input(base_request: InferenceRequest) -> None:
    """Test successful validation of a well-posed in-domain request."""
    validator = InputValidator()
    res = validator.validate(base_request)
    assert res.is_valid is True
    assert res.surrogate_compatible is True
    assert res.rejection.code == RejectionCode.NONE
    assert res.evaluated_k is not None
    assert res.evaluated_q is not None
    assert res.evaluated_k.shape == (32, 32)
    assert res.evaluated_q.shape == (32, 32)


@pytest.mark.unit
def test_validator_non_positive_conductivity(base_request: InferenceRequest) -> None:
    """Test rejection when conductivity contains zero or negative values."""
    validator = InputValidator()
    # 1. Zero conductivity
    req_zero = InferenceRequest(
        grid=base_request.grid,
        conductivity=0.0,
        source=100.0,
        boundary_conditions=base_request.boundary_conditions,
    )
    res = validator.validate(req_zero)
    assert res.is_valid is False
    assert res.rejection.code == RejectionCode.NON_POSITIVE_CONDUCTIVITY

    # 2. Negative cell in 2D array
    k_arr = np.full((32, 32), 2.0)
    k_arr[15, 15] = -0.5
    req_neg = InferenceRequest(
        grid=base_request.grid,
        conductivity=k_arr,
        source=100.0,
        boundary_conditions=base_request.boundary_conditions,
    )
    res = validator.validate(req_neg)
    assert res.is_valid is False
    assert res.rejection.code == RejectionCode.NON_POSITIVE_CONDUCTIVITY
    assert res.rejection.observed_value == -0.5


@pytest.mark.unit
def test_validator_non_finite_values(base_request: InferenceRequest) -> None:
    """Test rejection when NaN or Inf is present in input fields or BCs."""
    validator = InputValidator()

    # 1. NaN in source field
    q_arr = np.full((32, 32), 50.0)
    q_arr[5, 5] = np.nan
    req_nan_q = InferenceRequest(
        grid=base_request.grid,
        conductivity=2.0,
        source=q_arr,
        boundary_conditions=base_request.boundary_conditions,
    )
    res = validator.validate(req_nan_q)
    assert res.is_valid is False
    assert res.rejection.code == RejectionCode.NON_FINITE_VALUES

    # 2. Inf in boundary temperature
    bcs_inf = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, np.inf),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    req_inf_bc = InferenceRequest(
        grid=base_request.grid,
        conductivity=2.0,
        source=50.0,
        boundary_conditions=bcs_inf,
    )
    res = validator.validate(req_inf_bc)
    assert res.is_valid is False
    assert res.rejection.code == RejectionCode.NON_FINITE_VALUES


@pytest.mark.unit
def test_validator_pure_neumann_without_gauge() -> None:
    """Test rejection when all boundaries are Neumann and no gauge pin is provided."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    bcs_neumann = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.NEUMANN, 0.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, 0.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    validator = InputValidator()

    # 1. Without gauge pin: singular -> invalid
    req_no_gauge = InferenceRequest(
        grid=grid,
        conductivity=1.0,
        source=0.0,
        boundary_conditions=bcs_neumann,
        gauge_pin=None,
    )
    res = validator.validate(req_no_gauge)
    assert res.is_valid is False
    assert res.rejection.code == RejectionCode.PURE_NEUMANN_WITHOUT_GAUGE

    # 2. With gauge pin: well-posed -> valid
    req_gauge = InferenceRequest(
        grid=grid,
        conductivity=1.0,
        source=0.0,
        boundary_conditions=bcs_neumann,
        gauge_pin=(0, 0),
        gauge_temperature=300.0,
    )
    res = validator.validate(req_gauge)
    assert res.is_valid is True


@pytest.mark.unit
def test_validator_grid_compatibility_and_fallback_option() -> None:
    """Test behavior with non-surrogate grid dimensions (e.g. 16x16)."""
    grid_16 = Grid2D(0.0, 1.0, 0.0, 1.0, nx=16, ny=16)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )
    validator = InputValidator()

    # 1. Fallback enabled: physically valid for solver, but surrogate incompatible
    req_fb_on = InferenceRequest(
        grid=grid_16,
        conductivity=2.0,
        source=0.0,
        boundary_conditions=bcs,
        options=InferenceOptions(enable_fallback=True),
    )
    res = validator.validate(req_fb_on)
    assert res.is_valid is True
    assert res.surrogate_compatible is False
    assert res.rejection.code == RejectionCode.INCOMPATIBLE_GRID_DIMENSIONS

    # 2. Fallback disabled: fatal rejection
    req_fb_off = InferenceRequest(
        grid=grid_16,
        conductivity=2.0,
        source=0.0,
        boundary_conditions=bcs,
        options=InferenceOptions(enable_fallback=False),
    )
    res = validator.validate(req_fb_off)
    assert res.is_valid is False
    assert res.surrogate_compatible is False
    assert res.rejection.code == RejectionCode.INCOMPATIBLE_GRID_DIMENSIONS
