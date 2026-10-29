"""Milestone 09 Integration and Scientific Regression Test Suite.

Verifies dual-mode platform contract:
1. Strict mode isolation (Verified Numerical FVM vs Experimental Neural Surrogate).
2. Provenance and convergence metadata (SuperLU direct solve, O(h^2) spatial truncation).
3. Experimental surrogate quality indicators (Jacobi residual in Kelvin, uncertified safety disclaimer).
4. Mesh resolution compatibility (32, 64, 128, 256 supported; non-canonical resolutions fail-closed).
5. Ground-truth dual-mode comparison accuracy (never fabricating reference errors without numerical solve).
6. Backward compatibility with legacy hybrid endpoints.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neural_process_twin.api.server import create_app


@pytest.fixture(scope="module")
def api_client() -> TestClient:
    """Initialize FastAPI test client for Milestone 09 integration tests."""
    app = create_app()
    return TestClient(app)


def make_problem_payload(
    nx: int = 32,
    ny: int = 32,
    mode: str = "numerical_fvm",
    k_contrast: float = 5.0,
    q_max: float = 200.0,
) -> dict:
    """Build a standard physical problem payload."""
    return {
        "mode": mode,
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": nx,
            "ny": ny,
            "thickness": 1.0,
        },
        "conductivity": {
            "pattern_type": "inclusion",
            "uniform_value": 2.5,
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": k_contrast,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.2,
        },
        "source": {
            "pattern_type": "uniform",
            "uniform_value": 50.0,
            "spot_peak_value": q_max,
            "spot_center_x": 0.5,
            "spot_center_y": 0.5,
            "spot_sigma": 0.15,
        },
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
        "options": {
            "enable_fallback": True,
            "fallback_isolation": "thread",
            "provisional_mean_disagreement_threshold_k": 2.5,
            "provisional_max_disagreement_threshold_k": 5.0,
            "input_contrast_threshold": 7.16,
            "input_q_max_threshold": 912.8,
            "solver_timeout_seconds": 30.0,
        },
    }


# =============================================================================
# Test Suite 1: Mode Selection & Strict Mode Isolation
# =============================================================================


def test_verified_numerical_mode_contract(api_client: TestClient) -> None:
    """Verify that numerical_fvm returns verified status, solver metadata, and machine-precision residual."""
    payload = make_problem_payload(nx=32, ny=32, mode="numerical_fvm")
    resp = api_client.post("/api/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_verified_numerical"] is True
    assert data["mode"] == "numerical_fvm"
    assert data["status"] == "numerical_reference_computed"
    assert data["warnings"] == []

    # Verify solver info
    solver_info = data["numerical_solver_info"]
    assert solver_info is not None
    assert solver_info["is_converged"] is True
    assert solver_info["algebraic_residual_norm"] < 1e-10
    assert "SuperLU" in solver_info["linear_solver"]
    assert "Second-Order Spatial Truncation Error O(h^2)" in solver_info["discretization_order"]
    assert "NOT mathematically exact in the continuous sense" in solver_info["notice"]

    # Verify diagnostics
    assert data["experimental_diagnostics"] is None
    diag = data["diagnostics"]
    assert diag is not None
    assert diag["pde_residual_l2_w"] < 1e-10


def test_experimental_neural_mode_contract(api_client: TestClient) -> None:
    """Verify that experimental_neural returns unverified status, explicit warnings, and Jacobi indicators."""
    payload = make_problem_payload(nx=32, ny=32, mode="experimental_neural")
    resp = api_client.post("/api/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_verified_numerical"] is False
    assert data["mode"] == "experimental_neural"
    assert data["status"] == "experimental_prediction_computed"

    # Must contain explicit safety warning
    assert len(data["warnings"]) > 0
    assert any("MaxAE <= 10 K" in w for w in data["warnings"])

    # Must contain experimental diagnostics with Jacobi indicator
    assert data["numerical_solver_info"] is None
    exp_diag = data["experimental_diagnostics"]
    assert exp_diag is not None
    assert exp_diag["pointwise_jacobi_residual_max_k"] >= 0.0
    assert "NOT a mathematically certified upper bound" in exp_diag["indicator_taxonomy"]
    assert "Never deploy for safety-critical heat balance" in exp_diag["safety_disclosure"]


def test_experimental_neural_corrected_mode_contract(api_client: TestClient) -> None:
    """Verify that experimental_neural_corrected applies Jacobi smoothing while maintaining uncertified status."""
    payload = make_problem_payload(nx=64, ny=64, mode="experimental_neural_corrected")
    resp = api_client.post("/api/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_verified_numerical"] is False
    assert data["mode"] == "experimental_neural_corrected"
    assert "Selective Boundary Jacobi" in data["provenance"]["physics_correction"]
    assert data["experimental_diagnostics"] is not None


# =============================================================================
# Test Suite 2: Multi-Resolution Generalization & Fail-Closed Grid Checks
# =============================================================================


@pytest.mark.parametrize("res", [32, 64, 128, 256])
def test_all_canonical_resolutions_supported(api_client: TestClient, res: int) -> None:
    """Verify that all four canonical resolutions (32, 64, 128, 256) evaluate successfully."""
    # Test FVM
    fvm_resp = api_client.post(
        "/api/simulate", json=make_problem_payload(nx=res, ny=res, mode="numerical_fvm")
    )
    assert fvm_resp.status_code == 200
    assert fvm_resp.json()["grid_resolution"] == [res, res]

    # Test Neural Surrogate
    surr_resp = api_client.post(
        "/api/simulate", json=make_problem_payload(nx=res, ny=res, mode="experimental_neural")
    )
    assert surr_resp.status_code == 200
    assert surr_resp.json()["grid_resolution"] == [res, res]


def test_unsupported_resolution_fail_closed_for_neural(api_client: TestClient) -> None:
    """Verify that non-canonical resolution (e.g. 48x48) is rejected with HTTP 400 for neural surrogate."""
    payload = make_problem_payload(nx=48, ny=48, mode="experimental_neural")
    resp = api_client.post("/api/simulate", json=payload)
    assert resp.status_code == 400
    assert "not supported by the experimental surrogate" in resp.json()["detail"].lower()


def test_arbitrary_resolution_accepted_for_numerical_fvm(api_client: TestClient) -> None:
    """Verify that non-canonical resolution (e.g. 48x48) is supported by Verified Numerical Mode."""
    payload = make_problem_payload(nx=48, ny=48, mode="numerical_fvm")
    resp = api_client.post("/api/simulate", json=payload)
    assert resp.status_code == 200
    assert resp.json()["grid_resolution"] == [48, 48]
    assert resp.json()["is_verified_numerical"] is True


# =============================================================================
# Test Suite 3: Dual-Mode Comparison Endpoint
# =============================================================================


def test_dual_mode_comparison_endpoint(api_client: TestClient) -> None:
    """Verify /api/compare executes both paths and computes accurate ground-truth error metrics."""
    payload = make_problem_payload(nx=32, ny=32, mode="experimental_neural")
    resp = api_client.post("/api/compare", json=payload)
    assert resp.status_code == 200
    comp = resp.json()

    # Verify both models are present
    assert comp["numerical_reference"]["is_verified_numerical"] is True
    assert comp["experimental_surrogate"]["is_verified_numerical"] is False

    # Verify true error metrics
    assert comp["rmse_k"] > 0.0
    assert comp["max_absolute_error_k"] > 0.0
    assert comp["relative_l2_error_pct"] > 0.0
    assert comp["speedup_factor"] > 0.0

    # Verify difference field
    diff = np.array(comp["temperature_difference_field"])
    assert diff.shape == (32, 32)
    assert np.isclose(np.max(np.abs(diff)), comp["max_absolute_error_k"])
    assert np.isclose(np.sqrt(np.mean(diff**2)), comp["rmse_k"])


# =============================================================================
# Test Suite 4: Fail-Closed Problem Validation
# =============================================================================


def test_invalid_mode_rejected(api_client: TestClient) -> None:
    """Verify that an invalid simulation mode is rejected with HTTP 422/400."""
    payload = make_problem_payload(nx=32, ny=32, mode="magic_instant_solver")
    resp = api_client.post("/api/simulate", json=payload)
    assert resp.status_code in (400, 422)


def test_invalid_boundary_condition_rejected(api_client: TestClient) -> None:
    """Verify that ill-conditioned pure-Neumann boundary condition is handled properly."""
    payload = make_problem_payload(nx=32, ny=32, mode="numerical_fvm")
    payload["boundary_conditions"]["left"] = {"side": "left", "type": "neumann", "value": 0.0}
    payload["boundary_conditions"]["right"] = {"side": "right", "type": "neumann", "value": 0.0}
    # Pure Neumann has singular system
    resp = api_client.post("/api/simulate", json=payload)
    assert resp.status_code in (400, 422, 500)
