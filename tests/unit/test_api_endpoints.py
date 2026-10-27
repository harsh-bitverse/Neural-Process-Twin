"""Unit and integration tests for the Neural Process Twin FastAPI service."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neural_process_twin.api.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing a FastAPI TestClient."""
    app = create_app(static_dir="non_existent_dist_for_test")
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """Verify health endpoint returns 200 with operational environment details."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ensemble_loaded"] is True
    assert data["ensemble_member_count"] == 5


def test_capabilities_endpoint(client: TestClient) -> None:
    """Verify capabilities disclosure returns accurate units, thresholds, and performance disclosure."""
    response = client.get("/api/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert data["surrogate_supported_resolution"] == [32, 32]
    assert len(data["ensemble_members"]) == 5
    assert data["training_domain_limits"]["max_conductivity_contrast_ratio"] == 7.16
    assert (
        data["disagreement_screening_thresholds"]["spatial_mean_disagreement_tau_mean_k"] == 0.9978
    )
    assert "SCIENTIFIC PERFORMANCE REALITY" in data["performance_disclosure"]
    assert "11-13x faster" in data["performance_disclosure"]


def test_validate_endpoint_valid_problem(client: TestClient) -> None:
    """Verify problem validation passes on in-distribution specification."""
    payload = {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 32,
            "ny": 32,
            "thickness": 1.0,
        },
        "conductivity": {"pattern_type": "uniform", "uniform_value": 2.5},
        "source": {"pattern_type": "uniform", "uniform_value": 50.0},
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
    }
    response = client.post("/api/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["surrogate_compatible"] is True
    assert len(data["errors"]) == 0


def test_validate_endpoint_high_contrast_warning(client: TestClient) -> None:
    """Verify problem validation flags high contrast warning while remaining valid."""
    payload = {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 32,
            "ny": 32,
            "thickness": 1.0,
        },
        "conductivity": {
            "pattern_type": "inclusion",
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": 30.0,  # 30:1 contrast > 7.16
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.2,
        },
        "source": {"pattern_type": "uniform", "uniform_value": 50.0},
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
    }
    response = client.post("/api/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is True
    assert data["surrogate_compatible"] is False
    assert any("exceeds training envelope" in w for w in data["warnings"])


def test_hybrid_inference_accepted_path(client: TestClient) -> None:
    """Verify hybrid inference executes surrogate fast path on nominal problem."""
    payload = {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 32,
            "ny": 32,
            "thickness": 1.0,
        },
        "conductivity": {"pattern_type": "uniform", "uniform_value": 2.5},
        "source": {"pattern_type": "uniform", "uniform_value": 50.0},
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
    }
    response = client.post("/api/infer/hybrid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["execution_path"] == "surrogate_fast_path"
    assert data["temperature"] is not None
    assert len(data["temperature"]) == 32
    assert len(data["temperature"][0]) == 32
    assert data["temperature_min_k"] >= 300.0
    assert data["ensemble_disagreement_k"] is not None
    assert data["diagnostics"] is not None
    assert data["diagnostics"]["pde_residual_l2_w"] >= 0.0
    assert data["conformal_intervals"] is not None


def test_hybrid_inference_fallback_path(client: TestClient) -> None:
    """Verify hybrid inference executes FVM fallback on out-of-distribution contrast."""
    payload = {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 32,
            "ny": 32,
            "thickness": 1.0,
        },
        "conductivity": {
            "pattern_type": "inclusion",
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": 30.0,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.2,
        },
        "source": {"pattern_type": "uniform", "uniform_value": 50.0},
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
        "options": {"enable_fallback": True},
    }
    response = client.post("/api/infer/hybrid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "fallback_succeeded"
    assert data["execution_path"] == "solver_fallback_path"
    assert data["temperature"] is not None
    assert data["rejection"]["code"] == "high_input_contrast"


def test_hybrid_inference_fallback_disabled_fails_closed(client: TestClient) -> None:
    """Verify hybrid inference fails closed when fallback is disabled on rejected input."""
    payload = {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 32,
            "ny": 32,
            "thickness": 1.0,
        },
        "conductivity": {
            "pattern_type": "inclusion",
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": 30.0,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.2,
        },
        "source": {"pattern_type": "uniform", "uniform_value": 50.0},
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
        "options": {"enable_fallback": False},
    }
    response = client.post("/api/infer/hybrid", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "fallback_required"
    assert data["temperature"] is None  # Strict fail-closed guarantee


def test_direct_fvm_solve(client: TestClient) -> None:
    """Verify direct FVM endpoint executes reference solver and reports zero fabricated uncertainty."""
    payload = {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 32,
            "ny": 32,
            "thickness": 1.0,
        },
        "conductivity": {"pattern_type": "uniform", "uniform_value": 2.5},
        "source": {"pattern_type": "uniform", "uniform_value": 50.0},
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
    }
    response = client.post("/api/infer/fvm", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "fallback_succeeded"
    assert data["execution_path"] == "direct_fvm_solve"
    assert data["temperature"] is not None
    assert data["ensemble_disagreement_k"] is None  # No fabricated uncertainty
    assert data["conformal_intervals"] is None
    assert data["diagnostics"]["pde_residual_relative"] <= 1e-8


def test_compare_endpoint(client: TestClient) -> None:
    """Verify research comparison endpoint computes difference field, RMSE, and latency ratio."""
    payload = {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 32,
            "ny": 32,
            "thickness": 1.0,
        },
        "conductivity": {"pattern_type": "uniform", "uniform_value": 2.5},
        "source": {"pattern_type": "uniform", "uniform_value": 50.0},
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
    }
    response = client.post("/api/infer/compare", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "hybrid_inference" in data
    assert "fvm_reference" in data
    assert data["temperature_difference_field"] is not None
    assert data["rmse_k"] is not None
    assert data["rmse_k"] >= 0.0
    assert data["max_absolute_error_k"] >= 0.0
    assert data["fvm_speedup_factor"] > 1.0  # Surrogate takes longer than FVM on CPU
    assert "surrogate_acceptance_status" in data


def test_frontend_static_serving() -> None:
    """Verify create_app mounts frontend/dist when present and serves index.html at root."""
    from pathlib import Path

    dist_dir = Path("frontend/dist")
    if dist_dir.is_dir():
        app = create_app(static_dir=dist_dir)
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert '<div id="root">' in response.text
