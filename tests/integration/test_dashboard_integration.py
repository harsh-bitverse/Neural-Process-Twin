"""Integration tests for the Milestone 07 Dashboard and API service.

Verifies end-to-end workflow interactions: health, capabilities, validation,
hybrid fast-path inference, out-of-envelope fallback, and dual-solver comparison.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neural_process_twin.api.app import create_app


@pytest.fixture(scope="module")
def dashboard_client() -> TestClient:
    """Fixture providing TestClient with static frontend mounted if dist exists."""
    dist_dir = Path("frontend/dist")
    app = create_app(static_dir=dist_dir if dist_dir.is_dir() else None)
    return TestClient(app)


def test_dashboard_full_lifecycle(dashboard_client: TestClient) -> None:
    """Test full interactive user journey through the dashboard API."""
    # 1. Health inspection
    health_resp = dashboard_client.get("/api/health")
    assert health_resp.status_code == 200
    health = health_resp.json()
    assert health["status"] == "healthy"
    assert health["ensemble_loaded"] is True

    # 2. Capabilities lookup
    cap_resp = dashboard_client.get("/api/capabilities")
    assert cap_resp.status_code == 200
    caps = cap_resp.json()
    assert caps["surrogate_supported_resolution"] == [32, 32]
    assert len(caps["ensemble_members"]) == 5

    # 3. Interactive problem specification (Nominal composite preset)
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
            "uniform_value": 2.5,
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": 5.0,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.2,
        },
        "source": {
            "pattern_type": "uniform",
            "uniform_value": 50.0,
            "spot_peak_value": 200.0,
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
            "solver_timeout_seconds": 15.0,
        },
    }

    # 4. Problem pre-validation
    val_resp = dashboard_client.post("/api/validate", json=payload)
    assert val_resp.status_code == 200
    val = val_resp.json()
    assert val["is_valid"] is True
    assert val["within_contrast_envelope"] is True

    # 5. Hybrid inference execution
    infer_resp = dashboard_client.post("/api/infer/hybrid", json=payload)
    assert infer_resp.status_code == 200
    res = infer_resp.json()
    assert res["status"] in ("accepted", "fallback_succeeded")
    assert res["temperature"] is not None
    assert len(res["temperature"]) == 32
    assert len(res["temperature"][0]) == 32
    assert res["diagnostics"] is not None
    assert "signed_net_imbalance_w" in res["diagnostics"]

    # 6. Comparison mode execution
    comp_resp = dashboard_client.post("/api/infer/compare", json=payload)
    assert comp_resp.status_code == 200
    comp = comp_resp.json()
    assert "hybrid_inference" in comp
    assert "fvm_reference" in comp
    assert comp["temperature_difference_field"] is not None
    assert comp["fvm_speedup_factor"] > 1.0
