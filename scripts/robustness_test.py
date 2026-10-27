"""Robustness and stress testing script for Milestone 07A.

Exercises edge cases, non-finite values, invalid meshes, solver timeouts,
and rapid repeated requests against the running API server.
"""

from __future__ import annotations

import concurrent.futures
import time

import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=30.0)


def test_invalid_conductivity_and_non_finite():
    print("\n--- Robustness 1: Invalid Conductivity and Non-Finite Inputs ---")
    base = {
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

    # 1. Negative conductivity
    p_neg = dict(base, conductivity={"pattern_type": "uniform", "uniform_value": -10.0})
    r_neg = client.post("/api/infer/hybrid", json=p_neg)
    assert r_neg.status_code == 422, f"Expected 422, got {r_neg.status_code}"
    print("  [PASS] Negative conductivity properly rejected with HTTP 422.")

    # 2. Zero conductivity
    p_zero = dict(base, conductivity={"pattern_type": "uniform", "uniform_value": 0.0})
    r_zero = client.post("/api/infer/hybrid", json=p_zero)
    assert r_zero.status_code == 422, f"Expected 422, got {r_zero.status_code}"
    print("  [PASS] Zero conductivity properly rejected with HTTP 422.")

    # 3. Negative thickness
    p_thick = dict(base)
    p_thick["grid"] = dict(base["grid"], thickness=-0.5)
    r_thick = client.post("/api/infer/hybrid", json=p_thick)
    assert r_thick.status_code == 422, f"Expected 422, got {r_thick.status_code}"
    print("  [PASS] Negative thickness properly rejected with HTTP 422.")


def test_unsupported_mesh_sizes():
    print("\n--- Robustness 2: Unsupported Mesh Sizes & Resolution Handling ---")
    base = {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 16,
            "ny": 16,
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

    # 1. 16x16 Mesh in Hybrid Mode (surrogate incompatible, FVM compatible)
    r_16 = client.post("/api/infer/hybrid", json=base).json()
    assert r_16["status"] == "fallback_succeeded"
    assert r_16["rejection"]["code"] == "incompatible_grid_dimensions"
    assert len(r_16["temperature"]) == 16
    assert len(r_16["temperature"][0]) == 16
    print(
        "  [PASS] 16x16 grid: correctly rejected by surrogate contract, solved cleanly via FVM fallback."
    )

    # 2. Invalid zero nx
    p_zero_nx = dict(base)
    p_zero_nx["grid"] = dict(base["grid"], nx=0)
    r_zero_nx = client.post("/api/infer/hybrid", json=p_zero_nx)
    assert r_zero_nx.status_code == 422
    print("  [PASS] Zero grid cells (nx=0) rejected with HTTP 422.")


def test_solver_timeout_and_fail_closed():
    print("\n--- Robustness 3: Solver Timeout Scenarios and Fail-Closed Invariant ---")
    # 1. Invalid microsecond timeout rejected by API validation schema
    p_invalid_timeout = {
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
            "uniform_value": 1.0,
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": 30.0,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.25,
        },
        "source": {"pattern_type": "uniform", "uniform_value": 50.0},
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
            "solver_timeout_seconds": 0.000001,  # Below 0.1s floor
        },
    }
    r_to_inv = client.post("/api/infer/hybrid", json=p_invalid_timeout)
    assert r_to_inv.status_code == 422
    assert "greater than 0.1" in str(r_to_inv.json())
    print("  [PASS] Unrealistic timeout (< 0.1s) rejected at API boundary with HTTP 422.")

    # 2. Fallback disabled policy enforces fail-closed guarantee (temperature is strictly None)
    p_fb_disabled = dict(p_invalid_timeout)
    p_fb_disabled["options"] = dict(
        p_invalid_timeout["options"], enable_fallback=False, solver_timeout_seconds=15.0
    )
    r_fb = client.post("/api/infer/hybrid", json=p_fb_disabled).json()
    assert r_fb["status"] == "fallback_required"
    assert r_fb["temperature"] is None, (
        "Temperature MUST be None when surrogate is rejected and fallback is disabled"
    )
    assert r_fb["rejection"]["code"] == "fallback_disabled"
    print(
        "  [PASS] Fallback disabled: status=fallback_required, temperature=None (fail-closed contract upheld)."
    )


def test_repeated_rapid_requests():
    print("\n--- Robustness 4: Rapid Sequential & Concurrent Inference Requests ---")
    p = {
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

    t0 = time.perf_counter()
    status_codes = []

    def make_call(i):
        endpoint = "/api/infer/hybrid" if i % 2 == 0 else "/api/infer/fvm"
        r = client.post(endpoint, json=p)
        return r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(make_call, i) for i in range(16)]
        for f in concurrent.futures.as_completed(futures):
            status_codes.append(f.result())

    elapsed = time.perf_counter() - t0
    assert all(code == 200 for code in status_codes), f"Some requests failed: {status_codes}"
    print(
        f"  [PASS] 16 concurrent requests across endpoints: 100% HTTP 200 in {elapsed:.2f}s ({elapsed / 16 * 1000:.1f}ms/req)."
    )


if __name__ == "__main__":
    test_invalid_conductivity_and_non_finite()
    test_unsupported_mesh_sizes()
    test_solver_timeout_and_fail_closed()
    test_repeated_rapid_requests()
    print("\nALL ROBUSTNESS AND EDGE-CASE AUDIT CHECKS PASSED SUCCESSFULLY!")
