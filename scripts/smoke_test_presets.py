"""Smoke test script for Milestone 07A live application verification.

Executes all 5 problem presets against the running API server at http://127.0.0.1:8000.
Records HTTP status codes, latency, execution paths, and physical conservation metrics.
"""

from __future__ import annotations

import json

import httpx

client = httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=30.0)

PRESETS = {
    "nominal_composite": {
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
    },
    "high_contrast": {
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
            "inclusion_k_feature": 30.0,  # 30:1 > 7.16
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.25,
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
    },
    "uniform_conduction": {
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
            "pattern_type": "uniform",
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
    },
    "extreme_source": {
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
            "pattern_type": "uniform",
            "uniform_value": 2.0,
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": 5.0,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.2,
        },
        "source": {
            "pattern_type": "gaussian_spot",
            "uniform_value": 0.0,
            "spot_peak_value": 1200.0,  # 1200 > 912.8
            "spot_center_x": 0.5,
            "spot_center_y": 0.5,
            "spot_sigma": 0.12,
        },
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 300.0},
            "bottom": {"side": "bottom", "type": "dirichlet", "value": 300.0},
            "top": {"side": "top", "type": "dirichlet", "value": 300.0},
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
    },
    "grid_64": {
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": 64,
            "ny": 64,
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
    },
}


def run_smoke_tests():
    results = {}
    print("=" * 80)
    print("MILESTONE 07A — LIVE API SMOKE TEST RESULTS")
    print("=" * 80)

    for name, payload in PRESETS.items():
        print(f"\n--- Testing Preset: {name} ---")
        # 1. Validation
        v_resp = client.post("/api/validate", json=payload)
        val = v_resp.json()
        print(
            f"Validation: HTTP {v_resp.status_code} | valid={val['is_valid']}, surrogate_compatible={val['surrogate_compatible']}"
        )
        if val["errors"]:
            print(f"  Errors: {val['errors']}")
        if val["warnings"]:
            print(f"  Warnings: {val['warnings']}")

        # 2. Hybrid Inference
        h_resp = client.post("/api/infer/hybrid", json=payload)
        h = h_resp.json()
        print(
            f"Hybrid:     HTTP {h_resp.status_code} | status={h['status']}, path={h['execution_path']}"
        )
        print(
            f"  Latency: total={h['latency']['total_pipeline_ms']:.2f}ms | surrogate={h['latency']['surrogate_inference_ms']:.2f}ms | fallback={h['latency']['solver_fallback_ms']:.2f}ms"
        )
        if h["temperature"] is not None:
            t_min = h["temperature_min_k"]
            t_max = h["temperature_max_k"]
            print(f"  Temperature: min={t_min:.2f}K, max={t_max:.2f}K")
        else:
            print("  Temperature: NULL (fail-closed)")
        if h["rejection"]["code"] != "none":
            print(f"  Rejection: code={h['rejection']['code']} | msg={h['rejection']['message']}")

        # 3. Direct FVM Inference
        f_resp = client.post("/api/infer/fvm", json=payload)
        f = f_resp.json()
        print(
            f"Direct FVM: HTTP {f_resp.status_code} | status={f['status']}, path={f['execution_path']}"
        )
        print(
            f"  FVM Disagreement is None: {f['ensemble_disagreement_k'] is None} | Conformal is None: {f['conformal_intervals'] is None}"
        )
        print(f"  FVM Residual Rel: {f['diagnostics']['pde_residual_relative']:.2e}")

        # 4. Comparison Mode
        c_resp = client.post("/api/infer/compare", json=payload)
        c = c_resp.json()
        print(
            f"Comparison: HTTP {c_resp.status_code} | RMSE={c['rmse_k']} | Speedup FVM/Surr={c['fvm_speedup_factor']:.2f}x"
        )
        print(f"  Surrogate Acceptance Status: {c['surrogate_acceptance_status']}")

        results[name] = {
            "validation_code": v_resp.status_code,
            "hybrid_code": h_resp.status_code,
            "hybrid_status": h["status"],
            "hybrid_path": h["execution_path"],
            "fvm_code": f_resp.status_code,
            "fvm_rel_res": f["diagnostics"]["pde_residual_relative"],
            "comparison_code": c_resp.status_code,
            "rmse_k": c["rmse_k"],
            "speedup_factor": c["fvm_speedup_factor"],
        }

    return results


if __name__ == "__main__":
    res = run_smoke_tests()
    with open("artifacts/smoke_test_preset_results.json", "w") as fp:
        json.dump(res, fp, indent=2)
    print("\nResults saved to artifacts/smoke_test_preset_results.json")
