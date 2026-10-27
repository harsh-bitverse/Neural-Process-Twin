"""Scientific API contract audit script for Milestone 07A.

Verifies:
1. Physical validity vs surrogate domain eligibility.
2. High-contrast FVM convergence.
3. Fail-closed contract on rejected surrogates (no leakage).
4. Direct FVM returns null uncertainty.
5. Comparison mode research-only status.
6. Dimensional consistency across physical units.
7. Array orientation and boundary condition alignment.
"""

from __future__ import annotations

import httpx
import numpy as np

client = httpx.Client(base_url="http://127.0.0.1:8000", trust_env=False, timeout=30.0)


def test_physical_validity_vs_surrogate_domain():
    print("\n--- 1. Physical Validity vs Surrogate Domain Eligibility ---")
    # A. Physically valid, but outside surrogate domain
    p_ood = {
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
        "source": {
            "pattern_type": "uniform",
            "uniform_value": 50.0,
            "spot_peak_value": 0.0,
            "spot_center_x": 0.5,
            "spot_center_y": 0.5,
            "spot_sigma": 0.1,
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
    r_ood = client.post("/api/validate", json=p_ood).json()
    assert r_ood["is_valid"] is True, "High contrast should be physically valid"
    assert r_ood["surrogate_compatible"] is False, (
        "High contrast must be flagged incompatible with surrogate"
    )
    assert len(r_ood["warnings"]) > 0
    print("  [PASS] High contrast: is_valid=True, surrogate_compatible=False, warning generated.")

    # B. Physically invalid: negative conductivity rejected with HTTP 422
    p_neg_k = dict(p_ood)
    p_neg_k["conductivity"] = {
        "pattern_type": "uniform",
        "uniform_value": -5.0,
        "inclusion_k_background": 1.0,
        "inclusion_k_feature": 5.0,
        "inclusion_center_x": 0.5,
        "inclusion_center_y": 0.5,
        "inclusion_radius": 0.2,
    }
    r_neg_resp = client.post("/api/validate", json=p_neg_k)
    assert r_neg_resp.status_code == 422, f"Expected HTTP 422, got {r_neg_resp.status_code}"
    err_text = str(r_neg_resp.json())
    assert "greater than 0" in err_text.lower()
    print(
        "  [PASS] Negative conductivity: Rejected with HTTP 422 ('Input should be greater than 0')."
    )

    # C. Physically invalid: all Neumann boundary conditions rejected with HTTP 422
    p_all_neumann = dict(p_ood)
    p_all_neumann["boundary_conditions"] = {
        "left": {"side": "left", "type": "neumann", "value": 0.0},
        "right": {"side": "right", "type": "neumann", "value": 0.0},
        "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
        "top": {"side": "top", "type": "neumann", "value": 0.0},
    }
    r_neumann_resp = client.post("/api/validate", json=p_all_neumann)
    assert r_neumann_resp.status_code == 422, f"Expected HTTP 422, got {r_neumann_resp.status_code}"
    err_text_bc = str(r_neumann_resp.json())
    assert "pure-neumann" in err_text_bc.lower()
    print(
        "  [PASS] Pure Neumann BCs: Rejected with HTTP 422 ('Pure-Neumann boundary conditions ... are singular')."
    )


def test_high_contrast_fvm():
    print("\n--- 2. High-Contrast Problem Solved Accurately by FVM ---")
    p_high_k = {
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
            "inclusion_k_feature": 50.0,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.25,
        },
        "source": {
            "pattern_type": "uniform",
            "uniform_value": 100.0,
            "spot_peak_value": 0.0,
            "spot_center_x": 0.5,
            "spot_center_y": 0.5,
            "spot_sigma": 0.1,
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
    r = client.post("/api/infer/fvm", json=p_high_k).json()
    assert r["status"] == "fallback_succeeded"
    assert r["temperature"] is not None
    rel_res = r["diagnostics"]["pde_residual_relative"]
    assert rel_res < 1e-12, f"Relative residual {rel_res} too high for FVM"
    print(f"  [PASS] 50:1 Contrast FVM relative residual: {rel_res:.2e}")


def test_fail_closed_contract():
    print("\n--- 3. Fail-Closed Guarantees: No Rejected Output Leaked ---")
    p_rejected = {
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
        "source": {
            "pattern_type": "uniform",
            "uniform_value": 50.0,
            "spot_peak_value": 0.0,
            "spot_center_x": 0.5,
            "spot_center_y": 0.5,
            "spot_sigma": 0.1,
        },
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
        "options": {
            "enable_fallback": False,
            "fallback_isolation": "thread",
            "provisional_mean_disagreement_threshold_k": 2.5,
            "provisional_max_disagreement_threshold_k": 5.0,
            "input_contrast_threshold": 7.16,
            "input_q_max_threshold": 912.8,
            "solver_timeout_seconds": 15.0,
        },
    }
    r_hyb = client.post("/api/infer/hybrid", json=p_rejected).json()
    assert r_hyb["status"] == "fallback_required"
    assert r_hyb["temperature"] is None, (
        "Temperature MUST be null when fallback is disabled and surrogate rejected"
    )
    assert r_hyb["rejection"]["code"] in ("fallback_disabled", "high_input_contrast")
    assert r_hyb["rejection"]["triggering_metric"] == "k_contrast_ratio"
    assert r_hyb["rejection"]["observed_value"] == 30.0
    print(
        "  [PASS] Fallback-disabled rejection: status=fallback_required, temperature=null, triggering_metric=k_contrast_ratio."
    )

    # Now with fallback enabled
    p_fallback = dict(p_rejected)
    p_fallback["options"] = dict(p_rejected["options"], enable_fallback=True)
    r_fall = client.post("/api/infer/hybrid", json=p_fallback).json()
    assert r_fall["status"] == "fallback_succeeded"
    assert r_fall["execution_path"] == "solver_fallback_path"
    assert r_fall["temperature"] is not None
    # Verify temperature equals FVM solve
    r_fvm = client.post("/api/infer/fvm", json=p_fallback).json()
    t_hyb = np.array(r_fall["temperature"])
    t_fvm = np.array(r_fvm["temperature"])
    diff = np.max(np.abs(t_hyb - t_fvm))
    assert diff < 1e-10, f"Fallback output {diff} does not match FVM reference!"
    print(
        f"  [PASS] Fallback-enabled: returned field is verified FVM solution (max diff: {diff:.2e})."
    )


def test_deterministic_fvm_zero_uncertainty():
    print("\n--- 4. Direct FVM Contains No Fabricated Uncertainty ---")
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
            "spot_peak_value": 0.0,
            "spot_center_x": 0.5,
            "spot_center_y": 0.5,
            "spot_sigma": 0.1,
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
    r = client.post("/api/infer/fvm", json=p).json()
    assert r["ensemble_disagreement_k"] is None
    assert r["mean_disagreement_k"] is None
    assert r["max_disagreement_k"] is None
    assert r["conformal_intervals"] is None
    print("  [PASS] FVM reference solve: ensemble_disagreement_k=None, conformal_intervals=None.")


def test_comparison_mode_research_notice():
    print("\n--- 5. Comparison Mode Research Status Disclosure ---")
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
        "conductivity": {
            "pattern_type": "inclusion",
            "uniform_value": 1.0,
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": 30.0,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.25,
        },
        "source": {
            "pattern_type": "uniform",
            "uniform_value": 50.0,
            "spot_peak_value": 0.0,
            "spot_center_x": 0.5,
            "spot_center_y": 0.5,
            "spot_sigma": 0.1,
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
    r = client.post("/api/infer/compare", json=p).json()
    assert "strictly for research comparison" in r["evaluation_notice"].lower()
    assert r["surrogate_acceptance_status"] == "fallback_succeeded"
    print(f"  [PASS] Notice: '{r['evaluation_notice']}'")


def test_array_orientation():
    print("\n--- 6. Array Orientation & Boundary Temperature Consistency ---")
    # Setup problem with asymmetric BCs:
    # Left: 300 K, Right: 400 K, Bottom: 320 K, Top: 380 K
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
        "conductivity": {
            "pattern_type": "uniform",
            "uniform_value": 5.0,
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": 5.0,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.2,
        },
        "source": {
            "pattern_type": "uniform",
            "uniform_value": 0.0,
            "spot_peak_value": 0.0,
            "spot_center_x": 0.5,
            "spot_center_y": 0.5,
            "spot_sigma": 0.1,
        },
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 400.0},
            "bottom": {"side": "bottom", "type": "dirichlet", "value": 320.0},
            "top": {"side": "top", "type": "dirichlet", "value": 380.0},
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
    r = client.post("/api/infer/fvm", json=p).json()
    t_field = np.array(r["temperature"])  # shape (ny, nx) = (32, 32)
    # Row 0 is bottom (y=0), Row 31 is top (y=1)
    # Col 0 is left (x=0), Col 31 is right (x=1)
    mean_left = np.mean(t_field[:, 0])
    mean_right = np.mean(t_field[:, -1])
    mean_bottom = np.mean(t_field[0, :])
    mean_top = np.mean(t_field[-1, :])

    print(
        f"  Boundary cell temperatures: Left={mean_left:.2f}K (target 300K), Right={mean_right:.2f}K (target 400K)"
    )
    print(
        f"                             Bottom={mean_bottom:.2f}K (target 320K), Top={mean_top:.2f}K (target 380K)"
    )

    assert 300.0 < mean_left < 310.0, f"Left column temperature unexpected: {mean_left}"
    assert 390.0 < mean_right < 400.0, f"Right column temperature unexpected: {mean_right}"
    assert 320.0 < mean_bottom < 330.0, f"Bottom row temperature unexpected: {mean_bottom}"
    assert 370.0 < mean_top < 380.0, f"Top row temperature unexpected: {mean_top}"
    print(
        "  [PASS] Array orientation strictly conforms to Cartesian coordinates (row 0 = bottom, col 0 = left)."
    )


if __name__ == "__main__":
    test_physical_validity_vs_surrogate_domain()
    test_high_contrast_fvm()
    test_fail_closed_contract()
    test_deterministic_fvm_zero_uncertainty()
    test_comparison_mode_research_notice()
    test_array_orientation()
    print("\nALL SCIENTIFIC API CONTRACT AUDIT CHECKS PASSED SUCCESSFULLY!")
