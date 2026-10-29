"""Audit and reproduce Milestone 09 claims and discrepancies.

Investigates:
1. Why neural API requests are slower than FVM at every resolution.
2. Why corrected and uncorrected neural modes reported identical MaxAE in Milestone 09 benchmarks.
3. Conflation of Jacobi indicator (Kelvin) with discrete linear residual (Watts).
4. Exceedance of the 1e-10 W absolute residual threshold at 256x256 due to O(sqrt(N)) scaling.
5. Conflation of pure compute speedup with full API request latency.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.neural_process_twin.api.server import create_app  # noqa: E402


def run_phase1_audit() -> dict[str, Any]:
    print("=" * 80)
    print("MILESTONE 10 - PHASE 1: AUDIT OF MILESTONE 09 CLAIMS AND DISCREPANCIES")
    print("=" * 80)

    app = create_app()
    client = TestClient(app)

    audit_findings: dict[str, Any] = {}

    # Base problem payload
    def get_payload(res: int, mode: str) -> dict[str, Any]:
        return {
            "mode": mode,
            "grid": {
                "x_min": 0.0,
                "x_max": 1.0,
                "y_min": 0.0,
                "y_max": 1.0,
                "nx": res,
                "ny": res,
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
                "solver_timeout_seconds": 30.0,
            },
        }

    # -------------------------------------------------------------------------
    # Investigation 1 & 5: Latency Decomposition and Speedup Conflation
    # -------------------------------------------------------------------------
    print("\n[Audit 1 & 5: Latency Breakdown and Speedup Conflation]")
    latency_audit = {}
    for res in [32, 64, 128, 256]:
        # FVM
        t0 = time.perf_counter()
        fvm_resp = client.post("/api/simulate", json=get_payload(res, "numerical_fvm")).json()
        fvm_api_ms = (time.perf_counter() - t0) * 1000.0
        fvm_internal_ms = fvm_resp["latency_ms"].get("solver_solve_ms", 0.0)

        # Neural (uncorrected)
        t0 = time.perf_counter()
        surr_resp = client.post(
            "/api/simulate", json=get_payload(res, "experimental_neural")
        ).json()
        surr_api_ms = (time.perf_counter() - t0) * 1000.0
        surr_internal_ms = surr_resp["latency_ms"].get("inference_ms", 0.0)

        # Compare endpoint
        t0 = time.perf_counter()
        _ = client.post("/api/compare", json=get_payload(res, "experimental_neural")).json()
        comp_api_ms = (time.perf_counter() - t0) * 1000.0

        pure_compute_speedup = fvm_internal_ms / max(surr_internal_ms, 1e-6)
        api_speedup = fvm_api_ms / max(surr_api_ms, 1e-6)

        latency_audit[f"{res}x{res}"] = {
            "fvm_api_ms": fvm_api_ms,
            "fvm_internal_ms": fvm_internal_ms,
            "neural_api_ms": surr_api_ms,
            "neural_internal_ms": surr_internal_ms,
            "compare_api_ms": comp_api_ms,
            "pure_compute_speedup": pure_compute_speedup,
            "api_speedup": api_speedup,
        }
        print(f"  {res}x{res}:")
        print(f"    FVM: API = {fvm_api_ms:.2f} ms | Internal Solver = {fvm_internal_ms:.2f} ms")
        print(
            f"    Neural: API = {surr_api_ms:.2f} ms | Pure Model Forward = {surr_internal_ms:.2f} ms"
        )
        print(f"    Compare Endpoint: {comp_api_ms:.2f} ms")
        print(f"    Speedup (Internal Pure Compute): {pure_compute_speedup:.2f}x")
        print(f"    Speedup (End-to-End API):         {api_speedup:.2f}x")

    audit_findings["latency_conflation"] = latency_audit

    # -------------------------------------------------------------------------
    # Investigation 2: Corrected vs Uncorrected MaxAE
    # -------------------------------------------------------------------------
    print("\n[Audit 2: Corrected vs Uncorrected MaxAE and Jacobi Residuals]")
    correction_audit = {}
    for res in [32, 64, 128, 256]:
        fvm = np.array(
            client.post("/api/simulate", json=get_payload(res, "numerical_fvm")).json()[
                "temperature"
            ]
        )
        uncorr_data = client.post(
            "/api/simulate", json=get_payload(res, "experimental_neural")
        ).json()
        corr_data = client.post(
            "/api/simulate", json=get_payload(res, "experimental_neural_corrected")
        ).json()

        t_uncorr = np.array(uncorr_data["temperature"])
        t_corr = np.array(corr_data["temperature"])

        diff_uncorr = np.abs(t_uncorr - fvm)
        diff_corr = np.abs(t_corr - fvm)
        diff_correction = np.abs(t_corr - t_uncorr)

        # Boundary ring mask
        ring_mask = np.zeros((res, res), dtype=bool)
        rw = min(2, res // 2)
        ring_mask[:rw, :] = True
        ring_mask[-rw:, :] = True
        ring_mask[:, :rw] = True
        ring_mask[:, -rw:] = True
        interior_mask = ~ring_mask

        modified_cells = int(np.sum(diff_correction > 1e-6))
        total_ring_cells = int(np.sum(ring_mask))

        uncorr_max_loc = [int(x) for x in np.unravel_index(np.argmax(diff_uncorr), (res, res))]
        corr_max_loc = [int(x) for x in np.unravel_index(np.argmax(diff_corr), (res, res))]

        correction_audit[f"{res}x{res}"] = {
            "modified_cells": modified_cells,
            "expected_ring_cells": total_ring_cells,
            "max_correction_magnitude_k": float(np.max(diff_correction)),
            "uncorrected_max_ae_k": float(np.max(diff_uncorr)),
            "corrected_max_ae_k": float(np.max(diff_corr)),
            "uncorrected_ring_max_ae_k": float(np.max(diff_uncorr[ring_mask])),
            "corrected_ring_max_ae_k": float(np.max(diff_corr[ring_mask])),
            "uncorrected_interior_max_ae_k": float(np.max(diff_uncorr[interior_mask])),
            "corrected_interior_max_ae_k": float(np.max(diff_corr[interior_mask])),
            "uncorrected_max_loc": uncorr_max_loc,
            "corrected_max_loc": corr_max_loc,
            "uncorrected_jacobi_indicator_k": uncorr_data["experimental_diagnostics"][
                "pointwise_jacobi_residual_max_k"
            ],
            "corrected_jacobi_indicator_k": corr_data["experimental_diagnostics"][
                "pointwise_jacobi_residual_max_k"
            ],
        }
        print(f"  {res}x{res}:")
        print(f"    Modified Cells: {modified_cells} (Expected 2-cell ring: {total_ring_cells})")
        print(f"    Max Correction Magnitude: {float(np.max(diff_correction)):.3f} K")
        print(
            f"    Boundary Ring MaxAE: Uncorr = {float(np.max(diff_uncorr[ring_mask])):.3f} K -> Corr = {float(np.max(diff_corr[ring_mask])):.3f} K"
        )
        print(
            f"    Interior MaxAE:      Uncorr = {float(np.max(diff_uncorr[interior_mask])):.3f} K -> Corr = {float(np.max(diff_corr[interior_mask])):.3f} K"
        )
        print(
            f"    Domain-Wide MaxAE:   Uncorr = {float(np.max(diff_uncorr)):.3f} K -> Corr = {float(np.max(diff_corr)):.3f} K"
        )

    audit_findings["correction_audit"] = correction_audit

    # -------------------------------------------------------------------------
    # Investigation 3 & 4: Residual Scaling and Units Conflation
    # -------------------------------------------------------------------------
    print("\n[Audit 3 & 4: Discrete Residual Scaling and Units Conflation]")
    residual_audit = {}
    for res in [32, 64, 128, 256]:
        fvm_json = client.post("/api/simulate", json=get_payload(res, "numerical_fvm")).json()
        alg_res_l2 = fvm_json["numerical_solver_info"]["algebraic_residual_norm"]
        pde_rel = fvm_json["diagnostics"]["pde_residual_relative"]
        pde_linf = fvm_json["diagnostics"]["pde_residual_linf_w"]

        # Number of cells
        n_cells = res * res
        rms_cell_res = alg_res_l2 / np.sqrt(n_cells)

        residual_audit[f"{res}x{res}"] = {
            "n_cells": n_cells,
            "algebraic_residual_l2_w": alg_res_l2,
            "rms_cell_residual_w": float(rms_cell_res),
            "max_norm_linf_w": pde_linf,
            "relative_residual": pde_rel,
            "exceeds_1e10_absolute": alg_res_l2 > 1e-10,
        }
        print(f"  {res}x{res} (N={n_cells}):")
        print(
            f"    L2 Norm ||r||_2:     {alg_res_l2:.3e} W (Exceeds 1e-10 W: {alg_res_l2 > 1e-10})"
        )
        print(f"    RMS Cell Residual:   {rms_cell_res:.3e} W")
        print(f"    Max-Norm ||r||_inf:  {pde_linf:.3e} W")
        print(f"    Relative Residual:   {pde_rel:.3e} (Machine epsilon level: <= 1e-12)")

    audit_findings["residual_scaling"] = residual_audit

    # Save findings artifact
    out_dir = PROJECT_ROOT / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "milestone_10_phase1_audit_findings.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_findings, f, indent=2)

    print(f"\n[Artifact Saved]: {out_file}")
    return audit_findings


if __name__ == "__main__":
    run_phase1_audit()
