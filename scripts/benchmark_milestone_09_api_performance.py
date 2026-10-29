"""Benchmark Milestone 09: Dual-Mode Scientific Platform API Performance.

Measures cold-start vs warm request times for 32x32, 64x64, 128x128, and 256x256
across Verified Numerical FVM, Experimental Neural (A3), Experimental Neural Corrected (A4),
and Dual-Mode Scientific Comparison endpoints.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.neural_process_twin.api.server import create_app  # noqa: E402


def create_payload(resolution: int, mode: str) -> dict[str, Any]:
    """Generate a standard physical benchmark problem payload."""
    return {
        "mode": mode,
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": resolution,
            "ny": resolution,
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


def run_benchmarks(repetitions: int = 5) -> dict[str, Any]:
    """Execute end-to-end API benchmarks across resolutions and modes."""
    print("=" * 80)
    print("MILESTONE 09: DUAL-MODE SCIENTIFIC PLATFORM BENCHMARK SUITE")
    print("=" * 80)

    app = create_app()
    client = TestClient(app)

    resolutions = [32, 64, 128, 256]
    modes = ["numerical_fvm", "experimental_neural", "experimental_neural_corrected", "compare"]

    benchmark_data: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python_version": sys.version,
            "device": "CPU",
            "resolutions": resolutions,
            "repetitions": repetitions,
        },
        "results": {},
    }

    # First, test cold start for each mode at 32x32
    print("\n[Phase 1: Cold-Start Latencies (32x32)]")
    cold_starts: dict[str, float] = {}
    for mode in modes:
        endpoint = "/api/compare" if mode == "compare" else "/api/simulate"
        payload_mode = "experimental_neural" if mode == "compare" else mode
        payload = create_payload(32, payload_mode)
        t0 = time.perf_counter()
        resp = client.post(endpoint, json=payload)
        cold_time_ms = (time.perf_counter() - t0) * 1000.0
        assert resp.status_code == 200, f"Cold start failed for {mode}: {resp.text}"
        cold_starts[mode] = cold_time_ms
        print(f"  - {mode:<30} Cold-Start Latency: {cold_time_ms:8.2f} ms")

    benchmark_data["cold_start_latencies_ms"] = cold_starts

    # Second, evaluate warm latencies across all resolutions and modes
    print("\n[Phase 2: Multi-Resolution Warm Benchmark Suite]")
    print(
        f"{'Resolution':<12} | {'Mode':<28} | {'Warm Mean (ms)':<15} | {'Std (ms)':<10} | {'Speedup vs FVM':<15}"
    )
    print("-" * 88)

    fvm_means: dict[int, float] = {}

    for res in resolutions:
        benchmark_data["results"][f"{res}x{res}"] = {}

        # 1. Benchmark FVM reference first to establish baseline
        fvm_payload = create_payload(res, "numerical_fvm")
        # Warmup
        client.post("/api/simulate", json=fvm_payload)
        fvm_times = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            r = client.post("/api/simulate", json=fvm_payload)
            dt = (time.perf_counter() - t0) * 1000.0
            assert r.status_code == 200
            fvm_times.append(dt)

        fvm_mean = float(np.mean(fvm_times))
        fvm_std = float(np.std(fvm_times))
        fvm_means[res] = fvm_mean

        fvm_res_json = r.json()
        assert fvm_res_json["is_verified_numerical"] is True
        alg_res = fvm_res_json["numerical_solver_info"]["algebraic_residual_norm"]

        benchmark_data["results"][f"{res}x{res}"]["numerical_fvm"] = {
            "latencies_ms": fvm_times,
            "mean_ms": fvm_mean,
            "std_ms": fvm_std,
            "median_ms": float(np.median(fvm_times)),
            "p95_ms": float(np.percentile(fvm_times, 95)),
            "algebraic_residual_norm_w": alg_res,
            "is_verified_numerical": True,
            "speedup_vs_fvm": 1.0,
        }
        print(
            f"{f'{res}x{res}':<12} | {'numerical_fvm':<28} | {fvm_mean:15.2f} | {fvm_std:10.2f} | {'1.00x':<15}"
        )

        # 2. Benchmark Experimental Neural (A3)
        surr_payload = create_payload(res, "experimental_neural")
        client.post("/api/simulate", json=surr_payload)
        surr_times = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            r = client.post("/api/simulate", json=surr_payload)
            dt = (time.perf_counter() - t0) * 1000.0
            assert r.status_code == 200
            surr_times.append(dt)

        surr_mean = float(np.mean(surr_times))
        surr_std = float(np.std(surr_times))
        surr_speedup = float(fvm_mean / max(surr_mean, 1e-6))

        surr_res_json = r.json()
        assert surr_res_json["is_verified_numerical"] is False
        jacobi_res = surr_res_json["experimental_diagnostics"]["pointwise_jacobi_residual_max_k"]

        benchmark_data["results"][f"{res}x{res}"]["experimental_neural"] = {
            "latencies_ms": surr_times,
            "mean_ms": surr_mean,
            "std_ms": surr_std,
            "median_ms": float(np.median(surr_times)),
            "p95_ms": float(np.percentile(surr_times, 95)),
            "pointwise_jacobi_residual_max_k": jacobi_res,
            "is_verified_numerical": False,
            "speedup_vs_fvm": surr_speedup,
        }
        print(
            f"{f'{res}x{res}':<12} | {'experimental_neural':<28} | {surr_mean:15.2f} | {surr_std:10.2f} | {f'{surr_speedup:.2f}x':<15}"
        )

        # 3. Benchmark Experimental Neural Corrected (A4 Jacobi)
        corr_payload = create_payload(res, "experimental_neural_corrected")
        client.post("/api/simulate", json=corr_payload)
        corr_times = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            r = client.post("/api/simulate", json=corr_payload)
            dt = (time.perf_counter() - t0) * 1000.0
            assert r.status_code == 200
            corr_times.append(dt)

        corr_mean = float(np.mean(corr_times))
        corr_std = float(np.std(corr_times))
        corr_speedup = float(fvm_mean / max(corr_mean, 1e-6))

        corr_res_json = r.json()
        assert corr_res_json["is_verified_numerical"] is False
        jacobi_res_corr = corr_res_json["experimental_diagnostics"][
            "pointwise_jacobi_residual_max_k"
        ]

        benchmark_data["results"][f"{res}x{res}"]["experimental_neural_corrected"] = {
            "latencies_ms": corr_times,
            "mean_ms": corr_mean,
            "std_ms": corr_std,
            "median_ms": float(np.median(corr_times)),
            "p95_ms": float(np.percentile(corr_times, 95)),
            "pointwise_jacobi_residual_max_k": jacobi_res_corr,
            "is_verified_numerical": False,
            "speedup_vs_fvm": corr_speedup,
        }
        print(
            f"{f'{res}x{res}':<12} | {'experimental_neural_corr':<28} | {corr_mean:15.2f} | {corr_std:10.2f} | {f'{corr_speedup:.2f}x':<15}"
        )

        # 4. Benchmark Dual-Mode Comparison Endpoint
        comp_payload = create_payload(res, "experimental_neural")
        client.post("/api/compare", json=comp_payload)
        comp_times = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            r = client.post("/api/compare", json=comp_payload)
            dt = (time.perf_counter() - t0) * 1000.0
            assert r.status_code == 200
            comp_times.append(dt)

        comp_mean = float(np.mean(comp_times))
        comp_std = float(np.std(comp_times))
        comp_json = r.json()

        benchmark_data["results"][f"{res}x{res}"]["compare"] = {
            "latencies_ms": comp_times,
            "mean_ms": comp_mean,
            "std_ms": comp_std,
            "median_ms": float(np.median(comp_times)),
            "p95_ms": float(np.percentile(comp_times, 95)),
            "rmse_k": comp_json["rmse_k"],
            "max_absolute_error_k": comp_json["max_absolute_error_k"],
            "relative_l2_error_pct": comp_json["relative_l2_error_pct"],
            "fvm_latency_ms": comp_json["fvm_latency_ms"],
            "surrogate_latency_ms": comp_json["surrogate_latency_ms"],
            "speedup_factor": comp_json["speedup_factor"],
        }
        print(
            f"{f'{res}x{res}':<12} | {'compare (FVM + Neural)':<28} | {comp_mean:15.2f} | {comp_std:10.2f} | {'N/A (Both)':<15}"
        )
        print("-" * 88)

    # Save benchmark report to disk
    out_dir = PROJECT_ROOT / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "milestone_09_api_benchmarks.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\n[Saved Authoritative Benchmark Artifact]: {out_file}")
    return benchmark_data


if __name__ == "__main__":
    run_benchmarks(repetitions=5)
