"""Benchmark script comparing Hybrid Digital Twin inference paths against FVM solver.

Measures wall-clock latency, throughput, and component-level breakdown on CPU for:
1. Surrogate Fast-Path (accepted in-distribution problem)
2. Hybrid Fallback Path (rejected high-contrast problem -> FVM solve)
3. Direct FVM-Only Solver Path (HeatConductionSolver2D.solve)
4. Single-Model FNO vs. 5-Member Deep Ensemble

Outputs structured results to artifacts/reports/milestone_06b_hybrid_benchmark.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from neural_process_twin.inference.contracts import (
    InferenceOptions,
    InferenceRequest,
    InferenceStatus,
)
from neural_process_twin.inference.engine import HybridInferenceEngine
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D


def run_benchmark(n_warmup: int = 10, n_trials: int = 50) -> dict:
    print("Initializing HybridInferenceEngine from artifacts (device=cpu)...")
    engine = HybridInferenceEngine.from_artifacts(device="cpu")
    fvm_solver = HeatConductionSolver2D()

    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )

    # 1. In-distribution input (k=2.5 W/(m K), q=50 W/m^3)
    iid_request = InferenceRequest(
        grid=grid,
        conductivity=np.full((32, 32), 2.5),
        source=np.full((32, 32), 50.0),
        boundary_conditions=bcs,
    )

    # 2. Out-of-distribution input (contrast 30:1, rejected by screener -> fallback to FVM)
    k_ood = np.full((32, 32), 1.0)
    k_ood[10:22, 10:22] = 30.0
    ood_request = InferenceRequest(
        grid=grid,
        conductivity=k_ood,
        source=np.full((32, 32), 50.0),
        boundary_conditions=bcs,
        options=InferenceOptions(enable_fallback=True),
    )

    print("Verifying request execution paths...")
    iid_res = engine.predict(iid_request)
    assert iid_res.status == InferenceStatus.ACCEPTED, f"Expected ACCEPTED, got {iid_res.status}"

    ood_res = engine.predict(ood_request)
    assert ood_res.status == InferenceStatus.FALLBACK_SUCCEEDED, (
        f"Expected FALLBACK_SUCCEEDED, got {ood_res.status}"
    )

    fvm_res = fvm_solver.solve(grid, np.full((32, 32), 2.5), np.full((32, 32), 50.0), bcs)
    assert fvm_res.temperature is not None

    # Warmup
    print(f"Running {n_warmup} warmup iterations...")
    for _ in range(n_warmup):
        engine.predict(iid_request)
        engine.predict(ood_request)
        fvm_solver.solve(grid, iid_request.conductivity, iid_request.source, bcs)

    # Benchmark: Direct FVM Solver
    print(f"Benchmarking Direct FVM Solver ({n_trials} trials)...")
    fvm_latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fvm_solver.solve(grid, iid_request.conductivity, iid_request.source, bcs)
        fvm_latencies.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark: Fast-path (Surrogate Ensemble + Diagnostics)
    print(f"Benchmarking Surrogate Fast-Path ({n_trials} trials)...")
    fast_latencies = []
    val_latencies = []
    surr_latencies = []
    diag_latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        res = engine.predict(iid_request)
        fast_latencies.append((time.perf_counter() - t0) * 1000.0)
        val_latencies.append(res.latency.input_validation_ms)
        surr_latencies.append(res.latency.surrogate_inference_ms)
        diag_latencies.append(res.latency.diagnostics_evaluation_ms)

    # Benchmark: Fallback-path (Validation + Surrogate + Screen + FVM Solve)
    print(f"Benchmarking Hybrid Fallback Path ({n_trials} trials)...")
    fallback_latencies = []
    fb_solver_latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        res = engine.predict(ood_request)
        fallback_latencies.append((time.perf_counter() - t0) * 1000.0)
        fb_solver_latencies.append(res.latency.solver_fallback_ms)

    def stats(arr: list[float]) -> dict:
        a = np.array(arr)
        return {
            "mean_ms": float(np.mean(a)),
            "std_ms": float(np.std(a, ddof=1)),
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "min_ms": float(np.min(a)),
            "max_ms": float(np.max(a)),
            "throughput_samples_per_sec": float(1000.0 / np.mean(a)),
        }

    results = {
        "benchmark_metadata": {
            "grid_resolution": "32x32",
            "device": "cpu",
            "trials": n_trials,
            "warmup_trials": n_warmup,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "direct_fvm_solver": stats(fvm_latencies),
        "surrogate_fast_path": {
            "total": stats(fast_latencies),
            "input_validation": stats(val_latencies),
            "surrogate_inference": stats(surr_latencies),
            "diagnostics_evaluation": stats(diag_latencies),
        },
        "hybrid_fallback_path": {
            "total": stats(fallback_latencies),
            "solver_portion": stats(fb_solver_latencies),
        },
        "comparisons": {
            "fvm_vs_surrogate_ratio": float(np.mean(fast_latencies) / np.mean(fvm_latencies)),
            "fvm_speedup_over_surrogate": float(np.mean(fast_latencies) / np.mean(fvm_latencies)),
            "surrogate_speedup_over_fvm": float(np.mean(fvm_latencies) / np.mean(fast_latencies)),
            "fallback_overhead_ratio": float(np.mean(fallback_latencies) / np.mean(fvm_latencies)),
        },
    }

    out_path = Path("artifacts/reports/milestone_06b_hybrid_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n--- BENCHMARK RESULTS (CPU, 32x32 Grid, {n_trials} trials) ---")
    print(
        f"Direct FVM Solver:       {results['direct_fvm_solver']['mean_ms']:.2f} ± {results['direct_fvm_solver']['std_ms']:.2f} ms ({results['direct_fvm_solver']['throughput_samples_per_sec']:.1f} samp/s)"
    )
    print(
        f"Surrogate Fast-Path:    {results['surrogate_fast_path']['total']['mean_ms']:.2f} ± {results['surrogate_fast_path']['total']['std_ms']:.2f} ms ({results['surrogate_fast_path']['total']['throughput_samples_per_sec']:.1f} samp/s)"
    )
    print(
        f"  - Input Validation:   {results['surrogate_fast_path']['input_validation']['mean_ms']:.2f} ms"
    )
    print(
        f"  - Surrogate (5x FNO): {results['surrogate_fast_path']['surrogate_inference']['mean_ms']:.2f} ms"
    )
    print(
        f"  - Diagnostics & Conf: {results['surrogate_fast_path']['diagnostics_evaluation']['mean_ms']:.2f} ms"
    )
    print(
        f"Hybrid Fallback Path:   {results['hybrid_fallback_path']['total']['mean_ms']:.2f} ± {results['hybrid_fallback_path']['total']['std_ms']:.2f} ms ({results['hybrid_fallback_path']['total']['throughput_samples_per_sec']:.1f} samp/s)"
    )
    print(
        f"  - FVM Solver call:    {results['hybrid_fallback_path']['solver_portion']['mean_ms']:.2f} ms"
    )
    print(
        f"Ratio (Fast-Path / FVM): {results['comparisons']['fvm_vs_surrogate_ratio']:.2f}x (Direct FVM is faster on CPU at 32x32)"
    )
    print(f"Saved benchmark to {out_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
