"""Milestone 12: Final Frozen Reproducible Benchmark Suite.

Evaluates across all canonical resolutions: 32x32, 64x64, 128x128, 256x256
Systems evaluated:
1. Numerical Reference FVM (HeatConductionSolver2D)
2. Experimental Neural A3 (RefinedFNO2D uncorrected)
3. Experimental Neural A4 (RefinedFNO2D + Selective Boundary Jacobi)
4. Complete API Simulate Endpoint (FVM, A3, A4)
5. Dual-Mode API Compare Endpoint

Evaluates across 4 physical regimes:
- IID: Nominal conduction (k in [0.8, 6.0], q in [20, 80] W/m^3, smooth gradients)
- OOD High Contrast: Sharp material contrast (k_ratio in [15, 30])
- OOD High Source: Extreme heat generation (q in [300, 800] W/m^3)
- OOD Boundary Step: Severe temperature steps (200 K to 600 K)

Metrics tracked:
- RMSE [K], Relative L2 [%], MaxAE Median/P95/Max [K]
- Boundary ring MaxAE (outer 2 cells) [K], Interior MaxAE [K]
- Physical boundary flux imbalance [W and %]
- Engineering tolerance compliance rate (<= 5 K, <= 10 K) [%]
- Latency: Pure solve/inference [ms], Full API round-trip [ms], P50/P95/P99 [ms]
- Peak memory consumption [MB]
- Hardware and execution environment metadata

Saves results to: artifacts/reports/milestone_12_final_benchmark_suite.json
"""

from __future__ import annotations

import json
import os
import platform
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi.testclient import TestClient

from neural_process_twin.api.app import create_app
from neural_process_twin.api.converters import (
    build_boundary_conditions,
    build_conductivity_array,
    build_domain_grid,
    build_source_array,
)
from neural_process_twin.api.models import (
    BoundaryConditionItem,
    BoundaryConditionsModel,
    FieldPatternModel,
    GridModel,
    HeatSourcePatternModel,
)
from neural_process_twin.api.routes import get_experimental_surrogate, get_solver


def create_test_cohort(seed: int = 1200000) -> dict[str, list[dict[str, Any]]]:
    """Generate deterministic, disjoint problem specifications across physical regimes."""
    rng = np.random.default_rng(seed)
    cohort: dict[str, list[dict[str, Any]]] = {
        "IID": [],
        "OOD_Contrast": [],
        "OOD_Source": [],
        "OOD_Boundary": [],
    }

    # 1. IID cases (10 realizations)
    for i in range(10):
        cohort["IID"].append(
            {
                "case_id": f"iid_{i:02d}",
                "k_pattern": FieldPatternModel(
                    pattern_type="inclusion",
                    inclusion_k_background=float(rng.uniform(0.8, 1.5)),
                    inclusion_k_feature=float(rng.uniform(2.5, 6.0)),
                    inclusion_center_x=float(rng.uniform(0.3, 0.7)),
                    inclusion_center_y=float(rng.uniform(0.3, 0.7)),
                    inclusion_radius=float(rng.uniform(0.12, 0.28)),
                ),
                "q_pattern": HeatSourcePatternModel(
                    pattern_type="uniform",
                    uniform_value=float(rng.uniform(20.0, 80.0)),
                ),
                "bcs": BoundaryConditionsModel(
                    left=BoundaryConditionItem(
                        side="left", type="dirichlet", value=float(rng.uniform(280.0, 320.0))
                    ),
                    right=BoundaryConditionItem(
                        side="right", type="dirichlet", value=float(rng.uniform(330.0, 380.0))
                    ),
                    bottom=BoundaryConditionItem(side="bottom", type="neumann", value=0.0),
                    top=BoundaryConditionItem(side="top", type="neumann", value=0.0),
                ),
            }
        )

    # 2. OOD High Contrast (5 realizations)
    for i in range(5):
        cohort["OOD_Contrast"].append(
            {
                "case_id": f"ood_contrast_{i:02d}",
                "k_pattern": FieldPatternModel(
                    pattern_type="inclusion",
                    inclusion_k_background=float(rng.uniform(0.2, 0.5)),
                    inclusion_k_feature=float(rng.uniform(8.0, 15.0)),  # 16x - 75x contrast
                    inclusion_center_x=float(rng.uniform(0.35, 0.65)),
                    inclusion_center_y=float(rng.uniform(0.35, 0.65)),
                    inclusion_radius=float(rng.uniform(0.15, 0.25)),
                ),
                "q_pattern": HeatSourcePatternModel(
                    pattern_type="uniform",
                    uniform_value=float(rng.uniform(30.0, 60.0)),
                ),
                "bcs": BoundaryConditionsModel(
                    left=BoundaryConditionItem(side="left", type="dirichlet", value=300.0),
                    right=BoundaryConditionItem(side="right", type="dirichlet", value=350.0),
                    bottom=BoundaryConditionItem(side="bottom", type="neumann", value=0.0),
                    top=BoundaryConditionItem(side="top", type="neumann", value=0.0),
                ),
            }
        )

    # 3. OOD High Source (5 realizations)
    for i in range(5):
        cohort["OOD_Source"].append(
            {
                "case_id": f"ood_source_{i:02d}",
                "k_pattern": FieldPatternModel(
                    pattern_type="uniform",
                    uniform_value=float(rng.uniform(1.0, 2.5)),
                ),
                "q_pattern": HeatSourcePatternModel(
                    pattern_type="gaussian_spot",
                    spot_peak_value=float(rng.uniform(400.0, 900.0)),
                    spot_center_x=float(rng.uniform(0.3, 0.7)),
                    spot_center_y=float(rng.uniform(0.3, 0.7)),
                    spot_sigma=float(rng.uniform(0.10, 0.20)),
                ),
                "bcs": BoundaryConditionsModel(
                    left=BoundaryConditionItem(side="left", type="dirichlet", value=300.0),
                    right=BoundaryConditionItem(side="right", type="dirichlet", value=300.0),
                    bottom=BoundaryConditionItem(side="bottom", type="dirichlet", value=300.0),
                    top=BoundaryConditionItem(side="top", type="dirichlet", value=300.0),
                ),
            }
        )

    # 4. OOD Boundary Step (5 realizations)
    for i in range(5):
        cohort["OOD_Boundary"].append(
            {
                "case_id": f"ood_boundary_{i:02d}",
                "k_pattern": FieldPatternModel(
                    pattern_type="gradient",
                    inclusion_k_background=1.0,
                    inclusion_k_feature=3.0,
                ),
                "q_pattern": HeatSourcePatternModel(
                    pattern_type="uniform",
                    uniform_value=10.0,
                ),
                "bcs": BoundaryConditionsModel(
                    left=BoundaryConditionItem(
                        side="left", type="dirichlet", value=float(rng.uniform(180.0, 240.0))
                    ),
                    right=BoundaryConditionItem(
                        side="right", type="dirichlet", value=float(rng.uniform(520.0, 620.0))
                    ),
                    bottom=BoundaryConditionItem(
                        side="bottom", type="neumann", value=float(rng.uniform(-20.0, 20.0))
                    ),
                    top=BoundaryConditionItem(side="top", type="neumann", value=0.0),
                ),
            }
        )

    return cohort


def evaluate_benchmark() -> dict[str, Any]:
    print("=" * 80)
    print("Neural Process Twin: Milestone 12 Frozen Reproducible Benchmark Suite")
    print("=" * 80)

    tracemalloc.start()
    t_start = time.perf_counter()

    # Pre-warm singletons and API client
    app = create_app()
    client = TestClient(app)
    solver = get_solver()
    surrogate = get_experimental_surrogate()

    resolutions = [32, 64, 128, 256]
    cohort = create_test_cohort(seed=1200000)

    hardware_env = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "num_cpu_threads": torch.get_num_threads(),
        "physical_cores": os.cpu_count(),
    }

    results: dict[str, Any] = {
        "metadata": {
            "milestone": "12",
            "suite_title": "Final Frozen Reproducible Benchmark Suite",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware_environment": hardware_env,
            "total_cases_per_resolution": sum(len(v) for v in cohort.values()),
        },
        "per_resolution_results": {},
        "api_throughput_and_latency": {},
    }

    for res in resolutions:
        print(f"\n--- Benchmarking Resolution {res}x{res} ---")
        grid_model = GridModel(nx=res, ny=res, lx=1.0, ly=1.0)
        grid = build_domain_grid(grid_model)

        # 2-cell boundary ring mask for localized error tracking
        ring_mask = np.zeros((res, res), dtype=bool)
        ring_mask[:2, :] = True
        ring_mask[-2:, :] = True
        ring_mask[:, :2] = True
        ring_mask[:, -2:] = True
        interior_mask = ~ring_mask

        res_data: dict[str, Any] = {
            "subgroups": {},
            "aggregate": {},
        }

        all_fvm_times = []
        all_a3_times = []
        all_a4_times = []
        all_a3_rmse = []
        all_a4_rmse = []
        all_a3_maxae = []
        all_a4_maxae = []
        all_a3_ring_maxae = []
        all_a4_ring_maxae = []
        all_a3_interior_maxae = []
        all_a4_interior_maxae = []
        all_a3_flux_imb = []
        all_a4_flux_imb = []

        for subgroup_name, cases in cohort.items():
            sg_fvm_times = []
            sg_a3_times = []
            sg_a4_times = []
            sg_a3_rmse = []
            sg_a4_rmse = []
            sg_a3_maxae = []
            sg_a4_maxae = []

            for c in cases:
                k_field = build_conductivity_array(grid, c["k_pattern"])
                q_field = build_source_array(grid, c["q_pattern"])
                bcs = build_boundary_conditions(c["bcs"])

                # 1. Reference FVM solve
                t0_fvm = time.perf_counter()
                fvm_res = solver.solve(grid, k_field, q_field, bcs)
                t_fvm = (time.perf_counter() - t0_fvm) * 1000.0
                sg_fvm_times.append(t_fvm)
                all_fvm_times.append(t_fvm)
                t_true = fvm_res.temperature

                # 2. Neural A3 (Uncorrected)
                t0_a3 = time.perf_counter()
                a3_res = surrogate.predict(
                    grid, k_field, q_field, bcs, apply_physics_correction=False
                )
                t_a3 = (time.perf_counter() - t0_a3) * 1000.0
                sg_a3_times.append(t_a3)
                all_a3_times.append(t_a3)
                t_a3_pred = a3_res.temperature

                # 3. Neural A4 (Corrected with Selective Jacobi)
                t0_a4 = time.perf_counter()
                a4_res = surrogate.predict(
                    grid, k_field, q_field, bcs, apply_physics_correction=True
                )
                t_a4 = (time.perf_counter() - t0_a4) * 1000.0
                sg_a4_times.append(t_a4)
                all_a4_times.append(t_a4)
                t_a4_pred = a4_res.temperature

                # Compute physical accuracy metrics
                err_a3 = np.abs(t_a3_pred - t_true)
                err_a4 = np.abs(t_a4_pred - t_true)

                rmse_a3 = float(np.sqrt(np.mean((t_a3_pred - t_true) ** 2)))
                rmse_a4 = float(np.sqrt(np.mean((t_a4_pred - t_true) ** 2)))
                maxae_a3 = float(np.max(err_a3))
                maxae_a4 = float(np.max(err_a4))

                ring_a3 = float(np.max(err_a3[ring_mask]))
                ring_a4 = float(np.max(err_a4[ring_mask]))
                int_a3 = float(np.max(err_a3[interior_mask]))
                int_a4 = float(np.max(err_a4[interior_mask]))

                sg_a3_rmse.append(rmse_a3)
                sg_a4_rmse.append(rmse_a4)
                sg_a3_maxae.append(maxae_a3)
                sg_a4_maxae.append(maxae_a4)

                all_a3_rmse.append(rmse_a3)
                all_a4_rmse.append(rmse_a4)
                all_a3_maxae.append(maxae_a3)
                all_a4_maxae.append(maxae_a4)
                all_a3_ring_maxae.append(ring_a3)
                all_a4_ring_maxae.append(ring_a4)
                all_a3_interior_maxae.append(int_a3)
                all_a4_interior_maxae.append(int_a4)

                if a3_res.heat_balance is not None:
                    all_a3_flux_imb.append(a3_res.heat_balance.relative_imbalance)
                if a4_res.heat_balance is not None:
                    all_a4_flux_imb.append(a4_res.heat_balance.relative_imbalance)

            res_data["subgroups"][subgroup_name] = {
                "case_count": len(cases),
                "fvm_latency_mean_ms": float(np.mean(sg_fvm_times)),
                "a3_latency_mean_ms": float(np.mean(sg_a3_times)),
                "a4_latency_mean_ms": float(np.mean(sg_a4_times)),
                "a3_rmse_mean_k": float(np.mean(sg_a3_rmse)),
                "a4_rmse_mean_k": float(np.mean(sg_a4_rmse)),
                "a3_maxae_median_k": float(np.median(sg_a3_maxae)),
                "a4_maxae_median_k": float(np.median(sg_a4_maxae)),
                "a3_maxae_p95_k": float(np.percentile(sg_a3_maxae, 95)),
                "a4_maxae_p95_k": float(np.percentile(sg_a4_maxae, 95)),
            }

        # Aggregate metrics across all 25 problem realizations
        def pct_compliance(err_list: list[float], thresh: float) -> float:
            return float(np.mean(np.array(err_list) <= thresh) * 100.0)

        res_data["aggregate"] = {
            "total_cases": len(all_fvm_times),
            "fvm": {
                "latency_mean_ms": float(np.mean(all_fvm_times)),
                "latency_p50_ms": float(np.median(all_fvm_times)),
                "latency_p95_ms": float(np.percentile(all_fvm_times, 95)),
            },
            "a3_uncorrected": {
                "rmse_mean_k": float(np.mean(all_a3_rmse)),
                "maxae_median_k": float(np.median(all_a3_maxae)),
                "maxae_p95_k": float(np.percentile(all_a3_maxae, 95)),
                "maxae_max_k": float(np.max(all_a3_maxae)),
                "boundary_ring_maxae_mean_k": float(np.mean(all_a3_ring_maxae)),
                "interior_maxae_mean_k": float(np.mean(all_a3_interior_maxae)),
                "compliance_5k_pct": pct_compliance(all_a3_maxae, 5.0),
                "compliance_10k_pct": pct_compliance(all_a3_maxae, 10.0),
                "latency_mean_ms": float(np.mean(all_a3_times)),
                "latency_p50_ms": float(np.median(all_a3_times)),
                "latency_p95_ms": float(np.percentile(all_a3_times, 95)),
                "mean_heat_imbalance_pct": float(np.mean(all_a3_flux_imb) * 100.0),
            },
            "a4_corrected": {
                "rmse_mean_k": float(np.mean(all_a4_rmse)),
                "maxae_median_k": float(np.median(all_a4_maxae)),
                "maxae_p95_k": float(np.percentile(all_a4_maxae, 95)),
                "maxae_max_k": float(np.max(all_a4_maxae)),
                "boundary_ring_maxae_mean_k": float(np.mean(all_a4_ring_maxae)),
                "interior_maxae_mean_k": float(np.mean(all_a4_interior_maxae)),
                "compliance_5k_pct": pct_compliance(all_a4_maxae, 5.0),
                "compliance_10k_pct": pct_compliance(all_a4_maxae, 10.0),
                "latency_mean_ms": float(np.mean(all_a4_times)),
                "latency_p50_ms": float(np.median(all_a4_times)),
                "latency_p95_ms": float(np.percentile(all_a4_times, 95)),
                "mean_heat_imbalance_pct": float(np.mean(all_a4_flux_imb) * 100.0),
            },
            "pure_compute_speedup_a3_vs_fvm": float(np.mean(all_fvm_times) / np.mean(all_a3_times)),
            "pure_compute_speedup_a4_vs_fvm": float(np.mean(all_fvm_times) / np.mean(all_a4_times)),
        }

        # 4. API End-to-End Latency Benchmark for this resolution
        print(f"  Measuring complete API round-trip latency at {res}x{res}...")
        api_payload_fvm = {
            "mode": "numerical_fvm",
            "grid": {"nx": res, "ny": res, "lx": 1.0, "ly": 1.0},
            "conductivity": {"pattern_type": "uniform", "uniform_value": 2.0},
            "source": {"pattern_type": "uniform", "uniform_value": 50.0},
            "boundary_conditions": {
                "left": {"side": "left", "type": "dirichlet", "value": 300.0},
                "right": {"side": "right", "type": "dirichlet", "value": 350.0},
                "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
                "top": {"side": "top", "type": "neumann", "value": 0.0},
            },
        }
        api_payload_a3 = dict(
            api_payload_fvm, mode="experimental_neural", compute_diagnostics=False
        )
        api_payload_compare = dict(api_payload_fvm, mode="experimental_neural_corrected")

        def time_api(url: str, payload: dict[str, Any], reps: int = 10) -> dict[str, float]:
            lats = []
            for _ in range(reps):
                t0 = time.perf_counter()
                r = client.post(url, json=payload)
                assert r.status_code == 200
                lats.append((time.perf_counter() - t0) * 1000.0)
            return {
                "mean_ms": float(np.mean(lats)),
                "p50_ms": float(np.median(lats)),
                "p95_ms": float(np.percentile(lats, 95)),
            }

        res_data["api_performance"] = {
            "simulate_fvm": time_api("/api/simulate", api_payload_fvm),
            "simulate_a3_opt": time_api("/api/simulate", api_payload_a3),
            "compare_dual_mode": time_api("/api/compare", api_payload_compare),
        }

        results["per_resolution_results"][f"{res}x{res}"] = res_data

        print(
            f"  {res}x{res} Summary: FVM={res_data['aggregate']['fvm']['latency_mean_ms']:.1f}ms | "
            f"A3={res_data['aggregate']['a3_uncorrected']['latency_mean_ms']:.1f}ms (Speedup: {res_data['aggregate']['pure_compute_speedup_a3_vs_fvm']:.2f}x) | "
            f"A3 RMSE={res_data['aggregate']['a3_uncorrected']['rmse_mean_k']:.2f}K | "
            f"A4 RMSE={res_data['aggregate']['a4_corrected']['rmse_mean_k']:.2f}K"
        )

    _current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.perf_counter() - t_start

    results["metadata"]["total_benchmark_duration_sec"] = round(total_time, 2)
    results["metadata"]["peak_memory_mb"] = round(peak_mem / (1024 * 1024), 2)

    out_path = Path("artifacts/reports/milestone_12_final_benchmark_suite.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved final frozen benchmark suite to: {out_path}")
    print(
        f"Total benchmark time: {total_time:.2f}s | Peak memory: {results['metadata']['peak_memory_mb']} MB"
    )
    return results


if __name__ == "__main__":
    evaluate_benchmark()
