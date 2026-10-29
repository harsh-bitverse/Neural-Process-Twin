"""Milestone 10: 12-Stage API Latency Decomposition and Performance Attribution Benchmark.

Instruments and isolates every stage of the API request lifecycle:
1. Request parsing (raw JSON string to dictionary)
2. Input validation (Pydantic SimulationRequestModel validation)
3. Physical feature generation (k-field, q-field, boundary objects)
4. Continuous physical encoding (8-channel multi-resolution tensor)
5. FVM matrix assembly (sparse CSR matrix and RHS vector)
6. Numerical solving (Direct sparse SuperLU factorization)
7. Model forward pass (A3 RefinedFNO2D neural forward pass)
8. Physics correction (Selective Boundary Jacobi relaxation)
9. Discrete linear residual computation (||b - A*T||_2 and Jacobi indicator)
10. Energy balance diagnostics (cell-by-cell boundary heat balance)
11. Pydantic response formatting (constructing SimulationResponse)
12. JSON serialization (serializing response to JSON payload)

Evaluates:
- Cold-start vs warm requests (N=25 iterations)
- Across 32x32, 64x64, 128x128, and 256x256
- Modes: numerical_fvm, experimental_neural (uncorrected), experimental_neural_corrected
- Performance with diagnostics vs optimized without diagnostics
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
import torch

from neural_process_twin.api.converters import (
    build_boundary_conditions,
    build_conductivity_array,
    build_domain_grid,
    build_source_array,
    format_experimental_simulation_response,
    format_fvm_simulation_response,
)
from neural_process_twin.api.models import SimulationRequestModel
from neural_process_twin.api.routes import get_experimental_surrogate, get_solver
from neural_process_twin.data.continuous_encoding import encode_continuous_physical_inputs
from neural_process_twin.solvers.assembly import assemble_steady_heat_system
from neural_process_twin.solvers.diagnostics import compute_heat_balance


def generate_canonical_payload(n: int, mode: str, compute_diagnostics: bool = True) -> str:
    """Generate raw JSON request payload for testing."""
    payload = {
        "mode": mode,
        "grid": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "nx": n,
            "ny": n,
            "thickness": 1.0,
        },
        "conductivity": {
            "pattern_type": "inclusion",
            "inclusion_k_background": 1.0,
            "inclusion_k_feature": 5.0,
            "inclusion_center_x": 0.5,
            "inclusion_center_y": 0.5,
            "inclusion_radius": 0.2,
        },
        "source": {
            "pattern_type": "uniform",
            "uniform_value": 50.0,
        },
        "boundary_conditions": {
            "left": {"side": "left", "type": "dirichlet", "value": 300.0},
            "right": {"side": "right", "type": "dirichlet", "value": 350.0},
            "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
            "top": {"side": "top", "type": "neumann", "value": 0.0},
        },
        "compute_diagnostics": compute_diagnostics,
    }
    return json.dumps(payload)


def benchmark_fvm_stages(
    n: int, compute_diagnostics: bool = True, iterations: int = 25
) -> dict[str, Any]:
    """Isolate and measure the latency stages of verified numerical FVM mode."""
    raw_json = generate_canonical_payload(n, "numerical_fvm", compute_diagnostics)
    solver = get_solver()

    stage_timings: dict[str, list[float]] = {
        "1_request_parsing": [],
        "2_input_validation": [],
        "3_physical_feature_gen": [],
        "5_matrix_assembly": [],
        "6_numerical_solving": [],
        "9_residual_norm": [],
        "10_energy_diagnostics": [],
        "11_pydantic_formatting": [],
        "12_json_serialization": [],
        "total_pipeline": [],
    }

    for iter_idx in range(iterations + 1):
        is_cold = iter_idx == 0
        t_pipe_start = time.perf_counter()

        # Stage 1: Request parsing
        t0 = time.perf_counter()
        req_dict = json.loads(raw_json)
        t_parse = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Input validation
        t0 = time.perf_counter()
        req_model = SimulationRequestModel.model_validate(req_dict)
        t_val = (time.perf_counter() - t0) * 1000.0

        # Stage 3: Physical feature generation
        t0 = time.perf_counter()
        grid = build_domain_grid(req_model.grid)
        k_field = build_conductivity_array(grid, req_model.conductivity)
        q_field = build_source_array(grid, req_model.source)
        bcs = build_boundary_conditions(req_model.boundary_conditions)
        t_feat = (time.perf_counter() - t0) * 1000.0

        # Stage 5: Matrix assembly
        t0 = time.perf_counter()
        sys = assemble_steady_heat_system(grid=grid, k_field=k_field, q_field=q_field, bcs=bcs)
        t_assem = (time.perf_counter() - t0) * 1000.0

        # Stage 6: Numerical solving
        t0 = time.perf_counter()
        t_flat = spla.spsolve(sys.A, sys.b)
        t_field = t_flat.reshape((grid.ny, grid.nx))
        t_solve = (time.perf_counter() - t0) * 1000.0

        # Stage 9: Residual norm
        t0 = time.perf_counter()
        res_vec = sys.A.dot(t_flat) - sys.b
        _ = float(np.linalg.norm(res_vec))
        t_res = (time.perf_counter() - t0) * 1000.0

        # Stage 10: Energy diagnostics
        t0 = time.perf_counter()
        if compute_diagnostics:
            _ = compute_heat_balance(
                grid=grid,
                temperature=t_field,
                k_field=k_field,
                q_field=q_field,
                bcs=bcs,
            )
        t_diag = (time.perf_counter() - t0) * 1000.0

        # Stage 11: Pydantic response formatting
        t0 = time.perf_counter()
        solver_res = solver.solve(grid, k_field, q_field, bcs)  # complete solver object
        resp = format_fvm_simulation_response(solver_res, grid, k_field, q_field, bcs, t_solve)
        t_format = (time.perf_counter() - t0) * 1000.0

        # Stage 12: JSON serialization
        t0 = time.perf_counter()
        _ = json.dumps(resp.model_dump())
        t_serial = (time.perf_counter() - t0) * 1000.0

        t_pipe = (time.perf_counter() - t_pipe_start) * 1000.0

        if not is_cold:
            stage_timings["1_request_parsing"].append(t_parse)
            stage_timings["2_input_validation"].append(t_val)
            stage_timings["3_physical_feature_gen"].append(t_feat)
            stage_timings["5_matrix_assembly"].append(t_assem)
            stage_timings["6_numerical_solving"].append(t_solve)
            stage_timings["9_residual_norm"].append(t_res)
            stage_timings["10_energy_diagnostics"].append(t_diag)
            stage_timings["11_pydantic_formatting"].append(t_format)
            stage_timings["12_json_serialization"].append(t_serial)
            stage_timings["total_pipeline"].append(t_pipe)

    summary = {}
    for stage, times in stage_timings.items():
        summary[stage] = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "median_ms": float(np.median(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }
    return summary


def benchmark_neural_stages(
    n: int,
    apply_correction: bool,
    compute_diagnostics: bool = True,
    iterations: int = 25,
) -> dict[str, Any]:
    """Isolate and measure the latency stages of experimental neural surrogate mode."""
    mode_name = "experimental_neural_corrected" if apply_correction else "experimental_neural"
    raw_json = generate_canonical_payload(n, mode_name, compute_diagnostics)
    surrogate = get_experimental_surrogate()

    stage_timings: dict[str, list[float]] = {
        "1_request_parsing": [],
        "2_input_validation": [],
        "3_physical_feature_gen": [],
        "4_continuous_encoding": [],
        "5_matrix_assembly": [],
        "7_model_forward_pass": [],
        "8_physics_correction": [],
        "9_linear_residual": [],
        "10_energy_diagnostics": [],
        "11_pydantic_formatting": [],
        "12_json_serialization": [],
        "total_pipeline": [],
    }

    for iter_idx in range(iterations + 1):
        is_cold = iter_idx == 0
        t_pipe_start = time.perf_counter()

        # Stage 1: Request parsing
        t0 = time.perf_counter()
        req_dict = json.loads(raw_json)
        t_parse = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Input validation
        t0 = time.perf_counter()
        req_model = SimulationRequestModel.model_validate(req_dict)
        t_val = (time.perf_counter() - t0) * 1000.0

        # Stage 3: Physical feature generation
        t0 = time.perf_counter()
        grid = build_domain_grid(req_model.grid)
        k_field = build_conductivity_array(grid, req_model.conductivity)
        q_field = build_source_array(grid, req_model.source)
        bcs = build_boundary_conditions(req_model.boundary_conditions)
        t_feat = (time.perf_counter() - t0) * 1000.0

        # Stage 4: Continuous physical encoding
        t0 = time.perf_counter()
        x_in = encode_continuous_physical_inputs(
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            grid=grid,
            normalizer=surrogate.normalizer,
            use_log_k=True,
            log_k_mean=surrogate.log_k_mean,
            log_k_std=surrogate.log_k_std,
        )
        x_tensor = torch.from_numpy(x_in).unsqueeze(0).to(surrogate.device)
        t_enc = (time.perf_counter() - t0) * 1000.0

        # Stage 7: Model forward pass
        t0 = time.perf_counter()
        with torch.no_grad():
            out_norm = surrogate.model(x_tensor)
            t_pred_raw = surrogate.normalizer.inverse_transform_temperature(
                out_norm.cpu().numpy()[0, 0]
            ).astype(np.float64)
        t_fwd = (time.perf_counter() - t0) * 1000.0

        # Stage 5: Matrix assembly (conditional)
        t0 = time.perf_counter()
        sys = None
        if apply_correction or compute_diagnostics:
            sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)
        t_assem = (time.perf_counter() - t0) * 1000.0

        # Stage 8: Physics correction (conditional)
        t0 = time.perf_counter()
        if apply_correction and sys is not None:
            t_final = surrogate.apply_selective_boundary_jacobi(
                t_init=t_pred_raw,
                system=sys,
                grid=grid,
                ring_width=2,
                omega=0.67,
                num_iters=2,
            )
        else:
            t_final = t_pred_raw
        t_corr = (time.perf_counter() - t0) * 1000.0

        # Stage 9: Linear residual & Jacobi indicator (conditional)
        t0 = time.perf_counter()
        if compute_diagnostics and sys is not None:
            a_csr = sys.A.tocsr()
            t_flat = t_final.ravel()
            r = sys.b - a_csr.dot(t_flat)
            _ = float(np.linalg.norm(r))
            diag_a = a_csr.diagonal()
            inv_diag = np.where(np.abs(diag_a) > 1e-12, 1.0 / diag_a, 0.0)
            _ = float(np.max(np.abs(inv_diag * r)))
        t_res = (time.perf_counter() - t0) * 1000.0

        # Stage 10: Energy diagnostics (conditional)
        t0 = time.perf_counter()
        if compute_diagnostics:
            _ = compute_heat_balance(grid, t_final, k_field, q_field, bcs)
        t_diag = (time.perf_counter() - t0) * 1000.0

        # Stage 11: Pydantic formatting
        t0 = time.perf_counter()
        surr_res = surrogate.predict(
            grid, k_field, q_field, bcs, apply_correction, compute_diagnostics
        )
        resp = format_experimental_simulation_response(
            surr_res, grid, k_field, q_field, bcs, mode_name=mode_name
        )
        t_format = (time.perf_counter() - t0) * 1000.0

        # Stage 12: JSON serialization
        t0 = time.perf_counter()
        _ = json.dumps(resp.model_dump())
        t_serial = (time.perf_counter() - t0) * 1000.0

        t_pipe = (time.perf_counter() - t_pipe_start) * 1000.0

        if not is_cold:
            stage_timings["1_request_parsing"].append(t_parse)
            stage_timings["2_input_validation"].append(t_val)
            stage_timings["3_physical_feature_gen"].append(t_feat)
            stage_timings["4_continuous_encoding"].append(t_enc)
            stage_timings["5_matrix_assembly"].append(t_assem)
            stage_timings["7_model_forward_pass"].append(t_fwd)
            stage_timings["8_physics_correction"].append(t_corr)
            stage_timings["9_linear_residual"].append(t_res)
            stage_timings["10_energy_diagnostics"].append(t_diag)
            stage_timings["11_pydantic_formatting"].append(t_format)
            stage_timings["12_json_serialization"].append(t_serial)
            stage_timings["total_pipeline"].append(t_pipe)

    summary = {}
    for stage, times in stage_timings.items():
        summary[stage] = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "median_ms": float(np.median(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
        }
    return summary


def main() -> None:
    print("=" * 80)
    print("Milestone 10: 12-Stage API Latency Decomposition Benchmark")
    print("=" * 80)

    resolutions = [32, 64, 128, 256]
    benchmark_results: dict[str, Any] = {
        "metadata": {
            "benchmark": "Milestone 10 Latency Attribution & Optimization",
            "iterations_per_resolution": 25,
            "device": "cpu",
            "resolutions": resolutions,
        },
        "numerical_fvm": {},
        "experimental_neural_with_diagnostics": {},
        "experimental_neural_optimized_no_diagnostics": {},
        "experimental_neural_corrected_with_diagnostics": {},
        "experimental_neural_corrected_optimized": {},
        "comparison_table": [],
    }

    for n in resolutions:
        print(f"\n--- Benchmarking Resolution {n}x{n} ---")

        # 1. Numerical FVM
        print("  Running Verified Numerical FVM...")
        fvm_res = benchmark_fvm_stages(n, compute_diagnostics=True)
        benchmark_results["numerical_fvm"][f"{n}x{n}"] = fvm_res

        # 2. Experimental Neural (Uncorrected) with Diagnostics
        print("  Running Neural Uncorrected (with diagnostics)...")
        neural_with_diag = benchmark_neural_stages(
            n, apply_correction=False, compute_diagnostics=True
        )
        benchmark_results["experimental_neural_with_diagnostics"][f"{n}x{n}"] = neural_with_diag

        # 3. Experimental Neural (Uncorrected) Optimized (No Diagnostics)
        print("  Running Neural Uncorrected (optimized, no diagnostics)...")
        neural_opt = benchmark_neural_stages(n, apply_correction=False, compute_diagnostics=False)
        benchmark_results["experimental_neural_optimized_no_diagnostics"][f"{n}x{n}"] = neural_opt

        # 4. Experimental Neural Corrected with Diagnostics
        print("  Running Neural Corrected (with diagnostics)...")
        corr_with_diag = benchmark_neural_stages(n, apply_correction=True, compute_diagnostics=True)
        benchmark_results["experimental_neural_corrected_with_diagnostics"][f"{n}x{n}"] = (
            corr_with_diag
        )

        # 5. Experimental Neural Corrected Optimized (No Diagnostics)
        print("  Running Neural Corrected (optimized, no diagnostics)...")
        corr_opt = benchmark_neural_stages(n, apply_correction=True, compute_diagnostics=False)
        benchmark_results["experimental_neural_corrected_optimized"][f"{n}x{n}"] = corr_opt

        # Summary entry
        fvm_total = fvm_res["total_pipeline"]["mean_ms"]
        fvm_solve = fvm_res["6_numerical_solving"]["mean_ms"]
        fwd_mean = neural_opt["7_model_forward_pass"]["mean_ms"]
        neural_with_diag_total = neural_with_diag["total_pipeline"]["mean_ms"]
        neural_opt_total = neural_opt["total_pipeline"]["mean_ms"]

        speedup_pure_compute = fvm_solve / max(fwd_mean, 1e-6)
        speedup_api_optimized = fvm_total / max(neural_opt_total, 1e-6)

        benchmark_results["comparison_table"].append(
            {
                "resolution": f"{n}x{n}",
                "fvm_total_api_ms": round(fvm_total, 2),
                "fvm_solve_only_ms": round(fvm_solve, 2),
                "neural_forward_only_ms": round(fwd_mean, 2),
                "neural_api_with_diag_ms": round(neural_with_diag_total, 2),
                "neural_api_optimized_ms": round(neural_opt_total, 2),
                "pure_compute_speedup": round(speedup_pure_compute, 2),
                "api_end_to_end_speedup": round(speedup_api_optimized, 2),
            }
        )

    out_path = Path("artifacts/reports/milestone_10_latency_decomposition.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nSaved benchmark decomposition to: {out_path}")

    print("\n--- Latency Attribution Summary Table ---")
    print(
        f"{'Res':<10} | {'FVM API (ms)':<14} | {'FVM Solve (ms)':<16} | {'Neural Fwd (ms)':<16} | {'Neural w/ Diag (ms)':<20} | {'Neural Opt (ms)':<16} | {'Compute Speedup':<16} | {'API Speedup':<12}"
    )
    print("-" * 130)
    for row in benchmark_results["comparison_table"]:
        print(
            f"{row['resolution']:<10} | {row['fvm_total_api_ms']:<14.2f} | {row['fvm_solve_only_ms']:<16.2f} | {row['neural_forward_only_ms']:<16.2f} | {row['neural_api_with_diag_ms']:<20.2f} | {row['neural_api_optimized_ms']:<16.2f} | {row['pure_compute_speedup']:<16.2f} | {row['api_end_to_end_speedup']:<12.2f}"
        )


if __name__ == "__main__":
    main()
