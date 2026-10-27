"""Risk-Latency Optimization Benchmark for Milestone 08E (Phase 5).

Evaluates routing policies across 32x32, 64x64, 128x128, and 256x256:
- R0: Direct FVM
- R1: Production 5-Member Ensemble Hybrid
- R2: Pre-screening + U1 Single-Pass
- R3: Pre-screening + U3 T=3 Multi-Pass
- R4: Pre-screening + U3 T=5 Multi-Pass
- R5: Adaptive Cascade (Pre-screening -> U1 -> Ambiguous U3 -> FVM)

Measures:
- Acceptance rate, fallback rate, false acceptance rate (FAR)
- Accepted case MaxAE and RMSE
- Hybrid final-output RMSE
- Latency percentiles: Mean, P50, P95, P99
- Net speedup versus direct FVM
- Evaluated on test_iid, test_ood, and representative deployment mixtures (80/15/5).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_process_twin.inference.adaptive_routing import (
    AdaptiveHybridEngine,
    RoutingPolicy,
    RoutingThresholds,
)
from neural_process_twin.inference.contracts import InferenceStatus
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_08e_risk_latency")


def parse_bcs_row(grp: h5py.Group, idx: int) -> BoundaryConditions:
    """Parse BoundaryConditions instance for sample idx."""
    bc_grp = grp["boundary_conditions"]
    l_type = bc_grp["left_type"][idx]
    r_type = bc_grp["right_type"][idx]
    b_type = bc_grp["bottom_type"][idx]
    t_type = bc_grp["top_type"][idx]

    l_val = bc_grp["left_value"][idx]
    r_val = bc_grp["right_value"][idx]
    b_val = bc_grp["bottom_value"][idx]
    t_val = bc_grp["top_value"][idx]

    def _str(val: Any) -> str:
        return val.decode("utf-8").lower() if isinstance(val, bytes) else str(val).lower()

    def _side_bc(side: BoundarySide, t_str: str, val_arr: np.ndarray) -> BoundaryCondition:
        bc_t = BoundaryType.DIRICHLET if _str(t_str) == "dirichlet" else BoundaryType.NEUMANN
        return BoundaryCondition(side, bc_t, val_arr)

    return BoundaryConditions(
        left=_side_bc(BoundarySide.LEFT, l_type, l_val),
        right=_side_bc(BoundarySide.RIGHT, r_type, r_val),
        bottom=_side_bc(BoundarySide.BOTTOM, b_type, b_val),
        top=_side_bc(BoundarySide.TOP, t_type, t_val),
    )


def run_risk_latency_benchmark() -> dict[str, Any]:
    """Execute risk-latency optimization benchmark across resolutions and policies."""
    test_path = Path("artifacts/data/multires_dataset.h5")
    thresholds_path = Path("artifacts/reports/milestone_08e_calibrated_thresholds.json")

    with open(thresholds_path) as f:
        thresh_dict = json.load(f)

    r_thresh = RoutingThresholds(
        tau_k_max=thresh_dict["pre_screening"]["tau_k_max"],
        tau_q_max=thresh_dict["pre_screening"]["tau_q_max"],
        tau_u1_unc=thresh_dict["R2_U1"]["tau_unc"],
        tau_u3_t3_unc=thresh_dict["R3_U3_T3"]["tau_unc"],
        tau_u3_t5_unc=thresh_dict["R4_U3_T5"]["tau_unc"],
        tau_u1_cascade_safe=thresh_dict["R5_cascade"]["tau_u1_safe"],
        tau_u1_cascade_ambiguous=thresh_dict["R5_cascade"]["tau_u1_ambiguous"],
        tau_flux_imb_pct=thresh_dict["R5_cascade"]["tau_flux_imb_pct"],
    )

    logger.info("Initializing AdaptiveHybridEngine with calibrated thresholds...")
    engine = AdaptiveHybridEngine(thresholds=r_thresh)

    resolutions = [32, 64, 128, 256]
    policies = [
        RoutingPolicy.R0_DIRECT_FVM,
        RoutingPolicy.R1_PRODUCTION_HYBRID,
        RoutingPolicy.R2_U1_SINGLE_PASS,
        RoutingPolicy.R3_U3_T3_MULTI_PASS,
        RoutingPolicy.R4_U3_T5_MULTI_PASS,
        RoutingPolicy.R5_ADAPTIVE_CASCADE,
    ]

    benchmark_results: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": str(test_path),
            "thresholds": thresh_dict,
            "threads": 8,
            "profiler_instrumentation": "NONE (clean CPU timing, tracemalloc disabled)",
        },
        "per_resolution_benchmarks": {},
        "summary_table": {},
    }

    with h5py.File(test_path, "r") as f:
        for res in resolutions:
            res_key = f"res_{res}"
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
            logger.info(f"=== Benchmarking Grid Resolution {res}x{res} ===")

            benchmark_results["per_resolution_benchmarks"][res_key] = {}
            res_grp = f[res_key]

            # Load IID and OOD
            iid_grp = res_grp["test_iid"]
            ood_grp = res_grp["test_ood"]

            n_iid = min(50, iid_grp["temperature"].shape[0])
            n_ood = min(50, ood_grp["temperature"].shape[0])

            iid_k = iid_grp["k"][:n_iid]
            iid_q = iid_grp["q"][:n_iid]
            iid_t = iid_grp["temperature"][:n_iid]
            iid_bcs = [parse_bcs_row(iid_grp, i) for i in range(n_iid)]

            ood_k = ood_grp["k"][:n_ood]
            ood_q = ood_grp["q"][:n_ood]
            ood_t = ood_grp["temperature"][:n_ood]
            ood_subgroups = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in ood_grp["subgroup"][:n_ood]
            ]
            ood_bcs = [parse_bcs_row(ood_grp, i) for i in range(n_ood)]

            # Baseline reference time (R0)
            fvm_latencies = []
            for i in range(n_iid):
                res_r0 = engine.route_inference(
                    grid, iid_k[i], iid_q[i], iid_bcs[i], policy=RoutingPolicy.R0_DIRECT_FVM
                )
                fvm_latencies.append(res_r0.timings_ms["t_total"])
            baseline_fvm_mean_ms = float(np.mean(fvm_latencies))

            for pol in policies:
                pol_name = pol.value
                logger.info(f"  Evaluating {pol_name} at {res}x{res}...")

                # 1. Warm-up
                _ = engine.route_inference(grid, iid_k[0], iid_q[0], iid_bcs[0], policy=pol)

                # 2. Evaluate on IID
                iid_latencies = []
                iid_accepted = 0
                iid_false_accept_10k = 0
                iid_false_accept_5k = 0
                iid_accepted_maxaes = []
                iid_accepted_rmses = []
                iid_hybrid_errors = []

                for i in range(n_iid):
                    res_obj = engine.route_inference(
                        grid, iid_k[i], iid_q[i], iid_bcs[i], policy=pol
                    )
                    iid_latencies.append(res_obj.timings_ms["t_total"])

                    err_field = np.abs(res_obj.temperature - iid_t[i])
                    maxae = float(np.max(err_field))
                    rmse = float(np.sqrt(np.mean(err_field**2)))
                    iid_hybrid_errors.append(rmse)

                    if res_obj.status == InferenceStatus.ACCEPTED:
                        iid_accepted += 1
                        iid_accepted_maxaes.append(maxae)
                        iid_accepted_rmses.append(rmse)
                        if maxae > 10.0:
                            iid_false_accept_10k += 1
                        if maxae > 5.0:
                            iid_false_accept_5k += 1

                # 3. Evaluate on OOD
                ood_latencies = []
                ood_accepted = 0
                ood_false_accept_10k = 0
                ood_false_accept_5k = 0
                ood_accepted_maxaes = []
                ood_accepted_rmses = []
                ood_hybrid_errors = []
                ood_escalations = 0

                # Subgroup tracking
                hc_accepted = 0
                hc_total = 0
                grf_accepted = 0
                grf_total = 0

                for i in range(n_ood):
                    res_obj = engine.route_inference(
                        grid, ood_k[i], ood_q[i], ood_bcs[i], policy=pol
                    )
                    ood_latencies.append(res_obj.timings_ms["t_total"])

                    err_field = np.abs(res_obj.temperature - ood_t[i])
                    maxae = float(np.max(err_field))
                    rmse = float(np.sqrt(np.mean(err_field**2)))
                    ood_hybrid_errors.append(rmse)

                    is_hc = "inclusion" in ood_subgroups[i].lower()
                    if is_hc:
                        hc_total += 1
                    else:
                        grf_total += 1

                    if res_obj.status == InferenceStatus.ACCEPTED:
                        ood_accepted += 1
                        ood_accepted_maxaes.append(maxae)
                        ood_accepted_rmses.append(rmse)
                        if is_hc:
                            hc_accepted += 1
                        else:
                            grf_accepted += 1
                        if maxae > 10.0:
                            ood_false_accept_10k += 1
                        if maxae > 5.0:
                            ood_false_accept_5k += 1
                    else:
                        ood_escalations += 1

                # Mixture calculations: 80% IID, 15% HC, 5% GRF
                iid_lat_mean = float(np.mean(iid_latencies))
                ood_lat_mean = float(np.mean(ood_latencies))
                mix_lat_mean = 0.80 * iid_lat_mean + 0.20 * ood_lat_mean

                iid_acc_rate = float(iid_accepted / n_iid)
                ood_acc_rate = float(ood_accepted / n_ood)
                mix_acc_rate = 0.80 * iid_acc_rate + 0.20 * ood_acc_rate

                mix_hybrid_rmse = float(
                    0.80 * np.mean(iid_hybrid_errors) + 0.20 * np.mean(ood_hybrid_errors)
                )
                speedup_vs_fvm = float(baseline_fvm_mean_ms / max(mix_lat_mean, 1e-4))

                all_lats = iid_latencies + ood_latencies

                benchmark_results["per_resolution_benchmarks"][res_key][pol_name] = {
                    "latency_ms": {
                        "mean": float(np.mean(all_lats)),
                        "p50": float(np.percentile(all_lats, 50)),
                        "p95": float(np.percentile(all_lats, 95)),
                        "p99": float(np.percentile(all_lats, 99)),
                        "iid_mean": iid_lat_mean,
                        "ood_mean": ood_lat_mean,
                        "mix_mean": mix_lat_mean,
                        "speedup_vs_fvm": speedup_vs_fvm,
                    },
                    "iid_metrics": {
                        "n_samples": n_iid,
                        "acceptance_rate_pct": iid_acc_rate * 100.0,
                        "accepted_rmse_k": float(np.mean(iid_accepted_rmses))
                        if iid_accepted_rmses
                        else 0.0,
                        "accepted_maxae_k": float(np.mean(iid_accepted_maxaes))
                        if iid_accepted_maxaes
                        else 0.0,
                        "hybrid_rmse_k": float(np.mean(iid_hybrid_errors)),
                        "false_accept_count_10k": iid_false_accept_10k,
                        "false_accept_count_5k": iid_false_accept_5k,
                    },
                    "ood_metrics": {
                        "n_samples": n_ood,
                        "acceptance_rate_pct": ood_acc_rate * 100.0,
                        "escalation_rate_pct": float(ood_escalations / n_ood) * 100.0,
                        "accepted_rmse_k": float(np.mean(ood_accepted_rmses))
                        if ood_accepted_rmses
                        else 0.0,
                        "accepted_maxae_k": float(np.mean(ood_accepted_maxaes))
                        if ood_accepted_maxaes
                        else 0.0,
                        "hybrid_rmse_k": float(np.mean(ood_hybrid_errors)),
                        "false_accept_count_10k": ood_false_accept_10k,
                        "false_accept_count_5k": ood_false_accept_5k,
                        "subgroups": {
                            "high_contrast_accept_pct": (hc_accepted / max(hc_total, 1)) * 100.0,
                            "grf_accept_pct": (grf_accepted / max(grf_total, 1)) * 100.0,
                        },
                    },
                    "deployment_mixture": {
                        "acceptance_rate_pct": mix_acc_rate * 100.0,
                        "hybrid_rmse_k": mix_hybrid_rmse,
                        "speedup_vs_fvm": speedup_vs_fvm,
                    },
                }

    out_file = Path("artifacts/reports/milestone_08e_risk_latency.json")
    with open(out_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    logger.info(f"Risk-latency benchmarks saved to {out_file}")
    return benchmark_results


if __name__ == "__main__":
    res = run_risk_latency_benchmark()
    print("\n=== BENCHMARK SUMMARY (res_256) ===")
    r256 = res["per_resolution_benchmarks"]["res_256"]
    for pol_k, data in r256.items():
        mix = data["deployment_mixture"]
        lat = data["latency_ms"]
        print(
            f"{pol_k:25s} | Mix Lat: {lat['mix_mean']:6.2f} ms | P95: {lat['p95']:6.2f} ms | Speedup: {mix['speedup_vs_fvm']:5.2f}x | Mix RMSE: {mix['hybrid_rmse_k']:5.2f} K | Acc: {mix['acceptance_rate_pct']:4.1f}%"
        )
