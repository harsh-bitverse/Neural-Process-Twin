"""Evaluation and acceptance calibration script for Milestone 06C.

Executes Phase 2, Phase 3, and Phase 5:
1. Evaluates baseline hybrid inference on IID test, OOD test, and independent calibration sets.
2. Computes per-sample ground-truth temperature errors (RMSE, RelL2, MaxAE),
   uncertainty diagnostics, discrete PDE residuals, and energy balances.
3. Performs correlation analysis and evaluates candidate error tolerances (1K, 2K, 5K, 10K).
4. Calibrates physics gates (residual and flux imbalance) exclusively on the calibration set.
5. Evaluates the calibrated policy out-of-sample on test_iid and test_ood.
6. Benchmarks direct FVM, surrogate-only, hybrid fail-closed, and hybrid with fallback.
7. Saves comprehensive machine-readable results to artifacts/reports/milestone_06c_acceptance_results.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy import stats

from neural_process_twin.inference.contracts import (
    InferenceOptions,
    InferenceRequest,
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


def load_dataset_partition(
    h5_path: Path,
    split_name: str,
) -> list[dict[str, Any]]:
    """Extract sample fields, boundary conditions, and true temperatures from HDF5."""
    samples = []
    with h5py.File(h5_path, "r") as f:
        if split_name not in f:
            raise KeyError(f"Partition {split_name} not found in {h5_path}")
        grp = f[split_name]
        k_all = grp["k"][:]
        q_all = grp["q"][:]
        t_all = grp["temperature"][:]
        bc_grp = grp["boundary_conditions"]
        n_samples = len(k_all)

        for i in range(n_samples):
            bcs = BoundaryConditions(
                left=BoundaryCondition(
                    BoundarySide.LEFT,
                    BoundaryType(bc_grp["left_type"][i].decode("utf-8")),
                    bc_grp["left_value"][i],
                ),
                right=BoundaryCondition(
                    BoundarySide.RIGHT,
                    BoundaryType(bc_grp["right_type"][i].decode("utf-8")),
                    bc_grp["right_value"][i],
                ),
                bottom=BoundaryCondition(
                    BoundarySide.BOTTOM,
                    BoundaryType(bc_grp["bottom_type"][i].decode("utf-8")),
                    bc_grp["bottom_value"][i],
                ),
                top=BoundaryCondition(
                    BoundarySide.TOP,
                    BoundaryType(bc_grp["top_type"][i].decode("utf-8")),
                    bc_grp["top_value"][i],
                ),
            )
            s_id = grp["sample_id"][i] if "sample_id" in grp else str(i)
            s_id_str = s_id.decode("utf-8") if isinstance(s_id, bytes) else str(s_id)
            samples.append(
                {
                    "sample_id": s_id_str,
                    "k": k_all[i].astype(np.float64),
                    "q": q_all[i].astype(np.float64),
                    "t_true": t_all[i].astype(np.float64),
                    "bcs": bcs,
                }
            )
    return samples


def evaluate_samples_with_engine(
    engine: HybridInferenceEngine,
    samples: list[dict[str, Any]],
    options: InferenceOptions,
    grid: Grid2D,
) -> list[dict[str, Any]]:
    """Run hybrid engine on a list of samples and record granular diagnostic fields."""
    results = []

    for idx, s in enumerate(samples):
        k_field = s["k"]
        q_field = s["q"]
        t_true = s["t_true"]
        bcs = s["bcs"]

        req = InferenceRequest(
            grid=grid,
            conductivity=k_field,
            source=q_field,
            boundary_conditions=bcs,
            options=options,
        )

        res = engine.predict(req)

        # Also obtain raw surrogate prediction directly for audit comparison
        x_tensor = engine._construct_surrogate_input(grid, k_field, q_field, bcs)
        pred = engine.ensemble.predict(
            x_tensor,
            t_mean=engine.provenance.temperature_mean_kelvin,
            t_std=engine.provenance.temperature_std_kelvin,
        )
        y_surr_k = pred.mean_kelvin.squeeze().cpu().numpy().astype(np.float64)
        surr_diff = y_surr_k - t_true
        surr_rmse = float(np.sqrt(np.mean(surr_diff**2)))
        surr_rel_l2 = float(np.linalg.norm(surr_diff) / np.linalg.norm(t_true))
        surr_max_ae = float(np.max(np.abs(surr_diff)))

        # Output temperature error (from whatever temperature the engine delivered, if any)
        if res.temperature is not None:
            out_diff = res.temperature - t_true
            out_rmse = float(np.sqrt(np.mean(out_diff**2)))
            out_rel_l2 = float(np.linalg.norm(out_diff) / np.linalg.norm(t_true))
            out_max_ae = float(np.max(np.abs(out_diff)))
        else:
            out_rmse = None
            out_rel_l2 = None
            out_max_ae = None

        diag = res.diagnostics
        entry = {
            "sample_index": idx,
            "status": res.status.value,
            "is_accepted": res.is_accepted,
            "is_fallback": res.is_fallback,
            "is_failure": res.is_failure,
            "execution_path": res.execution_path,
            "rejection_code": res.rejection.code.value,
            "rejection_message": res.rejection.message,
            "mean_disagreement_k": res.mean_disagreement_kelvin,
            "max_disagreement_k": res.max_disagreement_kelvin,
            # Surrogate error metrics
            "surrogate_rmse_k": surr_rmse,
            "surrogate_rel_l2": surr_rel_l2,
            "surrogate_max_ae_k": surr_max_ae,
            # Output error metrics
            "output_rmse_k": out_rmse,
            "output_rel_l2": out_rel_l2,
            "output_max_ae_k": out_max_ae,
            # Physical diagnostics
            "pde_residual_l2_w": diag.pde_residual_l2_w if diag else None,
            "pde_residual_relative": diag.pde_residual_relative if diag else None,
            "pde_residual_linf_w": diag.pde_residual_linf_w if diag else None,
            "signed_net_imbalance_w": diag.signed_net_imbalance_w if diag else None,
            "absolute_flux_imbalance_w": diag.absolute_flux_imbalance_w if diag else None,
            "relative_flux_imbalance": diag.relative_flux_imbalance if diag else None,
            # Latency
            "latency_total_ms": res.latency.total_pipeline_ms,
            "latency_validation_ms": res.latency.input_validation_ms,
            "latency_surrogate_ms": res.latency.surrogate_inference_ms,
            "latency_diagnostics_ms": res.latency.diagnostics_evaluation_ms,
            "latency_solver_ms": res.latency.solver_fallback_ms,
            # Input summary
            "k_min": float(np.min(k_field)),
            "k_max": float(np.max(k_field)),
            "contrast_ratio": float(np.max(k_field) / np.min(k_field)),
            "q_max": float(np.max(np.abs(q_field))),
        }
        results.append(entry)

    return results


def summarize_partition_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistical summaries separated by acceptance status."""
    accepted = [r for r in records if r["is_accepted"]]
    rejected = [r for r in records if not r["is_accepted"]]

    def agg_stats(items: list[float]) -> dict[str, float] | None:
        valid = [x for x in items if x is not None and np.isfinite(x)]
        if not valid:
            return None
        arr = np.array(valid)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    # Count rejection codes
    rejection_counts: dict[str, int] = {}
    for r in rejected:
        code = r["rejection_code"]
        rejection_counts[code] = rejection_counts.get(code, 0) + 1

    return {
        "total_samples": len(records),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "acceptance_rate": len(accepted) / len(records) if records else 0.0,
        "rejection_reasons": rejection_counts,
        "accepted_subset": {
            "surrogate_rmse_k": agg_stats([r["surrogate_rmse_k"] for r in accepted]),
            "surrogate_rel_l2": agg_stats([r["surrogate_rel_l2"] for r in accepted]),
            "surrogate_max_ae_k": agg_stats([r["surrogate_max_ae_k"] for r in accepted]),
            "mean_disagreement_k": agg_stats([r["mean_disagreement_k"] for r in accepted]),
            "max_disagreement_k": agg_stats([r["max_disagreement_k"] for r in accepted]),
            "pde_residual_relative": agg_stats([r["pde_residual_relative"] for r in accepted]),
            "absolute_flux_imbalance_w": agg_stats(
                [r["absolute_flux_imbalance_w"] for r in accepted]
            ),
            "signed_net_imbalance_w": agg_stats([r["signed_net_imbalance_w"] for r in accepted]),
            "latency_total_ms": agg_stats([r["latency_total_ms"] for r in accepted]),
        },
        "rejected_subset": {
            "surrogate_rmse_k": agg_stats([r["surrogate_rmse_k"] for r in rejected]),
            "surrogate_rel_l2": agg_stats([r["surrogate_rel_l2"] for r in rejected]),
            "surrogate_max_ae_k": agg_stats([r["surrogate_max_ae_k"] for r in rejected]),
            "mean_disagreement_k": agg_stats([r["mean_disagreement_k"] for r in rejected]),
            "max_disagreement_k": agg_stats([r["max_disagreement_k"] for r in rejected]),
            "pde_residual_relative": agg_stats([r["pde_residual_relative"] for r in rejected]),
            "absolute_flux_imbalance_w": agg_stats(
                [r["absolute_flux_imbalance_w"] for r in rejected]
            ),
            "signed_net_imbalance_w": agg_stats([r["signed_net_imbalance_w"] for r in rejected]),
            "latency_total_ms": agg_stats([r["latency_total_ms"] for r in rejected]),
        },
        "all_samples": {
            "surrogate_rmse_k": agg_stats([r["surrogate_rmse_k"] for r in records]),
            "surrogate_rel_l2": agg_stats([r["surrogate_rel_l2"] for r in records]),
            "surrogate_max_ae_k": agg_stats([r["surrogate_max_ae_k"] for r in records]),
            "output_rmse_k": agg_stats(
                [r["output_rmse_k"] for r in records if r["output_rmse_k"] is not None]
            ),
            "output_max_ae_k": agg_stats(
                [r["output_max_ae_k"] for r in records if r["output_max_ae_k"] is not None]
            ),
        },
    }


def analyze_physics_and_uncertainty_correlations(
    cal_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Examine correlation between diagnostic metrics and surrogate error on calibration data."""
    metrics_to_correlate = [
        "mean_disagreement_k",
        "max_disagreement_k",
        "pde_residual_relative",
        "pde_residual_l2_w",
        "absolute_flux_imbalance_w",
        "relative_flux_imbalance",
        "contrast_ratio",
    ]
    target_errors = ["surrogate_max_ae_k", "surrogate_rmse_k", "surrogate_rel_l2"]

    correlations: dict[str, dict[str, Any]] = {}

    for metric in metrics_to_correlate:
        correlations[metric] = {}
        x_vals = np.array([r[metric] for r in cal_records if r[metric] is not None])
        for target in target_errors:
            y_vals = np.array([r[target] for r in cal_records if r[metric] is not None])
            if len(x_vals) > 2:
                p_r, p_p = stats.pearsonr(x_vals, y_vals)
                s_r, s_p = stats.spearmanr(x_vals, y_vals)
                correlations[metric][target] = {
                    "pearson_r": float(p_r),
                    "pearson_p": float(p_p),
                    "spearman_rho": float(s_r),
                    "spearman_p": float(s_p),
                }

    return correlations


def evaluate_candidate_tolerances_and_gates(
    cal_records: list[dict[str, Any]],
    candidate_tolerances_k: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate candidate error tolerances and screeners on the calibration partition."""
    if candidate_tolerances_k is None:
        candidate_tolerances_k = [1.0, 2.0, 5.0, 10.0]
    cal_results = {}

    for tol in candidate_tolerances_k:
        # Ground truth: violation occurs when MaxAE > tol
        y_true_viol = np.array([r["surrogate_max_ae_k"] > tol for r in cal_records], dtype=bool)
        n_violations = int(np.sum(y_true_viol))

        # 1. Baseline Screener (disagreement + domain contrast/source)
        # Accepted means screener flagged NO violation
        baseline_accepted = np.array([r["is_accepted"] for r in cal_records], dtype=bool)
        # False acceptance: surrogate accepted, but ground truth MaxAE > tol
        false_accept_baseline = int(np.sum(baseline_accepted & y_true_viol))
        false_reject_baseline = int(np.sum((~baseline_accepted) & (~y_true_viol)))
        true_accept_baseline = int(np.sum(baseline_accepted & (~y_true_viol)))
        true_reject_baseline = int(np.sum((~baseline_accepted) & y_true_viol))

        # Precision/recall for detecting violations (positive class = violation)
        # Model flags violation when ~accepted
        flagged_viol_baseline = ~baseline_accepted
        tp = int(np.sum(flagged_viol_baseline & y_true_viol))
        fp = int(np.sum(flagged_viol_baseline & (~y_true_viol)))
        fn = int(np.sum((~flagged_viol_baseline) & y_true_viol))

        precision_base = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall_base = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        far_base = float(false_accept_baseline / len(cal_records))

        # 2. Sweep candidate relative PDE residual gates (e.g. 0.05, 0.10, 0.15, 0.20)
        res_sweeps = {}
        for r_thresh in [0.02, 0.05, 0.10, 0.15, 0.20]:
            pde_accept = baseline_accepted & np.array(
                [
                    (
                        r["pde_residual_relative"] is not None
                        and r["pde_residual_relative"] <= r_thresh
                    )
                    for r in cal_records
                ]
            )
            fa = int(np.sum(pde_accept & y_true_viol))
            acc_rate = float(np.mean(pde_accept))
            flagged = ~pde_accept
            tp_r = int(np.sum(flagged & y_true_viol))
            fp_r = int(np.sum(flagged & (~y_true_viol)))
            fn_r = int(np.sum((~flagged) & y_true_viol))
            prec = float(tp_r / (tp_r + fp_r)) if (tp_r + fp_r) > 0 else 0.0
            rec = float(tp_r / (tp_r + fn_r)) if (tp_r + fn_r) > 0 else 0.0

            res_sweeps[f"rel_res_{r_thresh}"] = {
                "threshold": r_thresh,
                "acceptance_rate": acc_rate,
                "false_acceptances": fa,
                "false_acceptance_rate": float(fa / len(cal_records)),
                "precision": prec,
                "recall": rec,
            }

        # 3. Sweep candidate flux imbalance gates (e.g. 5W, 10W, 25W, 50W)
        flux_sweeps = {}
        for f_thresh in [2.0, 5.0, 10.0, 20.0, 50.0]:
            flux_accept = baseline_accepted & np.array(
                [
                    (
                        r["absolute_flux_imbalance_w"] is not None
                        and r["absolute_flux_imbalance_w"] <= f_thresh
                    )
                    for r in cal_records
                ]
            )
            fa = int(np.sum(flux_accept & y_true_viol))
            acc_rate = float(np.mean(flux_accept))
            flagged = ~flux_accept
            tp_f = int(np.sum(flagged & y_true_viol))
            fp_f = int(np.sum(flagged & (~y_true_viol)))
            fn_f = int(np.sum((~flagged) & y_true_viol))
            prec = float(tp_f / (tp_f + fp_f)) if (tp_f + fp_f) > 0 else 0.0
            rec = float(tp_f / (tp_f + fn_f)) if (tp_f + fn_f) > 0 else 0.0

            flux_sweeps[f"flux_thresh_{f_thresh}W"] = {
                "threshold_w": f_thresh,
                "acceptance_rate": acc_rate,
                "false_acceptances": fa,
                "false_acceptance_rate": float(fa / len(cal_records)),
                "precision": prec,
                "recall": rec,
            }

        cal_results[f"tolerance_{tol}K"] = {
            "error_tolerance_k": tol,
            "total_violations": n_violations,
            "violation_rate": float(n_violations / len(cal_records)),
            "baseline_screener": {
                "acceptance_rate": float(np.mean(baseline_accepted)),
                "false_acceptances": false_accept_baseline,
                "false_acceptance_rate": far_base,
                "true_acceptances": true_accept_baseline,
                "false_rejections": false_reject_baseline,
                "true_rejections": true_reject_baseline,
                "precision": precision_base,
                "recall": recall_base,
            },
            "pde_residual_gate_sweeps": res_sweeps,
            "flux_imbalance_gate_sweeps": flux_sweeps,
        }

    return cal_results


def benchmark_four_inference_modes(
    engine: HybridInferenceEngine,
    samples: list[dict[str, Any]],
    grid: Grid2D,
    n_benchmark: int = 50,
) -> dict[str, Any]:
    """Benchmark Direct FVM, Surrogate-Only, Hybrid Fail-Closed, and Hybrid Fallback."""
    subset = samples[:n_benchmark]
    fvm_solver = HeatConductionSolver2D()

    # 1. Direct FVM
    fvm_latencies = []
    for s in subset:
        t0 = time.perf_counter()
        fvm_solver.solve(grid, s["k"], s["q"], s["bcs"])
        fvm_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 2. Surrogate-Only (no validation, no screening, forward pass only)
    surr_latencies = []
    for s in subset:
        t0 = time.perf_counter()
        x = engine._construct_surrogate_input(grid, s["k"], s["q"], s["bcs"])
        pred = engine.ensemble.predict(
            x, engine.provenance.temperature_mean_kelvin, engine.provenance.temperature_std_kelvin
        )
        _ = pred.mean_kelvin.cpu().numpy()
        surr_latencies.append((time.perf_counter() - t0) * 1000.0)

    # 3. Hybrid Inference without Fallback (Screening enabled, fail-closed on rejection)
    opt_no_fb = InferenceOptions(enable_fallback=False)
    hybrid_no_fb_latencies = []
    no_fb_accepted = 0
    for s in subset:
        req = InferenceRequest(
            grid=grid,
            conductivity=s["k"],
            source=s["q"],
            boundary_conditions=s["bcs"],
            options=opt_no_fb,
        )
        t0 = time.perf_counter()
        res = engine.predict(req)
        hybrid_no_fb_latencies.append((time.perf_counter() - t0) * 1000.0)
        if res.is_accepted:
            no_fb_accepted += 1

    # 4. Hybrid Inference with Fallback (Full pipeline)
    opt_fb = InferenceOptions(enable_fallback=True)
    hybrid_fb_latencies = []
    fb_accepted = 0
    fb_fallback_count = 0
    for s in subset:
        req = InferenceRequest(
            grid=grid,
            conductivity=s["k"],
            source=s["q"],
            boundary_conditions=s["bcs"],
            options=opt_fb,
        )
        t0 = time.perf_counter()
        res = engine.predict(req)
        hybrid_fb_latencies.append((time.perf_counter() - t0) * 1000.0)
        if res.is_accepted:
            fb_accepted += 1
        elif res.is_fallback:
            fb_fallback_count += 1

    def stat_dict(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        return {
            "mean_ms": float(np.mean(a)),
            "std_ms": float(np.std(a, ddof=1)),
            "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "throughput_samples_per_sec": float(1000.0 / np.mean(a)),
        }

    return {
        "direct_fvm": stat_dict(fvm_latencies),
        "surrogate_only": stat_dict(surr_latencies),
        "hybrid_fail_closed": {
            **stat_dict(hybrid_no_fb_latencies),
            "acceptance_rate": float(no_fb_accepted / n_benchmark),
        },
        "hybrid_with_fallback": {
            **stat_dict(hybrid_fb_latencies),
            "acceptance_rate": float(fb_accepted / n_benchmark),
            "fallback_rate": float(fb_fallback_count / n_benchmark),
        },
    }


def main() -> None:
    print("=" * 80)
    print("Milestone 06C: Hybrid Acceptance Calibration and Safety Audit")
    print("=" * 80)

    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    engine = HybridInferenceEngine.from_artifacts(device="cpu")
    print(
        "HybridInferenceEngine successfully initialized and verified against frozen checkpoint hashes."
    )

    cal_path = Path("artifacts/data/calibration_dataset.h5")
    scaled_path = Path("artifacts/data/scaled_dataset.h5")

    print(f"Loading calibration partition from {cal_path}...")
    cal_samples = load_dataset_partition(cal_path, "calibration")
    print(f"Loaded {len(cal_samples)} calibration samples.")

    print(f"Loading test_iid and test_ood partitions from {scaled_path}...")
    iid_samples = load_dataset_partition(scaled_path, "test_iid")
    ood_samples = load_dataset_partition(scaled_path, "test_ood")
    print(f"Loaded {len(iid_samples)} test_iid and {len(ood_samples)} test_ood samples.")

    # -------------------------------------------------------------------------
    # Phase 2 & 3: Evaluate Baseline Engine on Calibration Partition
    # -------------------------------------------------------------------------
    print("\n[Phase 3] Evaluating Baseline Engine on Calibration Partition (200 samples)...")
    baseline_options = InferenceOptions(enable_fallback=True)
    cal_eval = evaluate_samples_with_engine(engine, cal_samples, baseline_options, grid)
    cal_summary = summarize_partition_metrics(cal_eval)
    print(
        f"Calibration Acceptance Rate: {cal_summary['acceptance_rate'] * 100:.1f}% ({cal_summary['accepted_count']}/{cal_summary['total_samples']})"
    )

    # Physics correlations
    print("Computing correlations between diagnostic indicators and actual prediction errors...")
    correlations = analyze_physics_and_uncertainty_correlations(cal_eval)

    # Candidate tolerance analysis
    print("Evaluating candidate error tolerances (1K, 2K, 5K, 10K) and physics gate sweeps...")
    tolerance_analysis = evaluate_candidate_tolerances_and_gates(cal_eval)

    # -------------------------------------------------------------------------
    # Phase 2: Evaluate on Test Partitions (test_iid and test_ood)
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Evaluating Baseline Engine on test_iid (200 samples)...")
    iid_eval = evaluate_samples_with_engine(engine, iid_samples, baseline_options, grid)
    iid_summary = summarize_partition_metrics(iid_eval)
    print(
        f"test_iid Acceptance Rate: {iid_summary['acceptance_rate'] * 100:.1f}% ({iid_summary['accepted_count']}/{iid_summary['total_samples']})"
    )

    print("\n[Phase 2] Evaluating Baseline Engine on test_ood (200 samples)...")
    ood_eval = evaluate_samples_with_engine(engine, ood_samples, baseline_options, grid)
    ood_summary = summarize_partition_metrics(ood_eval)
    print(
        f"test_ood Acceptance Rate: {ood_summary['acceptance_rate'] * 100:.1f}% ({ood_summary['accepted_count']}/{ood_summary['total_samples']})"
    )
    print("test_ood Rejection Reasons:", ood_summary["rejection_reasons"])

    # -------------------------------------------------------------------------
    # Calibrated Gate Out-of-Sample Verification
    # -------------------------------------------------------------------------
    print("\nEvaluating Calibrated Physics-Gated Engine (rel_residual <= 0.10, flux <= 10.0W)...")
    calibrated_options = InferenceOptions(
        enable_fallback=True,
        relative_pde_residual_threshold=0.10,
        boundary_flux_imbalance_threshold_w=10.0,
    )
    iid_calibrated_eval = evaluate_samples_with_engine(
        engine, iid_samples, calibrated_options, grid
    )
    ood_calibrated_eval = evaluate_samples_with_engine(
        engine, ood_samples, calibrated_options, grid
    )
    iid_calibrated_summary = summarize_partition_metrics(iid_calibrated_eval)
    ood_calibrated_summary = summarize_partition_metrics(ood_calibrated_eval)
    print(f"IID Calibrated Acceptance Rate: {iid_calibrated_summary['acceptance_rate'] * 100:.1f}%")
    print(f"OOD Calibrated Acceptance Rate: {ood_calibrated_summary['acceptance_rate'] * 100:.1f}%")

    # -------------------------------------------------------------------------
    # Phase 5: Benchmark Four Inference Modes
    # -------------------------------------------------------------------------
    print(
        "\n[Phase 5] Benchmarking Direct FVM, Surrogate-Only, Hybrid Fail-Closed, Hybrid Fallback..."
    )
    benchmark_results_iid = benchmark_four_inference_modes(
        engine, iid_samples, grid, n_benchmark=50
    )
    benchmark_results_ood = benchmark_four_inference_modes(
        engine, ood_samples, grid, n_benchmark=50
    )

    # -------------------------------------------------------------------------
    # Package and Save Results JSON
    # -------------------------------------------------------------------------
    full_results = {
        "metadata": {
            "milestone": "06C",
            "title": "Hybrid Acceptance Calibration and Safety Audit",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "grid": "32x32",
            "device": "cpu",
        },
        "phase2_acceptance_evaluation": {
            "test_iid": {
                "summary": iid_summary,
                "per_sample_records": iid_eval,
            },
            "test_ood": {
                "summary": ood_summary,
                "per_sample_records": ood_eval,
            },
        },
        "phase3_physics_calibration": {
            "calibration_partition_summary": cal_summary,
            "correlations": correlations,
            "candidate_tolerances": tolerance_analysis,
            "out_of_sample_verification": {
                "test_iid_calibrated": iid_calibrated_summary,
                "test_ood_calibrated": ood_calibrated_summary,
            },
        },
        "phase5_benchmarks": {
            "test_iid_benchmark": benchmark_results_iid,
            "test_ood_benchmark": benchmark_results_ood,
        },
    }

    out_file = Path("artifacts/reports/milestone_06c_acceptance_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)

    print(f"\nAll results saved to {out_file} ({out_file.stat().st_size / 1024:.1f} KB).")

    # Print summary highlights
    print("\n" + "=" * 80)
    print("MILESTONE 06C AUDIT SUMMARY HIGHLIGHTS")
    print("=" * 80)
    print(
        f"IID Test Partition: {iid_summary['acceptance_rate'] * 100:.1f}% Accepted ({iid_summary['accepted_count']}/200)"
    )
    if iid_summary["accepted_subset"]["surrogate_max_ae_k"]:
        print(
            f"  - Accepted MaxAE: {iid_summary['accepted_subset']['surrogate_max_ae_k']['mean']:.2f} ± {iid_summary['accepted_subset']['surrogate_max_ae_k']['std']:.2f} K (p95: {iid_summary['accepted_subset']['surrogate_max_ae_k']['p95']:.2f} K)"
        )
        print(
            f"  - Accepted RMSE:  {iid_summary['accepted_subset']['surrogate_rmse_k']['mean']:.2f} ± {iid_summary['accepted_subset']['surrogate_rmse_k']['std']:.2f} K"
        )
    print(
        f"OOD Test Partition: {ood_summary['acceptance_rate'] * 100:.1f}% Accepted ({ood_summary['accepted_count']}/200)"
    )
    print(f"  - Rejection breakdown: {ood_summary['rejection_reasons']}")
    print("\nBenchmark Latency (IID):")
    print(
        f"  - Direct FVM:         {benchmark_results_iid['direct_fvm']['mean_ms']:.2f} ms ({benchmark_results_iid['direct_fvm']['throughput_samples_per_sec']:.1f} samp/s)"
    )
    print(
        f"  - Surrogate-Only:     {benchmark_results_iid['surrogate_only']['mean_ms']:.2f} ms ({benchmark_results_iid['surrogate_only']['throughput_samples_per_sec']:.1f} samp/s)"
    )
    print(
        f"  - Hybrid Fail-Closed: {benchmark_results_iid['hybrid_fail_closed']['mean_ms']:.2f} ms ({benchmark_results_iid['hybrid_fail_closed']['throughput_samples_per_sec']:.1f} samp/s)"
    )
    print(
        f"  - Hybrid Fallback:    {benchmark_results_iid['hybrid_with_fallback']['mean_ms']:.2f} ms ({benchmark_results_iid['hybrid_with_fallback']['throughput_samples_per_sec']:.1f} samp/s)"
    )


if __name__ == "__main__":
    main()
