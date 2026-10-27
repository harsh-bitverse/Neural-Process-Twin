"""Independent Confirmation Evaluation for Milestone 08E (Phase 7).

Evaluates frozen routing policies and thresholds against the untouched,
locked independent confirmation cohort (artifacts/data/independent_cohort_08e.h5):
- seeds 800,000 - 800,099: eval_iid
- seeds 810,000 - 810,099: eval_ood
Across resolutions: 32x32, 64x64, 128x128, 256x256.

Strict Governance:
- Evaluated strictly ONCE with all models, thresholds, and policies frozen.
- Computes Wilson score 95% confidence intervals on failure probabilities.
- Reports failures without post-hoc retuning.
"""

from __future__ import annotations

import hashlib
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
logger = logging.getLogger("independent_confirmation_08e")


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Compute point estimate and Wilson score confidence interval."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p_hat = float(k / n)
    z = 1.959963984540054  # 95% two-sided normal quantile
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + ((z**2) / (4 * n**2)))) / denom
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    return p_hat, ci_lower, ci_upper


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


def run_independent_confirmation() -> dict[str, Any]:
    """Execute final independent confirmation audit."""
    cohort_path = Path("artifacts/data/independent_cohort_08e.h5")
    manifest_path = Path("artifacts/data/independent_cohort_08e_manifest.json")
    thresholds_path = Path("artifacts/reports/milestone_08e_calibrated_thresholds.json")

    logger.info("Verifying cohort provenance and manifest...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # SHA256 check
    hasher = hashlib.sha256()
    with open(cohort_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest()

    if actual_sha != manifest["sha256"]:
        raise ValueError(
            f"Integrity violation! File SHA-256 {actual_sha} does not match manifest {manifest['sha256']}"
        )
    logger.info(f"Cohort integrity verified (SHA-256: {actual_sha})")

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

    confirmation_results: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cohort_file": str(cohort_path),
            "cohort_sha256": actual_sha,
            "governance_status": "FROZEN_CONFIRMATION_EVALUATION",
            "thresholds": thresh_dict,
        },
        "confirmation_evaluations": {},
    }

    with h5py.File(cohort_path, "r") as f:
        for res in resolutions:
            res_key = f"res_{res}"
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
            logger.info(f"=== Confirmation Evaluation at {res}x{res} ===")

            confirmation_results["confirmation_evaluations"][res_key] = {}
            res_grp = f[res_key]

            iid_grp = res_grp["eval_iid"]
            ood_grp = res_grp["eval_ood"]

            n_iid = iid_grp["temperature"].shape[0]
            n_ood = ood_grp["temperature"].shape[0]

            iid_k = iid_grp["k"][:]
            iid_q = iid_grp["q"][:]
            iid_t = iid_grp["temperature"][:]
            iid_bcs = [parse_bcs_row(iid_grp, i) for i in range(n_iid)]

            ood_k = ood_grp["k"][:]
            ood_q = ood_grp["q"][:]
            ood_t = ood_grp["temperature"][:]
            ood_subgroups = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in ood_grp["subgroup"][:]
            ]
            ood_bcs = [parse_bcs_row(ood_grp, i) for i in range(n_ood)]

            # Baseline FVM latency
            fvm_lats = []
            for i in range(min(20, n_iid)):
                r0_res = engine.route_inference(
                    grid, iid_k[i], iid_q[i], iid_bcs[i], policy=RoutingPolicy.R0_DIRECT_FVM
                )
                fvm_lats.append(r0_res.timings_ms["t_total"])
            baseline_fvm_ms = float(np.mean(fvm_lats))

            for pol in policies:
                pol_name = pol.value
                logger.info(f"  Evaluating {pol_name}...")

                # Warm-up
                _ = engine.route_inference(grid, iid_k[0], iid_q[0], iid_bcs[0], policy=pol)

                # 1. Eval IID
                iid_lats = []
                iid_accepted = 0
                iid_false_10k = 0
                iid_false_5k = 0
                iid_acc_maxaes = []
                iid_acc_rmses = []
                iid_hybrid_rmses = []

                for i in range(n_iid):
                    res_obj = engine.route_inference(
                        grid, iid_k[i], iid_q[i], iid_bcs[i], policy=pol
                    )
                    iid_lats.append(res_obj.timings_ms["t_total"])

                    err = np.abs(res_obj.temperature - iid_t[i])
                    maxae = float(np.max(err))
                    rmse = float(np.sqrt(np.mean(err**2)))
                    iid_hybrid_rmses.append(rmse)

                    if res_obj.status == InferenceStatus.ACCEPTED:
                        iid_accepted += 1
                        iid_acc_maxaes.append(maxae)
                        iid_acc_rmses.append(rmse)
                        if maxae > 10.0:
                            iid_false_10k += 1
                        if maxae > 5.0:
                            iid_false_5k += 1

                # 2. Eval OOD
                ood_lats = []
                ood_accepted = 0
                ood_false_10k = 0
                ood_false_5k = 0
                ood_acc_maxaes = []
                ood_acc_rmses = []
                ood_hybrid_rmses = []
                ood_escalations = 0

                hc_acc = 0
                hc_tot = 0
                grf_acc = 0
                grf_tot = 0

                for i in range(n_ood):
                    res_obj = engine.route_inference(
                        grid, ood_k[i], ood_q[i], ood_bcs[i], policy=pol
                    )
                    ood_lats.append(res_obj.timings_ms["t_total"])

                    err = np.abs(res_obj.temperature - ood_t[i])
                    maxae = float(np.max(err))
                    rmse = float(np.sqrt(np.mean(err**2)))
                    ood_hybrid_rmses.append(rmse)

                    is_hc = "inclusion" in ood_subgroups[i].lower()
                    if is_hc:
                        hc_tot += 1
                    else:
                        grf_tot += 1

                    if res_obj.status == InferenceStatus.ACCEPTED:
                        ood_accepted += 1
                        ood_acc_maxaes.append(maxae)
                        ood_acc_rmses.append(rmse)
                        if is_hc:
                            hc_acc += 1
                        else:
                            grf_acc += 1
                        if maxae > 10.0:
                            ood_false_10k += 1
                        if maxae > 5.0:
                            ood_false_5k += 1
                    else:
                        ood_escalations += 1

                # Statistical Wilson score confidence intervals
                p_hat_iid_10k, iid_ci_l_10k, iid_ci_u_10k = compute_wilson_ci(
                    iid_false_10k, max(iid_accepted, 1)
                )
                p_hat_ood_10k, ood_ci_l_10k, ood_ci_u_10k = compute_wilson_ci(
                    ood_false_10k, max(ood_accepted, 1)
                )

                iid_lat_mean = float(np.mean(iid_lats))
                ood_lat_mean = float(np.mean(ood_lats))
                mix_lat_mean = 0.80 * iid_lat_mean + 0.20 * ood_lat_mean
                mix_rmse = float(
                    0.80 * np.mean(iid_hybrid_rmses) + 0.20 * np.mean(ood_hybrid_rmses)
                )
                speedup = float(baseline_fvm_ms / max(mix_lat_mean, 1e-4))

                all_lats = iid_lats + ood_lats

                confirmation_results["confirmation_evaluations"][res_key][pol_name] = {
                    "latency_ms": {
                        "mean": float(np.mean(all_lats)),
                        "p50": float(np.percentile(all_lats, 50)),
                        "p95": float(np.percentile(all_lats, 95)),
                        "iid_mean": iid_lat_mean,
                        "ood_mean": ood_lat_mean,
                        "mix_mean": mix_lat_mean,
                        "speedup_vs_fvm": speedup,
                    },
                    "eval_iid": {
                        "n_samples": n_iid,
                        "acceptance_rate_pct": (iid_accepted / n_iid) * 100.0,
                        "accepted_rmse_k": float(np.mean(iid_acc_rmses)) if iid_acc_rmses else 0.0,
                        "accepted_maxae_k": float(np.mean(iid_acc_maxaes))
                        if iid_acc_maxaes
                        else 0.0,
                        "hybrid_rmse_k": float(np.mean(iid_hybrid_rmses)),
                        "false_accept_count_10k": iid_false_10k,
                        "false_accept_rate_10k": p_hat_iid_10k,
                        "false_accept_ci_10k": [iid_ci_l_10k, iid_ci_u_10k],
                    },
                    "eval_ood": {
                        "n_samples": n_ood,
                        "acceptance_rate_pct": (ood_accepted / n_ood) * 100.0,
                        "escalation_rate_pct": (ood_escalations / n_ood) * 100.0,
                        "accepted_rmse_k": float(np.mean(ood_acc_rmses)) if ood_acc_rmses else 0.0,
                        "accepted_maxae_k": float(np.mean(ood_acc_maxaes))
                        if ood_acc_maxaes
                        else 0.0,
                        "hybrid_rmse_k": float(np.mean(ood_hybrid_rmses)),
                        "false_accept_count_10k": ood_false_10k,
                        "false_accept_rate_10k": p_hat_ood_10k,
                        "false_accept_ci_10k": [ood_ci_l_10k, ood_ci_u_10k],
                        "subgroup_acceptance": {
                            "high_contrast_pct": (hc_acc / max(hc_tot, 1)) * 100.0,
                            "grf_pct": (grf_acc / max(grf_tot, 1)) * 100.0,
                        },
                    },
                    "deployment_mixture": {
                        "acceptance_rate_pct": (
                            0.80 * (iid_accepted / n_iid) + 0.20 * (ood_accepted / n_ood)
                        )
                        * 100.0,
                        "hybrid_rmse_k": mix_rmse,
                        "speedup_vs_fvm": speedup,
                    },
                }

    out_file = Path("artifacts/reports/milestone_08e_independent_confirmation.json")
    with open(out_file, "w") as f:
        json.dump(confirmation_results, f, indent=2)
    logger.info(f"Independent confirmation audit saved to {out_file}")
    return confirmation_results


if __name__ == "__main__":
    conf = run_independent_confirmation()
    print("\n=== CONFIRMATION AUDIT SUMMARY (res_256) ===")
    r256 = conf["confirmation_evaluations"]["res_256"]
    for pol_k, data in r256.items():
        mix = data["deployment_mixture"]
        lat = data["latency_ms"]
        ood = data["eval_ood"]
        print(
            f"{pol_k:25s} | Speedup: {mix['speedup_vs_fvm']:5.2f}x | Mix Lat: {lat['mix_mean']:6.2f} ms | Hybrid RMSE: {mix['hybrid_rmse_k']:5.2f} K | OOD Esc: {ood['escalation_rate_pct']:5.1f}% | OOD FA: {ood['false_accept_count_10k']}"
        )
