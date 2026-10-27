"""Comprehensive evaluation script for Milestone 05B.

Executes:
1. Ensemble checkpoint provenance and SHA-256 hash verification.
2. Inference on calibration dataset, validation dataset, test_iid, and test_ood.
3. Conformal interval recalibration on independent calibration partition vs validation split.
4. Frozen-threshold OOD detection evaluation (calibration-calibrated & validation-calibrated).
5. Mechanistic OOD breakdown by shift regime:
   - High conductivity contrast (>= 20:1)
   - Moderate contrast / fine GRF (< 20:1)
   - High heat source (> 1500 W/m^3)
   - Shifted boundary conditions (outside IID range)
   - Compound shifts (High contrast + High source, High contrast + Shifted BC, Triple compound)
6. Structured JSON export to artifacts/reports/milestone_05b_calibration_results.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_process_twin.data.boundary_encoding import decode_boundary_conditions
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.evaluation.metrics import (
    compute_aggregate_statistics,
    compute_all_metrics,
)
from neural_process_twin.evaluation.residual import evaluate_physical_residuals
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.uncertainty.calibration import ConformalIntervalCalibrator
from neural_process_twin.uncertainty.ensemble import DeepEnsemble
from neural_process_twin.uncertainty.metrics import (
    compute_risk_coverage,
    compute_uncertainty_quality,
)
from neural_process_twin.uncertainty.ood import (
    EnsembleDisagreementDetector,
    InputDistributionDetector,
    evaluate_ood_detection,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_checkpoint_provenance(checkpoint_paths: list[Path]) -> list[dict[str, Any]]:
    """Compute SHA-256 hashes and metadata for frozen ensemble checkpoints."""
    provenance = []
    for p in checkpoint_paths:
        p = Path(p)
        with open(p, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", -1)
        val_loss = float(ckpt.get("val_loss", ckpt.get("loss", -1.0)))
        provenance.append(
            {
                "path": str(p),
                "sha256": digest,
                "size_bytes": p.stat().st_size,
                "epoch": epoch,
                "val_loss": val_loss,
                "architecture": "fno2d",
            }
        )
    return provenance


def evaluate_split(
    ensemble: DeepEnsemble,
    dataset_path: Path,
    split_name: str,
) -> dict[str, Any]:
    """Run ensemble inference, compute error metrics, residuals, and anomaly scores."""
    ds = HeatFieldDataset(
        file_path=dataset_path,
        split=split_name,
        include_coords=True,
        include_bcs=True,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=ds.nx, ny=ds.ny)

    mean_preds_list = []
    std_preds_list = []
    true_targets_list = []
    sample_metrics_list = []
    residual_metrics_list = []

    mean_s_t_list = []
    max_s_t_list = []
    k_contrast_list = []
    q_max_list = []
    k_raw_list = []
    q_raw_list = []
    bcs_shift_flags = []
    sample_ids = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"]
            pred = ensemble.predict(x)

            y_pred_k = pred.mean_kelvin.squeeze(0).squeeze(0).numpy()
            y_std_k = pred.std_kelvin.squeeze(0).squeeze(0).numpy()
            t_ref_k = batch["temperature_raw"].squeeze(0).squeeze(0).numpy()

            mean_preds_list.append(y_pred_k)
            std_preds_list.append(y_std_k)
            true_targets_list.append(t_ref_k)

            m = compute_all_metrics(y_pred_k, t_ref_k)
            sample_metrics_list.append(m)

            mean_s_t_list.append(float(np.mean(y_std_k)))
            max_s_t_list.append(float(np.max(y_std_k)))

            k_raw = batch["k_raw"].squeeze(0).squeeze(0).numpy()
            q_raw = batch["q_raw"].squeeze(0).squeeze(0).numpy()
            k_raw_list.append(k_raw)
            q_raw_list.append(q_raw)

            k_contrast_list.append(float(np.max(k_raw) / max(np.min(k_raw), 1e-6)))
            q_max_list.append(float(np.max(q_raw)))
            sample_ids.append(f"{split_name}_{batch_idx:05d}")

            # Physical residuals
            x_np = x.squeeze(0).numpy()
            bcs = decode_boundary_conditions(x_np, grid=grid, normalizer=ds.normalizer)
            res_diag = evaluate_physical_residuals(
                grid=grid,
                k_field=k_raw,
                q_field=q_raw,
                bcs=bcs,
                temperature_pred=y_pred_k,
            )
            residual_metrics_list.append(res_diag)

            # Check if BC values fall outside IID range
            is_shifted = False
            for edge in ("left", "right", "top", "bottom"):
                edge_bc = getattr(bcs, edge)
                val_arr = edge_bc.value
                if (
                    edge_bc.bc_type == "dirichlet"
                    and (np.min(val_arr) < 290.0 or np.max(val_arr) > 360.0)
                ) or (edge_bc.bc_type == "neumann" and np.max(np.abs(val_arr)) > 300.0):
                    is_shifted = True
            bcs_shift_flags.append(is_shifted)

    agg_err = compute_aggregate_statistics(sample_metrics_list)
    pde_res_vals = [r["pde_residual_l2_w"] for r in residual_metrics_list]
    flux_imb_vals = [r["flux_imbalance_w"] for r in residual_metrics_list]

    sample_rmse = np.array([m["rmse_k"] for m in sample_metrics_list])
    sample_rel_l2 = np.array([m["relative_l2"] * 100.0 for m in sample_metrics_list])
    mean_s_t_arr = np.array(mean_s_t_list)
    max_s_t_arr = np.array(max_s_t_list)

    rc_result = compute_risk_coverage(sample_rmse, mean_s_t_arr, sample_rel_l2, num_thresholds=10)
    uq_metrics = compute_uncertainty_quality(sample_rmse, sample_rel_l2, mean_s_t_arr, max_s_t_arr)

    return {
        "sample_ids": sample_ids,
        "mean_preds": np.stack(mean_preds_list, axis=0),
        "std_preds": np.stack(std_preds_list, axis=0),
        "true_targets": np.stack(true_targets_list, axis=0),
        "k_raw": np.stack(k_raw_list, axis=0),
        "q_raw": np.stack(q_raw_list, axis=0),
        "sample_rmse": sample_rmse,
        "sample_rel_l2": sample_rel_l2,
        "sample_metrics": sample_metrics_list,
        "residual_metrics": residual_metrics_list,
        "mean_s_t": mean_s_t_arr,
        "max_s_t": max_s_t_arr,
        "k_contrast": np.array(k_contrast_list),
        "q_max": np.array(q_max_list),
        "bcs_shift_flags": np.array(bcs_shift_flags, dtype=bool),
        "aggregate_error": {
            "relative_l2_mean": agg_err["relative_l2"]["mean"],
            "relative_l2_std": agg_err["relative_l2"]["std"],
            "rmse_kelvin_mean": agg_err["rmse_k"]["mean"],
            "rmse_kelvin_std": agg_err["rmse_k"]["std"],
            "mae_kelvin_mean": agg_err["mae_k"]["mean"],
            "mae_kelvin_std": agg_err["mae_k"]["std"],
            "max_error_kelvin_mean": agg_err["max_error_k"]["mean"],
            "max_error_kelvin_std": agg_err["max_error_k"]["std"],
            "pde_residual_w_mean": float(np.mean(pde_res_vals)),
            "pde_residual_w_std": float(np.std(pde_res_vals)),
            "flux_imbalance_w_mean": float(np.mean(flux_imb_vals)),
            "flux_imbalance_w_std": float(np.std(flux_imb_vals)),
        },
        "uncertainty_quality": {
            "mean_uncertainty_k": uq_metrics.mean_uncertainty_k,
            "max_uncertainty_k": uq_metrics.max_uncertainty_k,
            "pearson_r_rmse": uq_metrics.pearson_r_rmse,
            "spearman_rho_rmse": uq_metrics.spearman_rho_rmse,
            "pearson_r_rel_l2": uq_metrics.pearson_r_rel_l2,
            "spearman_rho_rel_l2": uq_metrics.spearman_rho_rel_l2,
            "worst_10pct_detection_auroc": uq_metrics.worst_10pct_detection_auroc,
            "worst_20pct_detection_auroc": uq_metrics.worst_20pct_detection_auroc,
        },
        "risk_coverage": {
            "coverages": rc_result.coverages.tolist(),
            "rmse_at_coverage": rc_result.rmse_at_coverage.tolist(),
            "rel_l2_at_coverage": rc_result.rel_l2_at_coverage.tolist(),
            "aurc": rc_result.aurc,
        },
    }


def run_milestone_05b_pipeline(
    checkpoint_paths: list[Path],
    calibration_dataset_path: Path = Path("artifacts/data/calibration_dataset.h5"),
    scaled_dataset_path: Path = Path("artifacts/data/scaled_dataset.h5"),
    output_report_path: Path = Path("artifacts/reports/milestone_05b_calibration_results.json"),
) -> dict[str, Any]:
    """Execute complete Milestone 05B evaluation pipeline."""
    logger.info("Step 1: Verifying ensemble checkpoint provenance and hashes...")
    provenance = compute_checkpoint_provenance(checkpoint_paths)
    for p in provenance:
        logger.info(
            f"  {Path(p['path']).parent.name}: sha256={p['sha256'][:16]}..., epoch={p['epoch']}, val_loss={p['val_loss']:.6f}"
        )

    logger.info("Loading frozen DeepEnsemble...")
    ensemble = DeepEnsemble.from_checkpoints(checkpoint_paths, device="cpu")

    # Step 2: Evaluate on all splits
    logger.info("Step 2: Evaluating ensemble on independent calibration dataset (200 samples)...")
    cal_data = evaluate_split(ensemble, calibration_dataset_path, "calibration")

    logger.info("Evaluating ensemble on validation split (150 samples)...")
    val_data = evaluate_split(ensemble, scaled_dataset_path, "validation")

    logger.info("Evaluating ensemble on test_iid split (200 samples)...")
    iid_data = evaluate_split(ensemble, scaled_dataset_path, "test_iid")

    logger.info("Evaluating ensemble on test_ood split (200 samples)...")
    ood_data = evaluate_split(ensemble, scaled_dataset_path, "test_ood")

    # Step 3: Conformal Calibration
    logger.info("Step 3: Recalibrating Conformal Prediction Intervals...")
    # Calibrate on independent calibration dataset vs validation split
    conformal_results: dict[str, Any] = {
        "calibration_source": {
            "independent_calibration": {},
            "validation_split_reference": {},
        }
    }

    cal_sources = [
        ("independent_calibration", cal_data),
        ("validation_split_reference", val_data),
    ]

    for src_name, src_split in cal_sources:
        src_mean = src_split["mean_preds"]
        src_std = src_split["std_preds"]
        src_true = src_split["true_targets"]
        src_results = {}

        for conf_level in [0.90, 0.95]:
            for cov_type in ["simultaneous", "pointwise"]:
                calibrator = ConformalIntervalCalibrator(
                    confidence_level=conf_level, coverage_type=cov_type
                )
                q_hat = calibrator.fit(src_mean, src_std, src_true)
                key = f"{cov_type}_{int(conf_level * 100)}"

                eval_cal = calibrator.evaluate(
                    cal_data["mean_preds"], cal_data["std_preds"], cal_data["true_targets"]
                )
                eval_val = calibrator.evaluate(
                    val_data["mean_preds"], val_data["std_preds"], val_data["true_targets"]
                )
                eval_iid = calibrator.evaluate(
                    iid_data["mean_preds"], iid_data["std_preds"], iid_data["true_targets"]
                )
                eval_ood = calibrator.evaluate(
                    ood_data["mean_preds"], ood_data["std_preds"], ood_data["true_targets"]
                )

                src_results[key] = {
                    "confidence_level": conf_level,
                    "coverage_type": cov_type,
                    "quantile_multiplier_q_hat": q_hat,
                    "calibration_scores_count": len(calibrator.calibration_scores),
                    "quantile_index": int(
                        np.ceil(
                            min(
                                1.0,
                                np.ceil((len(calibrator.calibration_scores) + 1) * conf_level)
                                / len(calibrator.calibration_scores),
                            )
                            * len(calibrator.calibration_scores)
                        )
                    )
                    - 1,
                    "splits": {
                        "calibration": eval_cal.__dict__,
                        "validation": eval_val.__dict__,
                        "test_iid": eval_iid.__dict__,
                        "test_ood": eval_ood.__dict__,
                    },
                }

        conformal_results["calibration_source"][src_name] = src_results

    # Step 4: Validate OOD Detector Thresholding
    logger.info("Step 4: Validating OOD Detector Thresholding with Frozen Operating Points...")
    det_ens_mean = EnsembleDisagreementDetector(aggregation="mean")
    det_ens_max = EnsembleDisagreementDetector(aggregation="max")
    det_k_contrast = InputDistributionDetector(mode="contrast")
    det_q_max = InputDistributionDetector(mode="heat_source")
    det_comp = InputDistributionDetector(mode="composite")

    # Compute anomaly scores for all splits
    scores_dict = {
        "ensemble_disagreement_mean": {
            "cal": det_ens_mean.score(cal_data["std_preds"]),
            "val": det_ens_mean.score(val_data["std_preds"]),
            "iid": det_ens_mean.score(iid_data["std_preds"]),
            "ood": det_ens_mean.score(ood_data["std_preds"]),
        },
        "ensemble_disagreement_max": {
            "cal": det_ens_max.score(cal_data["std_preds"]),
            "val": det_ens_max.score(val_data["std_preds"]),
            "iid": det_ens_max.score(iid_data["std_preds"]),
            "ood": det_ens_max.score(ood_data["std_preds"]),
        },
        "input_k_contrast": {
            "cal": det_k_contrast.score(cal_data["k_raw"], cal_data["q_raw"]),
            "val": det_k_contrast.score(val_data["k_raw"], val_data["q_raw"]),
            "iid": det_k_contrast.score(iid_data["k_raw"], iid_data["q_raw"]),
            "ood": det_k_contrast.score(ood_data["k_raw"], ood_data["q_raw"]),
        },
        "input_q_max": {
            "cal": det_q_max.score(cal_data["k_raw"], cal_data["q_raw"]),
            "val": det_q_max.score(val_data["k_raw"], val_data["q_raw"]),
            "iid": det_q_max.score(iid_data["k_raw"], iid_data["q_raw"]),
            "ood": det_q_max.score(ood_data["k_raw"], ood_data["q_raw"]),
        },
        "input_composite": {
            "cal": det_comp.score(cal_data["k_raw"], cal_data["q_raw"]),
            "val": det_comp.score(val_data["k_raw"], val_data["q_raw"]),
            "iid": det_comp.score(iid_data["k_raw"], iid_data["q_raw"]),
            "ood": det_comp.score(ood_data["k_raw"], ood_data["q_raw"]),
        },
    }

    ood_eval_results = {}
    for det_name, s_data in scores_dict.items():
        # Diagnostic test-set evaluation (sweeps threshold across combined test set)
        diag_res = evaluate_ood_detection(s_data["iid"], s_data["ood"], detector_name=det_name)

        # Frozen threshold selection using INDEPENDENT CALIBRATION split (target 5% FPR -> 95th percentile)
        thresh_frozen_cal = float(np.percentile(s_data["cal"], 95.0))
        # Frozen threshold selection using VALIDATION split (target 5% FPR -> 95th percentile)
        thresh_frozen_val = float(np.percentile(s_data["val"], 95.0))

        # Evaluate frozen threshold from CALIBRATION on test sets
        empirical_iid_fpr_cal = float(np.mean(s_data["iid"] >= thresh_frozen_cal))
        empirical_ood_tpr_cal = float(np.mean(s_data["ood"] >= thresh_frozen_cal))
        pred_iid_cal = s_data["iid"] >= thresh_frozen_cal
        pred_ood_cal = s_data["ood"] >= thresh_frozen_cal
        acc_cal = float(
            (np.sum(~pred_iid_cal) + np.sum(pred_ood_cal)) / (len(pred_iid_cal) + len(pred_ood_cal))
        )
        prec_cal = float(np.sum(pred_ood_cal) / max(np.sum(pred_ood_cal) + np.sum(pred_iid_cal), 1))
        rec_cal = empirical_ood_tpr_cal
        f1_cal = float(2 * prec_cal * rec_cal / max(prec_cal + rec_cal, 1e-12))

        # Evaluate frozen threshold from VALIDATION on test sets
        empirical_iid_fpr_val = float(np.mean(s_data["iid"] >= thresh_frozen_val))
        empirical_ood_tpr_val = float(np.mean(s_data["ood"] >= thresh_frozen_val))
        pred_iid_val = s_data["iid"] >= thresh_frozen_val
        pred_ood_val = s_data["ood"] >= thresh_frozen_val
        acc_val = float(
            (np.sum(~pred_iid_val) + np.sum(pred_ood_val)) / (len(pred_iid_val) + len(pred_ood_val))
        )
        prec_val = float(np.sum(pred_ood_val) / max(np.sum(pred_ood_val) + np.sum(pred_iid_val), 1))
        rec_val = empirical_ood_tpr_val
        f1_val = float(2 * prec_val * rec_val / max(prec_val + rec_val, 1e-12))

        ood_eval_results[det_name] = {
            "auroc": diag_res.auroc,
            "auprc": diag_res.auprc,
            "frozen_operating_point_calibration": {
                "frozen_threshold": thresh_frozen_cal,
                "nominal_cal_fpr": 0.05,
                "empirical_test_iid_fpr": empirical_iid_fpr_cal,
                "empirical_test_ood_tpr": empirical_ood_tpr_cal,
                "test_accuracy": acc_cal,
                "test_f1": f1_cal,
            },
            "frozen_operating_point_validation": {
                "frozen_threshold": thresh_frozen_val,
                "nominal_val_fpr": 0.05,
                "empirical_test_iid_fpr": empirical_iid_fpr_val,
                "empirical_test_ood_tpr": empirical_ood_tpr_val,
                "test_accuracy": acc_val,
                "test_f1": f1_val,
            },
            "test_set_optimal_diagnostic_only": {
                "diagnostic_threshold": diag_res.threshold,
                "tpr_at_5pct_fpr": diag_res.tpr_at_5pct_fpr,
                "optimal_f1": diag_res.optimal_f1,
                "accuracy_at_threshold": diag_res.accuracy_at_threshold,
                "note": "For diagnostic comparison only; represents transductive leakage if deployed.",
            },
        }

    # Step 5: Robustness by OOD Mechanism
    logger.info("Step 5: Evaluating Robustness by OOD Mechanism...")
    contrast_arr = ood_data["k_contrast"]
    q_max_arr = ood_data["q_max"]
    bc_shift_arr = ood_data["bcs_shift_flags"]

    mask_high_k = contrast_arr >= 20.0
    mask_mod_k = contrast_arr < 20.0
    mask_high_q = q_max_arr > 1500.0
    mask_bc_shift = bc_shift_arr
    mask_high_k_high_q = mask_high_k & mask_high_q
    mask_high_k_bc_shift = mask_high_k & mask_bc_shift
    mask_triple_shift = mask_high_k & mask_high_q & mask_bc_shift

    categories = {
        "all_ood": np.ones(len(contrast_arr), dtype=bool),
        "high_conductivity_contrast": mask_high_k,
        "moderate_contrast_fine_grf": mask_mod_k,
        "high_heat_source": mask_high_q,
        "shifted_boundary_conditions": mask_bc_shift,
        "compound_high_k_high_q": mask_high_k_high_q,
        "compound_high_k_shifted_bc": mask_high_k_bc_shift,
        "triple_compound_shift": mask_triple_shift,
    }

    # We evaluate conformal coverage per category using the recalibrated independent calibrator
    cal_sim_95 = ConformalIntervalCalibrator(confidence_level=0.95, coverage_type="simultaneous")
    cal_sim_95.fit(cal_data["mean_preds"], cal_data["std_preds"], cal_data["true_targets"])
    cal_point_95 = ConformalIntervalCalibrator(confidence_level=0.95, coverage_type="pointwise")
    cal_point_95.fit(cal_data["mean_preds"], cal_data["std_preds"], cal_data["true_targets"])

    # Disagreement detector frozen threshold
    thresh_s_mean = ood_eval_results["ensemble_disagreement_mean"][
        "frozen_operating_point_calibration"
    ]["frozen_threshold"]

    mechanistic_results = {}
    for cat_name, mask in categories.items():
        n_cat = int(np.sum(mask))
        if n_cat == 0:
            continue

        cat_mean_preds = ood_data["mean_preds"][mask]
        cat_std_preds = ood_data["std_preds"][mask]
        cat_true_targets = ood_data["true_targets"][mask]
        cat_mean_s_t = ood_data["mean_s_t"][mask]
        cat_max_s_t = ood_data["max_s_t"][mask]

        cat_sample_metrics = [ood_data["sample_metrics"][i] for i in range(len(mask)) if mask[i]]
        cat_res_metrics = [ood_data["residual_metrics"][i] for i in range(len(mask)) if mask[i]]
        cat_agg = compute_aggregate_statistics(cat_sample_metrics)

        pde_res = [r["pde_residual_l2_w"] for r in cat_res_metrics]
        flux_imb = [r["flux_imbalance_w"] for r in cat_res_metrics]

        # Conformal coverage on category
        cov_sim = cal_sim_95.evaluate(cat_mean_preds, cat_std_preds, cat_true_targets)
        cov_point = cal_point_95.evaluate(cat_mean_preds, cat_std_preds, cat_true_targets)

        # OOD detection TPR at frozen operating threshold
        tpr_disagree = float(np.mean(cat_mean_s_t >= thresh_s_mean))

        mechanistic_results[cat_name] = {
            "num_samples": n_cat,
            "sample_fraction": float(n_cat / len(contrast_arr)),
            "prediction_error": {
                "relative_l2_mean_pct": cat_agg["relative_l2"]["mean"] * 100.0,
                "relative_l2_std_pct": cat_agg["relative_l2"]["std"] * 100.0,
                "rmse_k_mean": cat_agg["rmse_k"]["mean"],
                "rmse_k_std": cat_agg["rmse_k"]["std"],
                "mae_k_mean": cat_agg["mae_k"]["mean"],
                "max_error_k_mean": cat_agg["max_error_k"]["mean"],
                "pde_residual_w_mean": float(np.mean(pde_res)),
                "pde_residual_w_std": float(np.std(pde_res)),
                "flux_imbalance_w_mean": float(np.mean(flux_imb)),
                "flux_imbalance_w_std": float(np.std(flux_imb)),
            },
            "uncertainty": {
                "mean_s_t_kelvin": float(np.mean(cat_mean_s_t)),
                "max_s_t_kelvin": float(np.mean(cat_max_s_t)),
            },
            "conformal_coverage": {
                "pointwise_95_empirical_cov": cov_point.empirical_coverage,
                "pointwise_95_mean_width_k": cov_point.mean_interval_width_k,
                "simultaneous_95_empirical_cov": cov_sim.empirical_coverage,
                "simultaneous_95_mean_width_k": cov_sim.mean_interval_width_k,
            },
            "ood_detection": {
                "ensemble_disagreement_tpr_at_frozen_threshold": tpr_disagree,
            },
        }

    # Step 6: Master Output Dictionary
    master_results = {
        "milestone": "05B",
        "ensemble_provenance": provenance,
        "calibration_dataset_info": {
            "path": str(calibration_dataset_path),
            "num_samples": len(cal_data["sample_ids"]),
            "sample_id_range": f"{cal_data['sample_ids'][0]} .. {cal_data['sample_ids'][-1]}",
            "seed_range": "5000042 .. 5000241",
            "k_range_w_m_k": [float(np.min(cal_data["k_raw"])), float(np.max(cal_data["k_raw"]))],
            "t_range_k": [
                float(np.min(cal_data["true_targets"])),
                float(np.max(cal_data["true_targets"])),
            ],
            "max_solver_residual": float(
                max(r["pde_residual_l2_w"] for r in cal_data["residual_metrics"])
            ),
        },
        "splits_aggregate": {
            "calibration": cal_data["aggregate_error"],
            "validation": val_data["aggregate_error"],
            "test_iid": iid_data["aggregate_error"],
            "test_ood": ood_data["aggregate_error"],
        },
        "conformal_calibration": conformal_results,
        "ood_detection": ood_eval_results,
        "robustness_by_mechanism": mechanistic_results,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=2)

    logger.info(f"Master Milestone 05B results written to {output_report_path}")
    return master_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 05B independent calibration and robustness."
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        nargs="+",
        default=[
            Path("artifacts/checkpoints/scaled_fno_supervised_seed42/best_model.pt"),
            Path("artifacts/checkpoints/scaled_fno_supervised_seed43/best_model.pt"),
            Path("artifacts/checkpoints/scaled_fno_supervised_seed44/best_model.pt"),
            Path("artifacts/checkpoints/scaled_fno_supervised_seed45/best_model.pt"),
            Path("artifacts/checkpoints/scaled_fno_supervised_seed46/best_model.pt"),
        ],
    )
    parser.add_argument(
        "--calibration-dataset", type=Path, default=Path("artifacts/data/calibration_dataset.h5")
    )
    parser.add_argument(
        "--scaled-dataset", type=Path, default=Path("artifacts/data/scaled_dataset.h5")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/milestone_05b_calibration_results.json"),
    )
    args = parser.parse_args()

    run_milestone_05b_pipeline(
        checkpoint_paths=args.checkpoints,
        calibration_dataset_path=args.calibration_dataset,
        scaled_dataset_path=args.scaled_dataset,
        output_report_path=args.output,
    )


if __name__ == "__main__":
    main()
