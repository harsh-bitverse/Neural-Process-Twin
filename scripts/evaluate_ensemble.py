"""Comprehensive uncertainty evaluation, conformal calibration, and OOD detection benchmark."""

from __future__ import annotations

import argparse
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
    evaluate_ood_detection,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate_ensemble_pipeline(
    checkpoint_paths: list[Path],
    dataset_path: Path = Path("artifacts/data/scaled_dataset.h5"),
    output_report_path: Path = Path("artifacts/reports/milestone_05_ensemble_results.json"),
) -> dict[str, Any]:
    """Execute complete Milestone 05 evaluation pipeline."""
    logger.info(f"Loading {len(checkpoint_paths)} ensemble checkpoints...")
    ensemble = DeepEnsemble.from_checkpoints(checkpoint_paths, device="cpu")
    logger.info(f"Initialized DeepEnsemble with members: {ensemble.member_names}")

    splits = ["validation", "test_iid", "test_ood"]
    split_eval_data: dict[str, dict[str, Any]] = {}

    for split_key in splits:
        logger.info(f">>> Evaluating ensemble on split: {split_key}...")
        ds = HeatFieldDataset(
            file_path=dataset_path,
            split=split_key,
            include_coords=True,
            include_bcs=True,
        )
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=ds.nx, ny=ds.ny)

        sample_metrics_list = []
        residual_metrics_list = []

        mean_preds_list = []
        std_preds_list = []
        true_targets_list = []

        mean_s_t_list = []
        max_s_t_list = []
        k_contrast_list = []
        q_max_list = []

        with torch.no_grad():
            for _batch_idx, batch in enumerate(loader):
                x = batch["x"]
                # Run ensemble prediction in Kelvin
                pred = ensemble.predict(x)

                y_pred_k = pred.mean_kelvin.squeeze(0).squeeze(0).numpy()
                y_std_k = pred.std_kelvin.squeeze(0).squeeze(0).numpy()
                t_ref_k = batch["temperature_raw"].squeeze(0).squeeze(0).numpy()

                mean_preds_list.append(y_pred_k)
                std_preds_list.append(y_std_k)
                true_targets_list.append(t_ref_k)

                # Standard error metrics
                m = compute_all_metrics(y_pred_k, t_ref_k)
                sample_metrics_list.append(m)

                # Uncertainty metrics per sample
                mean_s_t_list.append(float(np.mean(y_std_k)))
                max_s_t_list.append(float(np.max(y_std_k)))

                # Physical input properties
                k_raw = batch["k_raw"].squeeze(0).squeeze(0).numpy()
                q_raw = batch["q_raw"].squeeze(0).squeeze(0).numpy()
                k_contrast_list.append(float(np.max(k_raw) / max(np.min(k_raw), 1e-6)))
                q_max_list.append(float(np.max(q_raw)))

                # Physical discrete residuals and boundary flux balance
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

        agg_err = compute_aggregate_statistics(sample_metrics_list)
        pde_res_vals = [r["pde_residual_l2_w"] for r in residual_metrics_list]
        flux_imb_vals = [r["flux_imbalance_w"] for r in residual_metrics_list]

        sample_rmse = np.array([m["rmse_k"] for m in sample_metrics_list])
        sample_rel_l2 = np.array([m["relative_l2"] * 100.0 for m in sample_metrics_list])
        mean_s_t_arr = np.array(mean_s_t_list)
        max_s_t_arr = np.array(max_s_t_list)

        # Risk-coverage curves
        rc_result = compute_risk_coverage(
            sample_rmse, mean_s_t_arr, sample_rel_l2, num_thresholds=10
        )

        # Uncertainty quality (correlations, worst-sample AUROC)
        uq_metrics = compute_uncertainty_quality(
            sample_rmse, sample_rel_l2, mean_s_t_arr, max_s_t_arr
        )

        split_eval_data[split_key] = {
            "mean_preds": np.stack(mean_preds_list, axis=0),
            "std_preds": np.stack(std_preds_list, axis=0),
            "true_targets": np.stack(true_targets_list, axis=0),
            "sample_rmse": sample_rmse,
            "sample_rel_l2": sample_rel_l2,
            "mean_s_t": mean_s_t_arr,
            "max_s_t": max_s_t_arr,
            "k_contrast": np.array(k_contrast_list),
            "q_max": np.array(q_max_list),
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

    # =========================================================================
    # Conformal Calibration (Held-out validation calibration)
    # =========================================================================
    logger.info("Calibrating Conformal Prediction Intervals on Validation Split...")
    val_mean = split_eval_data["validation"]["mean_preds"]
    val_std = split_eval_data["validation"]["std_preds"]
    val_true = split_eval_data["validation"]["true_targets"]

    conformal_results: dict[str, dict[str, Any]] = {}
    for conf_level in [0.90, 0.95]:
        for cov_type in ["simultaneous", "pointwise"]:
            cal = ConformalIntervalCalibrator(confidence_level=conf_level, coverage_type=cov_type)
            q_hat = cal.fit(val_mean, val_std, val_true)
            key = f"{cov_type}_{int(conf_level * 100)}"

            conformal_results[key] = {
                "confidence_level": conf_level,
                "coverage_type": cov_type,
                "quantile_multiplier_q_hat": q_hat,
                "splits": {},
            }
            for s_key in ["validation", "test_iid", "test_ood"]:
                eval_res = cal.evaluate(
                    split_eval_data[s_key]["mean_preds"],
                    split_eval_data[s_key]["std_preds"],
                    split_eval_data[s_key]["true_targets"],
                )
                conformal_results[key]["splits"][s_key] = {
                    "empirical_coverage": eval_res.empirical_coverage,
                    "mean_interval_width_k": eval_res.mean_interval_width_k,
                    "median_interval_width_k": eval_res.median_interval_width_k,
                }

    # =========================================================================
    # Out-of-Distribution Detection (test_iid vs test_ood)
    # =========================================================================
    logger.info("Evaluating Out-of-Distribution Detection (test_iid vs test_ood)...")
    iid_data = split_eval_data["test_iid"]
    ood_data = split_eval_data["test_ood"]

    # 1. Ensemble Disagreement (mean s_T)
    det_ens_mean = EnsembleDisagreementDetector(aggregation="mean")
    s_iid_ens_mean = det_ens_mean.score(iid_data["std_preds"])
    s_ood_ens_mean = det_ens_mean.score(ood_data["std_preds"])
    res_ens_mean = evaluate_ood_detection(
        s_iid_ens_mean, s_ood_ens_mean, detector_name="ensemble_disagreement_mean"
    )

    # 2. Ensemble Disagreement (max s_T)
    det_ens_max = EnsembleDisagreementDetector(aggregation="max")
    s_iid_ens_max = det_ens_max.score(iid_data["std_preds"])
    s_ood_ens_max = det_ens_max.score(ood_data["std_preds"])
    res_ens_max = evaluate_ood_detection(
        s_iid_ens_max, s_ood_ens_max, detector_name="ensemble_disagreement_max"
    )

    # 3. Input Contrast Ratio (k_max / k_min)
    s_iid_contrast = iid_data["k_contrast"]
    s_ood_contrast = ood_data["k_contrast"]
    res_contrast = evaluate_ood_detection(
        s_iid_contrast, s_ood_contrast, detector_name="input_k_contrast"
    )

    # 4. Input Peak Heat Source (max q)
    s_iid_q = iid_data["q_max"]
    s_ood_q = ood_data["q_max"]
    res_q = evaluate_ood_detection(s_iid_q, s_ood_q, detector_name="input_q_max")

    # 5. Composite Input Detector
    s_iid_comp = np.maximum(s_iid_contrast / 7.16, s_iid_q / 912.8)
    s_ood_comp = np.maximum(s_ood_contrast / 7.16, s_ood_q / 912.8)
    res_comp = evaluate_ood_detection(s_iid_comp, s_ood_comp, detector_name="input_composite")

    ood_results = {
        "ensemble_disagreement_mean": res_ens_mean.__dict__,
        "ensemble_disagreement_max": res_ens_max.__dict__,
        "input_k_contrast": res_contrast.__dict__,
        "input_q_max": res_q.__dict__,
        "input_composite": res_comp.__dict__,
    }

    # =========================================================================
    # Clean output dictionary (exclude raw numpy arrays)
    # =========================================================================
    final_output = {
        "ensemble_summary": {
            "num_members": ensemble.num_members,
            "member_names": ensemble.member_names,
            "architecture": "fno2d",
        },
        "splits": {
            s: {
                "aggregate_error": split_eval_data[s]["aggregate_error"],
                "uncertainty_quality": split_eval_data[s]["uncertainty_quality"],
                "risk_coverage": split_eval_data[s]["risk_coverage"],
            }
            for s in splits
        },
        "conformal_calibration": conformal_results,
        "ood_detection": ood_results,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"Master ensemble results saved to {output_report_path}")
    return final_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deep ensemble and OOD detection.")
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
        help="List of model checkpoint files.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/scaled_dataset.h5"),
        help="Path to HDF5 dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/milestone_05_ensemble_results.json"),
        help="Path for destination report JSON.",
    )
    args = parser.parse_args()

    evaluate_ensemble_pipeline(
        checkpoint_paths=args.checkpoints,
        dataset_path=args.dataset,
        output_report_path=args.output,
    )


if __name__ == "__main__":
    main()
