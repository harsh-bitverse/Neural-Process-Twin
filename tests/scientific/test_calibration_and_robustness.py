"""Scientific and regression tests for Milestone 05B independent calibration and robustness."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from neural_process_twin.data.splits import (
    DatasetPartitionPlan,
    SplitName,
    verify_partitions_disjoint,
)
from neural_process_twin.uncertainty.calibration import ConformalIntervalCalibrator


@pytest.mark.scientific
def test_calibration_dataset_provenance_and_disjointness() -> None:
    """Verify that calibration dataset exists, has 200 samples, and has disjoint seeds."""
    cal_path = Path("artifacts/data/calibration_dataset.h5")
    assert cal_path.is_file(), f"Calibration dataset missing at {cal_path}"

    with h5py.File(cal_path, "r") as f:
        assert "calibration" in f, "Partition 'calibration' missing from archive"
        cal_grp = f["calibration"]
        n_samples = cal_grp.attrs["sample_count"]
        assert n_samples == 200, f"Expected 200 samples, got {n_samples}"

        # Check solver residuals and heat balance
        diag_grp = cal_grp["diagnostics"]
        rel_res = diag_grp["relative_residual"][:]
        heat_imb = diag_grp["heat_imbalance"][:]

        assert np.max(rel_res) <= 1e-6, f"Solver relative residual exceeded 1e-6: {np.max(rel_res)}"
        assert np.max(heat_imb) <= 1e-3, f"Heat imbalance exceeded 1e-3: {np.max(heat_imb)}"

        # Check temperature positivity
        t_vals = cal_grp["temperature"][:]
        assert np.min(t_vals) > 0.0, f"Non-positive temperature detected: {np.min(t_vals)}"

        # Verify seed disjointness
        cal_seeds = cal_grp["seed"][:].tolist()
        assert len(cal_seeds) == 200
        assert cal_seeds[0] == 5000042
        assert cal_seeds[-1] == 5000241

    plan = DatasetPartitionPlan(
        base_seed=42,
        num_train=1000,
        num_val=150,
        num_test_iid=200,
        num_test_ood=200,
        num_cal=200,
    )
    all_seeds = {
        s: plan.get_split_seeds(s)
        for s in (
            SplitName.TRAIN,
            SplitName.VALIDATION,
            SplitName.TEST_IID,
            SplitName.TEST_OOD,
            SplitName.CALIBRATION,
        )
    }
    verify_partitions_disjoint(all_seeds)


@pytest.mark.scientific
def test_conformal_finite_sample_quantile_exact_formula() -> None:
    """Verify finite-sample correction formula ceil((n+1)(1-alpha))/n on synthetic data."""
    calibrator_95 = ConformalIntervalCalibrator(confidence_level=0.95, coverage_type="simultaneous")
    calibrator_90 = ConformalIntervalCalibrator(confidence_level=0.90, coverage_type="simultaneous")

    n_samples = 200
    mean_preds = np.zeros((n_samples, 8, 8))
    std_preds = np.ones((n_samples, 8, 8))
    # Synthetic errors linearly increasing from 1.0 to 200.0
    targets = np.linspace(1.0, 200.0, n_samples).reshape(n_samples, 1, 1) * np.ones(
        (n_samples, 8, 8)
    )

    q95 = calibrator_95.fit(mean_preds, std_preds, targets)
    q90 = calibrator_90.fit(mean_preds, std_preds, targets)

    # For n=200, 1-alpha=0.95: ceil(201 * 0.95) = ceil(190.95) = 191
    # 0-indexed rank index = 190, corresponding to targets[190]
    expected_rank_95 = int(np.ceil(201 * 0.95))
    expected_q95 = float(
        np.sort(np.max(np.abs(targets) / (std_preds + 1e-4), axis=(1, 2)))[expected_rank_95 - 1]
    )
    assert np.isclose(q95, expected_q95, atol=1e-5)

    expected_rank_90 = int(np.ceil(201 * 0.90))
    expected_q90 = float(
        np.sort(np.max(np.abs(targets) / (std_preds + 1e-4), axis=(1, 2)))[expected_rank_90 - 1]
    )
    assert np.isclose(q90, expected_q90, atol=1e-5)


@pytest.mark.scientific
def test_milestone_05b_results_consistency() -> None:
    """Verify stored Milestone 05B json report metrics."""
    res_path = Path("artifacts/reports/milestone_05b_calibration_results.json")
    assert res_path.is_file(), f"Report JSON missing at {res_path}"

    with open(res_path) as f:
        data = json.load(f)

    # 1. Provenance
    assert len(data["ensemble_provenance"]) == 5
    for p in data["ensemble_provenance"]:
        assert len(p["sha256"]) == 64

    # 2. Conformal calibration: check that independent calibration achieves >= 94% on test_iid
    conf = data["conformal_calibration"]["calibration_source"]["independent_calibration"]
    assert conf["pointwise_95"]["splits"]["test_iid"]["empirical_coverage"] >= 0.94
    assert conf["simultaneous_95"]["splits"]["test_iid"]["empirical_coverage"] >= 0.95

    # 3. OOD detection frozen thresholds: IID FPR <= 0.10 and OOD TPR >= 0.99
    ood = data["ood_detection"]["ensemble_disagreement_mean"]
    cal_op = ood["frozen_operating_point_calibration"]
    assert cal_op["empirical_test_iid_fpr"] <= 0.08
    assert cal_op["empirical_test_ood_tpr"] == 1.0

    # 4. Mechanistic breakdown: triple compound shift exhibits higher relative error than all_ood
    mech = data["robustness_by_mechanism"]
    all_ood_err = mech["all_ood"]["prediction_error"]["relative_l2_mean_pct"]
    triple_err = mech["triple_compound_shift"]["prediction_error"]["relative_l2_mean_pct"]
    assert triple_err > all_ood_err, (
        f"Triple compound error ({triple_err}%) must exceed aggregate OOD error ({all_ood_err}%)"
    )
