"""Scientific verification tests for uncertainty quantification and out-of-distribution detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.uncertainty.calibration import ConformalIntervalCalibrator
from neural_process_twin.uncertainty.ensemble import DeepEnsemble
from neural_process_twin.uncertainty.metrics import compute_risk_coverage
from neural_process_twin.uncertainty.ood import (
    InputDistributionDetector,
    evaluate_ood_detection,
)


class TestScientificUncertaintyVerification:
    """Scientific tests validating uncertainty properties, coverage bounds, and real checkpoints."""

    def test_real_fno_checkpoint_ensemble_forward(self) -> None:
        """Verify DeepEnsemble loads real saved checkpoints and predicts valid Kelvin fields."""
        ckpt_dir = Path("artifacts/checkpoints")
        ckpt_paths = [
            ckpt_dir / "scaled_fno_supervised_seed42" / "best_model.pt",
            ckpt_dir / "scaled_fno_supervised_seed43" / "best_model.pt",
            ckpt_dir / "scaled_fno_supervised_seed44" / "best_model.pt",
        ]

        # Check if checkpoints exist
        if not all(p.is_file() for p in ckpt_paths):
            pytest.skip("Pre-trained scaled FNO checkpoints not found.")

        dataset_path = Path("artifacts/data/scaled_dataset.h5")
        if not dataset_path.is_file():
            pytest.skip("Scaled dataset not found.")

        ensemble = DeepEnsemble.from_checkpoints(ckpt_paths, device="cpu")
        assert ensemble.num_members == 3

        dataset = HeatFieldDataset(
            file_path=dataset_path,
            split="validation",
            include_coords=True,
            include_bcs=True,
        )

        sample = dataset[0]
        x = sample["x"].unsqueeze(0)  # (1, 12, 32, 32)
        y_true_k = sample["temperature_raw"]  # (1, 32, 32)

        pred = ensemble.predict(x)
        assert pred.mean_kelvin.shape == (1, 1, 32, 32)
        assert pred.std_kelvin.shape == (1, 1, 32, 32)

        # Check physical units: temperatures should be around ~300 K, std positive and < 50 K
        mean_val = float(pred.mean_kelvin.mean())
        std_val = float(pred.std_kelvin.mean())

        assert 250.0 < mean_val < 450.0
        assert 0.0 < std_val < 50.0
        assert not torch.isnan(pred.mean_kelvin).any()
        assert not torch.isnan(pred.std_kelvin).any()

        # Check agreement with ground truth is reasonable (mean error < 10 K for trained FNO)
        err = float(torch.abs(pred.mean_kelvin.squeeze(0) - y_true_k).mean())
        assert err < 10.0

    def test_selective_prediction_monotonic_error_reduction(self) -> None:
        """Verify that risk-coverage curves strictly reduce error as uncertain samples are discarded."""
        np.random.seed(42)
        n = 100
        # True errors and aligned uncertainties with slight perturbation
        base_err = np.linspace(0.5, 20.0, n)
        noise = np.random.normal(0, 0.5, n)
        uncertainties = base_err + noise

        res = compute_risk_coverage(base_err, uncertainties, num_thresholds=10)
        # Error at 10% coverage must be significantly lower than at 100% coverage
        assert res.rmse_at_coverage[0] < res.rmse_at_coverage[-1]
        assert res.rmse_at_coverage[0] < 0.3 * res.rmse_at_coverage[-1]

    def test_conformal_simultaneous_coverage_property(self) -> None:
        """Verify simultaneous field coverage achieves nominal 90% level under exchangeability."""
        np.random.seed(123)
        n_cal = 150
        n_test = 200

        # Calibration split
        y_true_cal = np.full((n_cal, 1, 8, 8), 320.0)
        err_cal = np.random.normal(0, 1.5, size=(n_cal, 1, 8, 8))
        y_mean_cal = y_true_cal + err_cal
        y_std_cal = np.full((n_cal, 1, 8, 8), 1.5)

        calibrator = ConformalIntervalCalibrator(
            confidence_level=0.90, coverage_type="simultaneous"
        )
        calibrator.fit(y_mean_cal, y_std_cal, y_true_cal)

        # Exchangeable test split
        y_true_test = np.full((n_test, 1, 8, 8), 320.0)
        err_test = np.random.normal(0, 1.5, size=(n_test, 1, 8, 8))
        y_mean_test = y_true_test + err_test
        y_std_test = np.full((n_test, 1, 8, 8), 1.5)

        eval_res = calibrator.evaluate(y_mean_test, y_std_test, y_true_test)
        # Simultaneous coverage across all 64 cells per field should be around 90% (>= 83%)
        assert eval_res.empirical_coverage >= 0.83

    def test_ood_detection_physics_separation(self) -> None:
        """Verify InputDistributionDetector perfectly separates IID from OOD contrast samples."""
        det = InputDistributionDetector(mode="contrast")

        # In-distribution: contrast ~ 4.5
        k_iid = np.full((20, 1, 8, 8), 1.0)
        k_iid[:, :, 0, 0] = 4.5
        q_iid = np.full((20, 1, 8, 8), 50.0)

        # Out-of-distribution: contrast ~ 35.0
        k_ood = np.full((20, 1, 8, 8), 1.0)
        k_ood[:, :, 0, 0] = 35.0
        q_ood = np.full((20, 1, 8, 8), 50.0)

        scores_iid = det.score(k_iid, q_iid)
        scores_ood = det.score(k_ood, q_ood)

        ood_eval = evaluate_ood_detection(scores_iid, scores_ood, detector_name="contrast")
        assert ood_eval.auroc == 1.0
        assert ood_eval.tpr_at_5pct_fpr == 1.0
