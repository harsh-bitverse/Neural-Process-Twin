"""Unit tests for uncertainty quantification, deep ensembles, conformal prediction, and OOD detection."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_process_twin.models.base import SurrogateModel
from neural_process_twin.uncertainty.calibration import (
    ConformalCalibrationResult,
    ConformalIntervalCalibrator,
)
from neural_process_twin.uncertainty.ensemble import (
    DeepEnsemble,
    EnsemblePrediction,
)
from neural_process_twin.uncertainty.metrics import (
    RiskCoverageResult,
    UncertaintyQualityMetrics,
    compute_risk_coverage,
    compute_uncertainty_quality,
)
from neural_process_twin.uncertainty.ood import (
    InputDistributionDetector,
    OODDetectionResult,
    evaluate_ood_detection,
)


class DummySurrogate(SurrogateModel):
    """Simple deterministic mock surrogate returning scaled input."""

    def __init__(self, multiplier: float, offset: float = 0.0) -> None:
        super().__init__(in_channels=1, out_channels=1)
        self.multiplier = nn.Parameter(torch.tensor(float(multiplier)))
        self.offset = nn.Parameter(torch.tensor(float(offset)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0:1] * self.multiplier + self.offset


class TestDeepEnsemble:
    """Tests for DeepEnsemble creation, mathematical formulas, and inference."""

    def test_ensemble_validation(self) -> None:
        """DeepEnsemble must require at least 2 models for sample standard deviation."""
        with pytest.raises(ValueError, match="at least one model"):
            DeepEnsemble(models=[])

        m1 = DummySurrogate(1.0)
        with pytest.raises(ValueError, match="at least 2 members"):
            DeepEnsemble(models=[m1])

    def test_mathematical_mean_and_sample_std(self) -> None:
        """Verify predictive mean and sample std (ddof=1) against explicit NumPy calculations."""
        # 3 members with constant multipliers: 1.0, 2.0, 3.0
        m1 = DummySurrogate(1.0)
        m2 = DummySurrogate(2.0)
        m3 = DummySurrogate(3.0)
        ensemble = DeepEnsemble(models=[m1, m2, m3])

        # Input: batch of 2, 1 channel, 4x4 spatial grid of ones
        x = torch.ones(2, 1, 4, 4)
        t_mean = 300.0
        t_std = 20.0

        res = ensemble.predict(x, t_mean=t_mean, t_std=t_std)

        assert isinstance(res, EnsemblePrediction)
        assert res.num_members == 3
        assert res.mean_kelvin.shape == (2, 1, 4, 4)
        assert res.std_kelvin.shape == (2, 1, 4, 4)
        assert res.members_kelvin.shape == (3, 2, 1, 4, 4)

        # Expected normalized predictions: 1.0, 2.0, 3.0
        # Mean normalized: (1 + 2 + 3) / 3 = 2.0
        # Sample std normalized (ddof=1): sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2) / 2) = sqrt(2/2) = 1.0
        assert torch.allclose(res.mean_normalized, torch.tensor(2.0))
        assert torch.allclose(res.std_normalized, torch.tensor(1.0))

        # Expected Kelvin:
        # Mean Kelvin: 2.0 * 20.0 + 300.0 = 340.0 K
        # Std Kelvin: 1.0 * 20.0 = 20.0 K
        assert torch.allclose(res.mean_kelvin, torch.tensor(340.0))
        assert torch.allclose(res.std_kelvin, torch.tensor(20.0))

    def test_per_sample_summary_properties(self) -> None:
        """Verify per_sample_mean_uncertainty and per_sample_max_uncertainty."""
        m1 = DummySurrogate(1.0)
        m2 = DummySurrogate(3.0)
        ensemble = DeepEnsemble(models=[m1, m2])

        x = torch.ones(3, 1, 4, 4)
        res = ensemble.predict(x, t_mean=300.0, t_std=10.0)

        # 2 members: 1.0 and 3.0 -> sample std = sqrt((1-2)^2 + (3-2)^2) = sqrt(2) * 10.0 K
        expected_std = np.sqrt(2.0) * 10.0
        assert res.per_sample_mean_uncertainty.shape == (3,)
        assert res.per_sample_max_uncertainty.shape == (3,)
        assert np.allclose(res.per_sample_mean_uncertainty.numpy(), expected_std)
        assert np.allclose(res.per_sample_max_uncertainty.numpy(), expected_std)

    def test_forward_drop_in_replacement(self) -> None:
        """Forward pass should output predictive mean tensor in normalized space."""
        m1 = DummySurrogate(2.0)
        m2 = DummySurrogate(4.0)
        ensemble = DeepEnsemble(models=[m1, m2])

        x = torch.ones(1, 1, 4, 4)
        y_mean = ensemble(x)
        assert y_mean.shape == (1, 1, 4, 4)
        assert torch.allclose(y_mean, torch.tensor(3.0))

    def test_from_checkpoints_error_handling(self, tmp_path: pytest.TempPathFactory) -> None:
        """Verify error handling on invalid or missing checkpoint paths."""
        with pytest.raises(ValueError, match="No checkpoint paths"):
            DeepEnsemble.from_checkpoints([])

        with pytest.raises(ValueError, match="at least 2 checkpoints"):
            DeepEnsemble.from_checkpoints(["nonexistent.pt"])

        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            DeepEnsemble.from_checkpoints(["nonexistent_1.pt", "nonexistent_2.pt"])


class TestConformalCalibration:
    """Tests for distribution-free split conformal prediction intervals."""

    def test_calibration_fit_and_evaluate(self) -> None:
        """Calibrate on synthetic data and verify empirical coverage on exchangeable test data."""
        np.random.seed(42)
        n_cal = 100
        n_test = 200

        # Ground truth: 300.0 K
        y_true_cal = np.full((n_cal, 1, 4, 4), 300.0)
        # Predictions with Gaussian noise: mean 300, std 2.0
        y_mean_cal = y_true_cal + np.random.normal(0, 2.0, size=(n_cal, 1, 4, 4))
        # Known uncertainty estimate: 2.0 K
        y_std_cal = np.full((n_cal, 1, 4, 4), 2.0)

        calibrator = ConformalIntervalCalibrator(confidence_level=0.90, coverage_type="pointwise")
        q_hat = calibrator.fit(y_mean_cal, y_std_cal, y_true_cal)

        # For Gaussian with true std=2.0, 90% normal quantile is ~1.645
        assert 1.3 < q_hat < 2.1

        # Test on exchangeable distribution
        y_true_test = np.full((n_test, 1, 4, 4), 300.0)
        y_mean_test = y_true_test + np.random.normal(0, 2.0, size=(n_test, 1, 4, 4))
        y_std_test = np.full((n_test, 1, 4, 4), 2.0)

        result = calibrator.evaluate(y_mean_test, y_std_test, y_true_test)
        assert isinstance(result, ConformalCalibrationResult)
        assert result.confidence_level == 0.90
        # Finite-sample coverage should be close to 90% (allow sampling margin 85% - 95%)
        assert 0.85 <= result.empirical_coverage <= 0.95
        assert result.mean_interval_width_k > 0.0

    def test_unfitted_eval_raises(self) -> None:
        """Calling evaluate before fit must raise RuntimeError."""
        calibrator = ConformalIntervalCalibrator()
        with pytest.raises(RuntimeError, match="not been fitted"):
            calibrator.evaluate(np.zeros((10, 4, 4)), np.ones((10, 4, 4)), np.zeros((10, 4, 4)))


class TestOODDetection:
    """Tests for OOD detection metrics, disagreement detectors, and input diagnostics."""

    def test_perfect_separation_auroc(self) -> None:
        """Verify AUROC and AUPRC equal 1.0 when IID and OOD scores are perfectly separated."""
        iid_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ood_scores = np.array([1.1, 1.2, 1.3, 1.4, 1.5])

        res = evaluate_ood_detection(iid_scores, ood_scores, detector_name="perfect")
        assert isinstance(res, OODDetectionResult)
        assert np.isclose(res.auroc, 1.0)
        assert np.isclose(res.auprc, 1.0)
        assert np.isclose(res.tpr_at_5pct_fpr, 1.0)
        assert np.isclose(res.optimal_f1, 1.0)
        assert res.num_iid == 5
        assert res.num_ood == 5

    def test_inverse_separation_auroc(self) -> None:
        """Verify AUROC equals 0.0 when scores are completely inverted."""
        iid_scores = np.array([10.0, 20.0, 30.0])
        ood_scores = np.array([1.0, 2.0, 3.0])

        res = evaluate_ood_detection(iid_scores, ood_scores, detector_name="inverted")
        assert np.isclose(res.auroc, 0.0)

    def test_input_distribution_detector(self) -> None:
        """Test physical input diagnostics for thermal conductivity contrast and heat source."""
        det_contrast = InputDistributionDetector(mode="contrast")
        det_source = InputDistributionDetector(mode="heat_source")

        # Sample 1: k in [1.0, 4.0] -> contrast 4.0; max q = 100.0
        k1 = np.full((1, 4, 4), 1.0)
        k1[0, 0, 0] = 4.0
        q1 = np.full((1, 4, 4), 50.0)
        q1[0, 0, 0] = 100.0

        # Sample 2: k in [1.0, 20.0] -> contrast 20.0 (OOD); max q = 2000.0 (OOD)
        k2 = np.full((1, 4, 4), 1.0)
        k2[0, 0, 0] = 20.0
        q2 = np.full((1, 4, 4), 50.0)
        q2[0, 0, 0] = 2000.0

        k_batch = np.concatenate([k1, k2], axis=0)
        q_batch = np.concatenate([q1, q2], axis=0)

        contrast_scores = det_contrast.score(k_batch, q_batch)
        source_scores = det_source.score(k_batch, q_batch)

        assert np.allclose(contrast_scores, [4.0, 20.0])
        assert np.allclose(source_scores, [100.0, 2000.0])


class TestUncertaintyQualityMetrics:
    """Tests for risk-coverage curves, error correlation, and selective prediction."""

    def test_risk_coverage_curve(self) -> None:
        """Verify that risk decreases monotonically when uncertainty ranks error accurately."""
        # 10 samples with perfectly correlated error and uncertainty
        errors = np.linspace(1.0, 10.0, 10)
        uncertainties = np.linspace(0.1, 1.0, 10)

        res = compute_risk_coverage(errors, uncertainties, num_thresholds=5)
        assert isinstance(res, RiskCoverageResult)
        assert len(res.coverages) == 5
        # Retaining fewer samples (lower coverage) must yield lower mean RMSE
        assert res.rmse_at_coverage[0] < res.rmse_at_coverage[-1]
        assert np.all(np.diff(res.rmse_at_coverage) >= 0)

    def test_uncertainty_quality_correlations(self) -> None:
        """Verify Pearson and Spearman correlations are positive for aligned uncertainty."""
        errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rel_errors = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        mean_s_t = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        max_s_t = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        metrics = compute_uncertainty_quality(errors, rel_errors, mean_s_t, max_s_t)
        assert isinstance(metrics, UncertaintyQualityMetrics)
        assert np.isclose(metrics.pearson_r_rmse, 1.0)
        assert np.isclose(metrics.spearman_rho_rmse, 1.0)
        assert np.isclose(metrics.pearson_r_rel_l2, 1.0)
        assert np.isclose(metrics.worst_10pct_detection_auroc, 1.0)
