"""Split conformal prediction calibration for spatial temperature field intervals.

Implements distribution-free prediction intervals calibrated on held-out validation data
to guarantee finite-sample coverage without distributional assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ConformalCalibrationResult:
    """Results from evaluating conformal prediction intervals on a test split.

    Attributes:
        confidence_level: Nominal confidence level 1 - alpha (e.g. 0.90 or 0.95).
        quantile_multiplier: Calibrated empirical quantile multiplier q_hat.
        coverage_type: 'simultaneous' (field-level) or 'pointwise' (cell-level).
        empirical_coverage: Actual fraction of targets covered by the interval.
        mean_interval_width_k: Average interval width 2 * q_hat * s_T in Kelvin [K].
        median_interval_width_k: Median interval width in Kelvin [K].
    """

    confidence_level: float
    quantile_multiplier: float
    coverage_type: str
    empirical_coverage: float
    mean_interval_width_k: float
    median_interval_width_k: float


class ConformalIntervalCalibrator:
    """Split conformal prediction calibrator for ensemble spatial field predictions.

    Given validation fields (y_pred, s_pred, y_true), computes nonconformity scores
    R_i and determines the finite-sample corrected empirical quantile:
        q_hat = Quantile(scores, ceil((N + 1) * (1 - alpha)) / N)

    Prediction intervals are constructed as:
        [ \\bar{T}(x, y) - q_hat * (s_T(x, y) + eps), \\bar{T}(x, y) + q_hat * (s_T(x, y) + eps) ]
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        coverage_type: str = "simultaneous",
        eps: float = 1e-4,
    ) -> None:
        """Initialize ConformalIntervalCalibrator.

        Args:
            confidence_level: Desired coverage probability 1 - alpha (e.g. 0.95).
            coverage_type: 'simultaneous' (entire 2D field covered with prob >= 1-alpha)
                           or 'pointwise' (average cell coverage >= 1-alpha).
            eps: Numerical stability constant added to standard deviation.
        """
        if not (0.0 < confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}.")
        if coverage_type not in ("simultaneous", "pointwise"):
            raise ValueError(
                f"coverage_type must be 'simultaneous' or 'pointwise', got '{coverage_type}'."
            )

        self.confidence_level = float(confidence_level)
        self.coverage_type = coverage_type
        self.eps = float(eps)
        self.quantile_multiplier: float | None = None
        self.calibration_scores: np.ndarray | None = None

    def fit(
        self,
        mean_preds: np.ndarray | torch.Tensor,
        std_preds: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
    ) -> float:
        """Calibrate prediction intervals on a held-out calibration split (e.g. validation).

        Args:
            mean_preds: Predictive mean in Kelvin (N, 1, ny, nx) or (N, ny, nx).
            std_preds: Predictive std in Kelvin (N, 1, ny, nx) or (N, ny, nx).
            targets: Reference target fields in Kelvin (N, 1, ny, nx) or (N, ny, nx).

        Returns:
            Calibrated empirical quantile multiplier q_hat.
        """
        y_mean = np.asarray(mean_preds, dtype=np.float64)
        y_std = np.asarray(std_preds, dtype=np.float64)
        y_true = np.asarray(targets, dtype=np.float64)

        if y_mean.shape != y_true.shape or y_mean.shape != y_std.shape:
            raise ValueError(
                f"Shape mismatch: mean {y_mean.shape}, std {y_std.shape}, true {y_true.shape}"
            )

        n_samples = y_mean.shape[0]
        if n_samples < 5:
            raise ValueError(f"Need at least 5 calibration samples, got {n_samples}.")

        # Pointwise absolute normalized errors
        abs_err = np.abs(y_true - y_mean)
        norm_scores = abs_err / (y_std + self.eps)

        if self.coverage_type == "simultaneous":
            # Field-level max score per sample: ensures entire 2D field is simultaneously covered
            # Shape: (N,)
            sample_scores = np.max(norm_scores, axis=tuple(range(1, norm_scores.ndim)))
        else:
            # Pointwise scores across all spatial cells and samples
            sample_scores = norm_scores.ravel()

        self.calibration_scores = np.sort(sample_scores)
        n = len(sample_scores)

        # Finite-sample correction: ceil((n + 1) * (1 - alpha)) / n
        level = min(1.0, np.ceil((n + 1) * self.confidence_level) / n)
        # Empirical quantile
        q_idx = int(np.ceil(level * n)) - 1
        q_idx = max(0, min(q_idx, n - 1))
        self.quantile_multiplier = float(self.calibration_scores[q_idx])

        return self.quantile_multiplier

    def evaluate(
        self,
        mean_preds: np.ndarray | torch.Tensor,
        std_preds: np.ndarray | torch.Tensor,
        targets: np.ndarray | torch.Tensor,
    ) -> ConformalCalibrationResult:
        """Evaluate empirical coverage and interval widths on an evaluation partition.

        Args:
            mean_preds: Predictive mean in Kelvin (N, 1, ny, nx) or (N, ny, nx).
            std_preds: Predictive std in Kelvin (N, 1, ny, nx) or (N, ny, nx).
            targets: Reference target fields in Kelvin (N, 1, ny, nx) or (N, ny, nx).

        Returns:
            ConformalCalibrationResult with empirical coverage and interval width metrics.
        """
        if self.quantile_multiplier is None:
            raise RuntimeError("Calibrator has not been fitted. Call fit() first.")

        y_mean = np.asarray(mean_preds, dtype=np.float64)
        y_std = np.asarray(std_preds, dtype=np.float64)
        y_true = np.asarray(targets, dtype=np.float64)

        q = self.quantile_multiplier
        half_width = q * (y_std + self.eps)
        lower = y_mean - half_width
        upper = y_mean + half_width

        is_covered = (y_true >= lower) & (y_true <= upper)

        if self.coverage_type == "simultaneous":
            # Sample is covered only if ALL cells in its 2D domain are covered
            covered_samples = np.all(is_covered, axis=tuple(range(1, is_covered.ndim)))
            empirical_cov = float(np.mean(covered_samples))
        else:
            empirical_cov = float(np.mean(is_covered))

        width_k = 2.0 * half_width
        mean_width = float(np.mean(width_k))
        median_width = float(np.median(width_k))

        return ConformalCalibrationResult(
            confidence_level=self.confidence_level,
            quantile_multiplier=q,
            coverage_type=self.coverage_type,
            empirical_coverage=empirical_cov,
            mean_interval_width_k=mean_width,
            median_interval_width_k=median_width,
        )
