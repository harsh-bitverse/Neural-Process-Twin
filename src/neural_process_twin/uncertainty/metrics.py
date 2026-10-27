"""Evaluation metrics for surrogate uncertainty quality and selective prediction.

Implements risk-coverage curves, error-uncertainty correlation (Pearson & Spearman),
and worst-case error detection performance for spatial fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.stats as stats

from neural_process_twin.uncertainty.ood import compute_auroc_auprc


@dataclass(frozen=True)
class RiskCoverageResult:
    """Risk-coverage evaluation results for selective surrogate prediction.

    Attributes:
        coverages: Monotonically increasing array of sample retention fractions in (0, 1].
        rmse_at_coverage: Mean RMSE [K] computed on retained samples at each coverage level.
        rel_l2_at_coverage: Mean relative L2 error [%] on retained samples at each coverage level.
        aurc: Area under the risk-coverage curve (lower is better selective prediction).
    """

    coverages: np.ndarray
    rmse_at_coverage: np.ndarray
    rel_l2_at_coverage: np.ndarray
    aurc: float


@dataclass(frozen=True)
class UncertaintyQualityMetrics:
    """Comprehensive uncertainty evaluation metrics across a dataset partition.

    Attributes:
        mean_uncertainty_k: Average predictive standard deviation across all samples [K].
        max_uncertainty_k: Average peak spatial standard deviation across samples [K].
        pearson_r_rmse: Pearson correlation between sample mean s_T and sample RMSE.
        spearman_rho_rmse: Spearman rank correlation between sample mean s_T and sample RMSE.
        pearson_r_rel_l2: Pearson correlation between sample mean s_T and sample relative L2 error.
        spearman_rho_rel_l2: Spearman rank correlation between sample mean s_T and sample relative L2.
        worst_10pct_detection_auroc: AUROC for identifying the top 10% highest-error samples using s_T.
        worst_20pct_detection_auroc: AUROC for identifying the top 20% highest-error samples using s_T.
    """

    mean_uncertainty_k: float
    max_uncertainty_k: float
    pearson_r_rmse: float
    spearman_rho_rmse: float
    pearson_r_rel_l2: float
    spearman_rho_rel_l2: float
    worst_10pct_detection_auroc: float
    worst_20pct_detection_auroc: float


def compute_risk_coverage(
    errors: np.ndarray,
    uncertainties: np.ndarray,
    relative_errors: np.ndarray | None = None,
    num_thresholds: int = 20,
) -> RiskCoverageResult:
    """Compute risk-coverage curve by selectively retaining low-uncertainty samples.

    Args:
        errors: 1D array of sample RMSE values in Kelvin (N,).
        uncertainties: 1D array of sample uncertainty scores (e.g. mean s_T) (N,).
        relative_errors: Optional 1D array of sample relative L2 errors [%] (N,).
        num_thresholds: Number of coverage evaluation steps from 0.05 to 1.0.

    Returns:
        RiskCoverageResult containing coverages, error trajectories, and AURC.
    """
    err = np.asarray(errors, dtype=np.float64).ravel()
    unc = np.asarray(uncertainties, dtype=np.float64).ravel()
    rel_err = (
        np.asarray(relative_errors, dtype=np.float64).ravel()
        if relative_errors is not None
        else err
    )

    n = len(err)
    if n < 2:
        return RiskCoverageResult(
            coverages=np.array([1.0]),
            rmse_at_coverage=np.array([float(np.mean(err))]),
            rel_l2_at_coverage=np.array([float(np.mean(rel_err))]),
            aurc=float(np.mean(err)),
        )

    # Sort samples by uncertainty ascending (best/most confident first)
    sort_idx = np.argsort(unc)
    err_sorted = err[sort_idx]
    rel_err_sorted = rel_err[sort_idx]

    coverages = np.linspace(0.1, 1.0, num_thresholds)
    rmse_list = []
    rel_l2_list = []

    for cov in coverages:
        k = max(1, int(np.ceil(cov * n)))
        retained_err = err_sorted[:k]
        retained_rel = rel_err_sorted[:k]
        rmse_list.append(float(np.mean(retained_err)))
        rel_l2_list.append(float(np.mean(retained_rel)))

    rmse_arr = np.array(rmse_list)
    rel_l2_arr = np.array(rel_l2_list)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(trapz_fn(rmse_arr, coverages))

    return RiskCoverageResult(
        coverages=coverages,
        rmse_at_coverage=rmse_arr,
        rel_l2_at_coverage=rel_l2_arr,
        aurc=aurc,
    )


def compute_uncertainty_quality(
    sample_rmse_k: np.ndarray,
    sample_rel_l2: np.ndarray,
    sample_mean_s_t: np.ndarray,
    sample_max_s_t: np.ndarray,
) -> UncertaintyQualityMetrics:
    """Evaluate uncertainty calibration, correlation, and error-detection performance.

    Args:
        sample_rmse_k: 1D array of sample RMSE values in Kelvin (N,).
        sample_rel_l2: 1D array of sample relative L2 error values in % (N,).
        sample_mean_s_t: 1D array of per-sample mean standard deviation in Kelvin (N,).
        sample_max_s_t: 1D array of per-sample peak standard deviation in Kelvin (N,).

    Returns:
        UncertaintyQualityMetrics object.
    """
    rmse = np.asarray(sample_rmse_k, dtype=np.float64).ravel()
    rel_l2 = np.asarray(sample_rel_l2, dtype=np.float64).ravel()
    mean_s = np.asarray(sample_mean_s_t, dtype=np.float64).ravel()
    max_s = np.asarray(sample_max_s_t, dtype=np.float64).ravel()

    # 1. Error vs. Uncertainty Correlations
    if np.std(mean_s) > 1e-12 and np.std(rmse) > 1e-12:
        r_rmse, _ = stats.pearsonr(mean_s, rmse)
        rho_rmse, _ = stats.spearmanr(mean_s, rmse)
    else:
        r_rmse, rho_rmse = 0.0, 0.0

    if np.std(mean_s) > 1e-12 and np.std(rel_l2) > 1e-12:
        r_l2, _ = stats.pearsonr(mean_s, rel_l2)
        rho_l2, _ = stats.spearmanr(mean_s, rel_l2)
    else:
        r_l2, rho_l2 = 0.0, 0.0

    # 2. Worst-case sample identification (AUROC)
    n = len(rmse)
    k_10 = max(1, int(np.ceil(0.10 * n)))
    k_20 = max(1, int(np.ceil(0.20 * n)))

    # Labels: 1 for worst-predicted samples, 0 for remaining
    worst_10_thresh = np.partition(rmse, -k_10)[-k_10]
    worst_20_thresh = np.partition(rmse, -k_20)[-k_20]

    y_10 = (rmse >= worst_10_thresh).astype(np.int32)
    y_20 = (rmse >= worst_20_thresh).astype(np.int32)

    auroc_10, _, _, _, _, _ = compute_auroc_auprc(y_10, mean_s)
    auroc_20, _, _, _, _, _ = compute_auroc_auprc(y_20, mean_s)

    return UncertaintyQualityMetrics(
        mean_uncertainty_k=float(np.mean(mean_s)),
        max_uncertainty_k=float(np.mean(max_s)),
        pearson_r_rmse=float(r_rmse),
        spearman_rho_rmse=float(rho_rmse),
        pearson_r_rel_l2=float(r_l2),
        spearman_rho_rel_l2=float(rho_l2),
        worst_10pct_detection_auroc=float(auroc_10),
        worst_20pct_detection_auroc=float(auroc_20),
    )
