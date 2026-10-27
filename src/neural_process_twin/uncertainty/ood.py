"""Out-of-distribution (OOD) detection baselines and evaluation metrics.

Provides physics-aware input domain diagnostics, ensemble disagreement detection,
and formal binary classification metrics (AUROC, AUPRC, TPR@FPR=5%, F1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class OODDetectionResult:
    """Quantitative performance metrics for an OOD detection score.

    Attributes:
        detector_name: Identifying name of the detector baseline.
        num_iid: Number of in-distribution evaluation samples (negative class, label 0).
        num_ood: Number of out-of-distribution evaluation samples (positive class, label 1).
        auroc: Area Under Receiver Operating Characteristic curve in [0, 1].
        auprc: Area Under Precision-Recall Curve in [0, 1].
        threshold: Decision threshold selected at the specified in-distribution FPR.
        tpr_at_5pct_fpr: True Positive Rate (recall) when FPR is constrained to 5% (95% specificity).
        optimal_f1: Maximum achievable F1 score across all potential operating thresholds.
        accuracy_at_threshold: Classification accuracy at the selected operating threshold.
    """

    detector_name: str
    num_iid: int
    num_ood: int
    auroc: float
    auprc: float
    threshold: float
    tpr_at_5pct_fpr: float
    optimal_f1: float
    accuracy_at_threshold: float


def compute_auroc_auprc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """Compute AUROC, AUPRC, TPR@5%FPR, optimal F1, and threshold from binary scores.

    Args:
        y_true: 1D binary labels (0 for IID, 1 for OOD).
        y_scores: 1D continuous anomaly scores (higher indicates more likely OOD).

    Returns:
        Tuple of (auroc, auprc, threshold_at_5pct_fpr, tpr_at_5pct_fpr, optimal_f1, acc_at_thresh).
    """
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_scores = np.asarray(y_scores, dtype=np.float64).ravel()

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return 0.5, float(n_pos) / max(len(y_true), 1), 0.0, 0.0, 0.0, 0.0

    # Sort descending by score
    desc_idx = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_idx]
    y_scores_sorted = y_scores[desc_idx]

    # Unique thresholds
    distinct_mask = np.diff(y_scores_sorted) != 0
    threshold_idxs = np.where(distinct_mask)[0]
    threshold_idxs = np.concatenate([threshold_idxs, [len(y_scores_sorted) - 1]])

    tps = np.cumsum(y_true_sorted == 1)[threshold_idxs]
    fps = np.cumsum(y_true_sorted == 0)[threshold_idxs]

    tpr = tps / n_pos
    fpr = fps / n_neg
    precision = tps / (tps + fps)

    # 1. AUROC via trapezoidal integration: add (0, 0)
    tpr_roc = np.concatenate([[0.0], tpr, [1.0]])
    fpr_roc = np.concatenate([[0.0], fpr, [1.0]])
    sort_fpr_idx = np.argsort(fpr_roc)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auroc = float(trapz_fn(tpr_roc[sort_fpr_idx], fpr_roc[sort_fpr_idx]))
    auroc = max(0.0, min(1.0, auroc))

    # 2. AUPRC via trapezoidal integration of precision-recall
    rec_prc = np.concatenate([[0.0], tpr])
    prec_prc = np.concatenate([[1.0], precision])
    sort_rec_idx = np.argsort(rec_prc)
    auprc = float(trapz_fn(prec_prc[sort_rec_idx], rec_prc[sort_rec_idx]))
    auprc = max(0.0, min(1.0, auprc))

    # 3. TPR at 5% FPR
    # Find operating point where FPR <= 0.05
    valid_fps = np.where(fpr <= 0.05)[0]
    if len(valid_fps) > 0:
        best_fpr_idx = valid_fps[-1]
        tpr_at_5pct_fpr = float(tpr[best_fpr_idx])
        threshold_at_5pct = float(y_scores_sorted[threshold_idxs[best_fpr_idx]])
    else:
        tpr_at_5pct_fpr = float(tpr[0])
        threshold_at_5pct = float(y_scores_sorted[0])

    # 4. Optimal F1
    f1_scores = 2.0 * (precision * tpr) / (precision + tpr + 1e-12)
    optimal_f1 = float(np.max(f1_scores))

    # 5. Accuracy at threshold
    pred_binary = (y_scores >= threshold_at_5pct).astype(np.int32)
    acc_at_thresh = float(np.mean(pred_binary == y_true))

    return (
        auroc,
        auprc,
        threshold_at_5pct,
        tpr_at_5pct_fpr,
        optimal_f1,
        acc_at_thresh,
    )


def evaluate_ood_detection(
    iid_scores: np.ndarray | torch.Tensor,
    ood_scores: np.ndarray | torch.Tensor,
    detector_name: str = "detector",
) -> OODDetectionResult:
    """Evaluate binary classification performance distinguishing IID (0) from OOD (1).

    Args:
        iid_scores: 1D array of scores for in-distribution samples (N_iid,).
        ood_scores: 1D array of scores for out-of-distribution samples (N_ood,).
        detector_name: Name label for reporting.

    Returns:
        OODDetectionResult with AUROC, AUPRC, and operating metrics.
    """
    s_iid = np.asarray(iid_scores, dtype=np.float64).ravel()
    s_ood = np.asarray(ood_scores, dtype=np.float64).ravel()

    y_true = np.concatenate([np.zeros(len(s_iid)), np.ones(len(s_ood))])
    y_scores = np.concatenate([s_iid, s_ood])

    auroc, auprc, thresh, tpr_5, f1, acc = compute_auroc_auprc(y_true, y_scores)

    return OODDetectionResult(
        detector_name=detector_name,
        num_iid=len(s_iid),
        num_ood=len(s_ood),
        auroc=auroc,
        auprc=auprc,
        threshold=thresh,
        tpr_at_5pct_fpr=tpr_5,
        optimal_f1=f1,
        accuracy_at_threshold=acc,
    )


class EnsembleDisagreementDetector:
    """OOD detector based on spatial predictive standard deviation across ensemble members."""

    def __init__(self, aggregation: str = "mean") -> None:
        """Initialize EnsembleDisagreementDetector.

        Args:
            aggregation: 'mean' (spatial average s_T) or 'max' (peak spatial s_T).
        """
        if aggregation not in ("mean", "max"):
            raise ValueError(f"aggregation must be 'mean' or 'max', got '{aggregation}'.")
        self.aggregation = aggregation

    def score(self, std_kelvin: np.ndarray | torch.Tensor) -> np.ndarray:
        """Compute anomaly score from predictive standard deviation fields in Kelvin.

        Args:
            std_kelvin: Tensor of shape (B, 1, ny, nx) or (B, ny, nx).

        Returns:
            1D array of anomaly scores of shape (B,).
        """
        s = np.asarray(std_kelvin, dtype=np.float64)
        if s.ndim == 4:
            s = s.squeeze(1)
        if self.aggregation == "mean":
            return np.mean(s, axis=(-2, -1))
        return np.max(s, axis=(-2, -1))


class InputDistributionDetector:
    """Physics-aware OOD detector inspecting physical input fields (k contrast, max q)."""

    def __init__(
        self,
        mode: str = "contrast",
        k_contrast_train_max: float = 7.16,
        q_train_max: float = 912.8,
    ) -> None:
        """Initialize InputDistributionDetector.

        Args:
            mode: 'contrast' (k_max / k_min), 'heat_source' (max q), or 'composite' (normalized sum).
            k_contrast_train_max: Maximum contrast observed during training (default 7.16).
            q_train_max: Maximum volumetric heat source observed during training (default 912.8 W/m^3).
        """
        if mode not in ("contrast", "heat_source", "composite"):
            raise ValueError(
                f"Unsupported mode '{mode}'. Must be 'contrast', 'heat_source', or 'composite'."
            )
        self.mode = mode
        self.k_contrast_train_max = float(k_contrast_train_max)
        self.q_train_max = float(q_train_max)

    def score(
        self,
        k_field: np.ndarray | torch.Tensor,
        q_field: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Compute input domain anomaly scores for a batch of samples.

        Args:
            k_field: Conductivity field [W/(m K)] of shape (B, 1, ny, nx) or (B, ny, nx).
            q_field: Heat source field [W/m^3] of shape (B, 1, ny, nx) or (B, ny, nx).

        Returns:
            1D array of anomaly scores of shape (B,).
        """
        k = np.asarray(k_field, dtype=np.float64)
        q = np.asarray(q_field, dtype=np.float64)
        if k.ndim == 4:
            k = k.squeeze(1)
        if q.ndim == 4:
            q = q.squeeze(1)

        k_min = np.min(k, axis=(-2, -1))
        k_max = np.max(k, axis=(-2, -1))
        contrast = k_max / np.maximum(k_min, 1e-6)

        q_max = np.max(q, axis=(-2, -1))

        if self.mode == "contrast":
            return contrast
        if self.mode == "heat_source":
            return q_max
        # Composite normalized score
        norm_contrast = contrast / self.k_contrast_train_max
        norm_q = q_max / max(self.q_train_max, 1.0)
        return np.maximum(norm_contrast, norm_q)
