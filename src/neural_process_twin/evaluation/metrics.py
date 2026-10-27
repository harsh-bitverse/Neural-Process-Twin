"""Statistical and physical error metrics for solution field evaluation.

All physical metrics (RMSE, MAE, L_infinity) operate on unnormalized temperature
fields and report values in physical units (Kelvin).
"""

from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor or NumPy array to contiguous float64 NumPy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def compute_relative_l2_error(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
    eps: float = 1e-7,
) -> float:
    """Compute relative L2 error norm: ||y_pred - y_true||_2 / (||y_true||_2 + eps).

    Args:
        y_pred: Predicted physical field array or tensor.
        y_true: Ground-truth reference field array or tensor.
        eps: Small positive constant preventing zero division.

    Returns:
        Dimensionless relative L2 error scalar.
    """
    pred = _to_numpy(y_pred)
    true = _to_numpy(y_true)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: y_pred {pred.shape} != y_true {true.shape}.")

    diff_norm = np.linalg.norm(pred - true)
    ref_norm = np.linalg.norm(true)
    return float(diff_norm / (ref_norm + eps))


def compute_rmse(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
) -> float:
    """Compute Root Mean Square Error (RMSE) in physical units (Kelvin).

    Args:
        y_pred: Predicted temperature field [K].
        y_true: Reference temperature field [K].

    Returns:
        RMSE in Kelvin [K].
    """
    pred = _to_numpy(y_pred)
    true = _to_numpy(y_true)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: y_pred {pred.shape} != y_true {true.shape}.")

    return float(np.sqrt(np.mean((pred - true) ** 2)))


def compute_mae(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
) -> float:
    """Compute Mean Absolute Error (MAE) in physical units (Kelvin).

    Args:
        y_pred: Predicted temperature field [K].
        y_true: Reference temperature field [K].

    Returns:
        MAE in Kelvin [K].
    """
    pred = _to_numpy(y_pred)
    true = _to_numpy(y_true)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: y_pred {pred.shape} != y_true {true.shape}.")

    return float(np.mean(np.abs(pred - true)))


def compute_max_absolute_error(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
) -> float:
    """Compute Maximum Absolute Error (L_infinity norm) in physical units (Kelvin).

    Args:
        y_pred: Predicted temperature field [K].
        y_true: Reference temperature field [K].

    Returns:
        Max absolute error in Kelvin [K].
    """
    pred = _to_numpy(y_pred)
    true = _to_numpy(y_true)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: y_pred {pred.shape} != y_true {true.shape}.")

    return float(np.max(np.abs(pred - true)))


def compute_all_metrics(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Calculate complete suite of scientific evaluation metrics.

    Args:
        y_pred: Predicted temperature field in Kelvin [K].
        y_true: Reference temperature field in Kelvin [K].
        eps: Small constant for relative L2 denominator.

    Returns:
        Dictionary mapping metric names to scalar values.
    """
    return {
        "relative_l2": compute_relative_l2_error(y_pred, y_true, eps=eps),
        "rmse_k": compute_rmse(y_pred, y_true),
        "mae_k": compute_mae(y_pred, y_true),
        "max_error_k": compute_max_absolute_error(y_pred, y_true),
    }


def compute_aggregate_statistics(
    metrics_list: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Compute distribution statistics (mean, std, median, p95, min, max) across samples.

    Args:
        metrics_list: List of metric dictionaries per sample.

    Returns:
        Nested dictionary with statistics for each metric.
    """
    if not metrics_list:
        return {}

    keys = metrics_list[0].keys()
    aggregated: dict[str, dict[str, float]] = {}

    for k in keys:
        vals = np.array([m[k] for m in metrics_list], dtype=np.float64)
        aggregated[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    return aggregated
