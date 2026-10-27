"""Comprehensive evaluation pipeline for Neural Process Twin surrogate models.

Computes physical metrics (relative L2, RMSE, MAE, L_infinity) on physical
temperature fields (Kelvin), performs distribution analysis, identifies worst-case
outliers, and evaluates discrete PDE residuals and boundary flux balance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_process_twin.data.boundary_encoding import decode_boundary_conditions
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.evaluation.metrics import (
    compute_aggregate_statistics,
    compute_all_metrics,
)
from neural_process_twin.evaluation.residual import evaluate_physical_residuals

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Structured evaluation results for a dataset partition.

    Attributes:
        split_name: Name of the evaluated dataset partition ('val', 'test_iid', 'test_ood').
        num_samples: Number of samples evaluated.
        aggregate_metrics: Summary statistics (mean, std, median, p95, min, max) per metric.
        sample_metrics: Individual sample metrics and identifiers.
        worst_samples: Top outlier samples ranked by highest relative L2 error.
        residual_metrics: Summary statistics for physical residuals and flux balance (if computed).
    """

    split_name: str
    num_samples: int
    aggregate_metrics: dict[str, dict[str, float]]
    sample_metrics: list[dict[str, Any]]
    worst_samples: list[dict[str, Any]]
    residual_metrics: dict[str, dict[str, float]] | None = None


def evaluate_surrogate(
    model: nn.Module,
    dataset: HeatFieldDataset | DataLoader,
    split_name: str = "evaluation",
    device: str | torch.device = "cpu",
    compute_residuals: bool = False,
    max_residual_samples: int = 50,
) -> EvaluationResult:
    """Evaluate a trained surrogate model on a dataset partition.

    Temperature predictions are mapped to physical units (Kelvin) before computing
    relative L2, RMSE, MAE, and L_infinity errors against numerical reference solutions.

    Args:
        model: Trained surrogate model.
        dataset: HeatFieldDataset instance or DataLoader.
        split_name: Name identifier for the dataset split.
        device: Computing device for model evaluation.
        compute_residuals: If True, computes discrete PDE residuals and flux balance.
        max_residual_samples: Maximum number of samples on which to evaluate PDE residuals.

    Returns:
        EvaluationResult containing aggregate metrics, sample details, and outlier diagnostics.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model.to(dev)
    model.eval()

    if isinstance(dataset, HeatFieldDataset):
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        normalizer = dataset.normalizer
        grid = dataset.grid
    elif isinstance(dataset, DataLoader):
        loader = dataset
        ds = dataset.dataset
        normalizer = getattr(ds, "normalizer", None)
        grid = getattr(ds, "grid", None)
    else:
        raise TypeError(f"Expected HeatFieldDataset or DataLoader, got {type(dataset)}.")

    sample_metrics_list: list[dict[str, Any]] = []
    residual_metrics_list: list[dict[str, float]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(dev)
            sample_id = batch.get("sample_id", [f"sample_{batch_idx}"])[0]

            # Forward pass: (1, 1, ny, nx)
            y_pred = model(x)
            y_pred_np = y_pred.squeeze(0).squeeze(0).cpu().numpy()

            # Physical temperature in Kelvin
            if normalizer is not None and normalizer.is_fitted:
                t_pred_k = normalizer.inverse_transform_temperature(y_pred_np)
            else:
                t_pred_k = y_pred_np

            # Reference temperature in Kelvin
            if "temperature_raw" in batch:
                t_ref_k = batch["temperature_raw"].squeeze(0).squeeze(0).cpu().numpy()
            elif normalizer is not None and normalizer.is_fitted:
                y_true_np = batch["y"].squeeze(0).squeeze(0).cpu().numpy()
                t_ref_k = normalizer.inverse_transform_temperature(y_true_np)
            else:
                t_ref_k = batch["y"].squeeze(0).squeeze(0).cpu().numpy()

            # Compute physical metrics in Kelvin
            metrics = compute_all_metrics(t_pred_k, t_ref_k)
            metrics["sample_id"] = sample_id
            sample_metrics_list.append(metrics)

            # Compute physical PDE residuals if requested
            if compute_residuals and batch_idx < max_residual_samples and grid is not None:
                x_np = x.squeeze(0).cpu().numpy()
                k_raw = batch["k_raw"].squeeze(0).squeeze(0).cpu().numpy()
                q_raw = batch["q_raw"].squeeze(0).squeeze(0).cpu().numpy()

                try:
                    bcs = decode_boundary_conditions(x_np, grid=grid, normalizer=normalizer)
                    res_diag = evaluate_physical_residuals(
                        grid=grid,
                        k_field=k_raw,
                        q_field=q_raw,
                        bcs=bcs,
                        temperature_pred=t_pred_k,
                    )
                    residual_metrics_list.append(res_diag)
                except Exception as exc:
                    logger.debug(f"Residual calculation skipped for sample {sample_id}: {exc}")

    # Aggregate metric statistics
    aggregate = compute_aggregate_statistics(
        [{k: v for k, v in m.items() if k != "sample_id"} for m in sample_metrics_list]
    )

    # Rank worst-case outliers by relative L2 error
    sorted_samples = sorted(sample_metrics_list, key=lambda m: m["relative_l2"], reverse=True)
    worst_samples = sorted_samples[: min(5, len(sorted_samples))]

    # Aggregate residual statistics
    residual_aggregate = (
        compute_aggregate_statistics(residual_metrics_list) if residual_metrics_list else None
    )

    return EvaluationResult(
        split_name=split_name,
        num_samples=len(sample_metrics_list),
        aggregate_metrics=aggregate,
        sample_metrics=sample_metrics_list,
        worst_samples=worst_samples,
        residual_metrics=residual_aggregate,
    )


def evaluate_all_splits(
    model: nn.Module,
    dataset_path: Path | str,
    splits: tuple[str, ...] = ("validation", "test_iid", "test_ood"),
    device: str | torch.device = "cpu",
    compute_residuals: bool = True,
) -> dict[str, EvaluationResult]:
    """Evaluate surrogate model across multiple dataset partitions.

    Args:
        model: Trained surrogate model.
        dataset_path: Path to HDF5 dataset file.
        splits: Tuple of split partition names to evaluate.
        device: Computing device.
        compute_residuals: Whether to compute discrete PDE residuals.

    Returns:
        Dictionary mapping split name to EvaluationResult.
    """
    results: dict[str, EvaluationResult] = {}
    path = Path(dataset_path)

    for split in splits:
        split_canonical = "validation" if split.lower() in {"val", "valid"} else split
        logger.info(f"Evaluating surrogate on partition '{split_canonical}' from {path.name}...")
        dataset = HeatFieldDataset(
            file_path=path,
            split=split_canonical,
            include_coords=True,
            include_bcs=True,
        )
        try:
            res = evaluate_surrogate(
                model=model,
                dataset=dataset,
                split_name=split_canonical,
                device=device,
                compute_residuals=compute_residuals,
            )
            results[split] = res
        finally:
            dataset.close()

    return results
