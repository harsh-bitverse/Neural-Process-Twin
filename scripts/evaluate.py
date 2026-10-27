#!/usr/bin/env python3
"""Surrogate model evaluation CLI for Neural Process Twin.

Evaluates trained surrogate checkpoints across validation, test_iid, and test_ood
partitions, calculating physical metrics in Kelvin (relative L2, RMSE, MAE, L_inf)
and discrete finite-volume physical residual diagnostics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

from neural_process_twin.evaluation.evaluator import evaluate_all_splits
from neural_process_twin.training.checkpoint import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate surrogate checkpoint on physical test partitions."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained model checkpoint file (.pt).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to HDF5 dataset file.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test_iid", "test_ood"],
        help="Dataset partitions to evaluate (default: val test_iid test_ood).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device ('auto', 'cpu', or 'cuda').",
    )
    parser.add_argument(
        "--compute-residuals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute discrete finite-volume PDE residuals and boundary flux balance (default: true).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination path for JSON summary report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.checkpoint.is_file():
        logger.error(f"Checkpoint file not found: {args.checkpoint}")
        return 1
    if not args.dataset.is_file():
        logger.error(f"Dataset file not found: {args.dataset}")
        return 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Loading checkpoint from {args.checkpoint} on device {device}...")
    checkpoint_data = load_checkpoint(args.checkpoint, device=device)
    model = checkpoint_data["model"]

    logger.info(
        f"Model: {type(model).__name__}, Epoch: {checkpoint_data.get('epoch')}, "
        f"Saved Val Metric: {checkpoint_data.get('val_metric')}"
    )

    results = evaluate_all_splits(
        model=model,
        dataset_path=args.dataset,
        splits=tuple(args.splits),
        device=device,
        compute_residuals=args.compute_residuals,
    )

    print("\n" + "=" * 80)
    print(f"SURROGATE EVALUATION REPORT: {args.checkpoint.name}")
    print("=" * 80)

    summary_export: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "model_architecture": checkpoint_data.get("model_config", {}).get(
            "architecture", "unknown"
        ),
        "splits": {},
    }

    for split_name, res in results.items():
        print(f"\n--- Partition: {split_name.upper()} ({res.num_samples} samples) ---")
        rel_l2 = res.aggregate_metrics.get("relative_l2", {})
        rmse = res.aggregate_metrics.get("rmse_k", {})
        mae = res.aggregate_metrics.get("mae_k", {})
        max_err = res.aggregate_metrics.get("max_error_k", {})

        print(
            f"  Relative L2 Error:  mean={rel_l2.get('mean', 0.0):.4e} ± {rel_l2.get('std', 0.0):.4e}  "
            f"[median={rel_l2.get('median', 0.0):.4e}, p95={rel_l2.get('p95', 0.0):.4e}]"
        )
        print(
            f"  RMSE (Kelvin):      mean={rmse.get('mean', 0.0):.4e} K  "
            f"[median={rmse.get('median', 0.0):.4e} K, max={rmse.get('max', 0.0):.4e} K]"
        )
        print(
            f"  MAE (Kelvin):       mean={mae.get('mean', 0.0):.4e} K  "
            f"[median={mae.get('median', 0.0):.4e} K, max={mae.get('max', 0.0):.4e} K]"
        )
        print(
            f"  Max Error (Kelvin): mean={max_err.get('mean', 0.0):.4e} K  "
            f"[p95={max_err.get('p95', 0.0):.4e} K, worst={max_err.get('max', 0.0):.4e} K]"
        )

        if res.residual_metrics:
            pde_res = res.residual_metrics.get("pde_residual_l2_w", {})
            flux_res = res.residual_metrics.get("flux_imbalance_w", {})
            print(
                f"  Discrete PDE Res:   mean={pde_res.get('mean', 0.0):.4e} W  "
                f"[median={pde_res.get('median', 0.0):.4e} W]"
            )
            print(
                f"  Flux Imbalance:     mean={flux_res.get('mean', 0.0):.4e} W  "
                f"[median={flux_res.get('median', 0.0):.4e} W]"
            )

        summary_export["splits"][split_name] = {
            "num_samples": res.num_samples,
            "aggregate_metrics": res.aggregate_metrics,
            "residual_metrics": res.residual_metrics,
            "worst_samples": res.worst_samples,
        }

    print("\n" + "=" * 80 + "\n")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary_export, f, indent=2)
        logger.info(f"Evaluation report written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
