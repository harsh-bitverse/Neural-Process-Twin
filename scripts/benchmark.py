#!/usr/bin/env python3
"""Inference latency and throughput benchmarking CLI for Neural Process Twin.

Fairly benchmarks inference runtime between the conservative finite-volume reference
solver and trained neural surrogate models across single-sample latency and batched throughput.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_process_twin.evaluation.benchmark import (
    benchmark_reference_solver,
    benchmark_surrogate,
    compare_benchmarks,
)
from neural_process_twin.models.factory import build_model
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.training.checkpoint import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark inference performance between surrogate and numerical reference solver."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to trained model checkpoint (.pt). If omitted, an initialized model is used.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fno2d",
        choices=["fno2d", "cnn2d", "cnn"],
        help="Model architecture if no checkpoint is specified (default: fno2d).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        default=[32, 32],
        metavar=("NX", "NY"),
        help="Grid resolution for benchmark (default: 32 32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for throughput measurement (default: 32).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of timed execution iterations (default: 50).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device for surrogate ('auto', 'cpu', or 'cuda').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON benchmark report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nx, ny = args.grid_size

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 1. Setup surrogate model
    if args.checkpoint is not None and args.checkpoint.is_file():
        logger.info(f"Loading surrogate model from {args.checkpoint}...")
        chk = load_checkpoint(args.checkpoint, device=device)
        model = chk["model"]
        model_name = chk.get("model_config", {}).get("architecture", type(model).__name__)
    else:
        arch = "cnn2d" if args.model == "cnn" else args.model
        logger.info(f"Initializing {arch} surrogate model...")
        model_config = {
            "architecture": arch,
            "in_channels": 12,
            "out_channels": 1,
            "width": 32,
            "depth": 4,
            "modes_x": max(1, min(12, nx // 2 + 1)),
            "modes_y": max(1, min(12, ny // 2)),
        }
        model = build_model(model_config).to(device)
        model_name = arch

    # 2. Benchmark surrogate
    logger.info(
        f"Benchmarking surrogate '{model_name}' on {device} (repeats={args.repeats}, batch_size={args.batch_size})..."
    )
    surrogate_rep = benchmark_surrogate(
        model=model,
        input_shape=(1, 12, ny, nx),
        batch_size=args.batch_size,
        device=device,
        repeats=args.repeats,
    )

    # 3. Setup physical problem for reference solver
    logger.info(
        f"Benchmarking reference numerical solver on {nx}x{ny} grid (repeats={min(args.repeats, 25)})..."
    )
    grid = Grid2D(nx=nx, ny=ny, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
    rng = np.random.default_rng(42)
    k_field = 1.0 + rng.uniform(0.1, 2.0, size=(ny, nx))
    q_field = rng.uniform(0.0, 500.0, size=(ny, nx))
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )

    solver_rep = benchmark_reference_solver(
        grid=grid,
        k_field=k_field,
        q_field=q_field,
        bcs=bcs,
        repeats=min(args.repeats, 25),
    )

    comparison = compare_benchmarks(solver_rep, surrogate_rep)

    print("\n" + "=" * 80)
    print("INFERENCE BENCHMARK REPORT: REFERENCE SOLVER VS SURROGATE")
    print("=" * 80)
    print(f"Grid Resolution: {nx} x {ny} control volumes")
    print(f"Surrogate Model: {model_name} (device: {device})")
    print(f"Batch Size:      {args.batch_size}")
    print("-" * 80)
    print(
        f"Reference Numerical Solver Latency: {solver_rep.single_sample_latency_mean_ms:.3f} ms ± "
        f"{solver_rep.single_sample_latency_std_ms:.3f} ms  (throughput: {solver_rep.throughput_samples_per_sec:.1f} solves/s)"
    )
    print(
        f"Surrogate Model Latency (batch=1):  {surrogate_rep.single_sample_latency_mean_ms:.3f} ms ± "
        f"{surrogate_rep.single_sample_latency_std_ms:.3f} ms  (median: {surrogate_rep.single_sample_latency_median_ms:.3f} ms)"
    )
    print(
        f"Surrogate Batched Throughput:       {surrogate_rep.throughput_samples_per_sec:.1f} samples/s"
    )
    print("-" * 80)
    print(f"Single-Sample Latency Speedup:      {comparison['single_sample_speedup']:.2f}x")
    print(f"Batched Throughput Speedup:         {comparison['throughput_speedup']:.2f}x")
    print("=" * 80 + "\n")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_payload: dict[str, Any] = {
            "grid_size": [nx, ny],
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "reference_solver": solver_rep.__dict__,
            "surrogate": surrogate_rep.__dict__,
            "comparison": comparison,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, indent=2)
        logger.info(f"Benchmark results written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
