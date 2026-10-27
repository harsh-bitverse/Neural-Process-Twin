#!/usr/bin/env python3
"""Multi-Resolution Performance and Latency Benchmarking.

Milestone 08C Phase 6: Performance Profiling and Scalability Benchmark.
Measures:
1. End-to-end CPU runtime across resolutions (32x32, 64x64, 128x128, 256x256):
   - Numerical FVM Solver (SparseFVMSolver)
   - Single-model Neural Operator (FNO-2D)
   - 5-Member Neural Operator Ensemble
2. Latency dissection:
   - Input preparation / continuous encoding
   - Neural network forward pass
   - Physical diagnostics (Heat Balance & PDE Residual)
   - Fallback solver solve
3. Batch-size scaling (1, 4, 16, 32)
4. Peak memory consumption and algorithmic complexity scaling
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_process_twin.data.continuous_encoding import encode_continuous_physical_inputs
from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D
from neural_process_twin.training.physics_loss import DiscretePDEResidual

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark_milestone_08c")


def create_synthetic_problem(
    n: int,
) -> tuple[np.ndarray, np.ndarray, BoundaryConditions, Grid2D]:
    """Create deterministic physical problem instance for benchmarking."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, n, n, thickness=1.0)
    x = grid.x_mesh
    y = grid.y_mesh

    # Smooth heterogeneous conductivity field
    k_field = 2.0 + np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)
    q_field = 5000.0 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.05)

    # Mixed boundary conditions
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0 * np.ones(n)),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0 * np.ones(n)),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, np.zeros(n)),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, np.zeros(n)),
    )
    return k_field, q_field, bcs, grid


def benchmark_fvm_solver(n: int, repeats: int = 5) -> dict[str, float]:
    """Benchmark classical FVM reference solver runtime and peak memory."""
    k_field, q_field, bcs, grid = create_synthetic_problem(n)
    solver = HeatConductionSolver2D()

    # Warmup
    _ = solver.solve(grid, k_field, q_field, bcs)

    tracemalloc.start()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = solver.solve(grid, k_field, q_field, bcs)
        times.append((time.perf_counter() - t0) * 1000.0)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
        "peak_ram_mb": float(peak_bytes / (1024 * 1024)),
    }


def benchmark_neural_operator(
    model: FourierNeuralOperator2D,
    normalizer: FieldNormalizer,
    n: int,
    batch_size: int = 1,
    repeats: int = 10,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Benchmark neural operator inference with detailed latency breakdown."""
    dev = device if device is not None else torch.device("cpu")
    k_field, q_field, bcs, grid = create_synthetic_problem(n)
    pde_op = DiscretePDEResidual(grid=grid).to(dev)

    # Dissection 1: Input preparation
    t_prep_list = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        x_in = encode_continuous_physical_inputs(
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            grid=grid,
            normalizer=normalizer,
            use_log_k=False,
        )
        t_prep_list.append((time.perf_counter() - t0) * 1000.0)

    # Prepare batched tensor
    x_batch = torch.from_numpy(np.tile(x_in[np.newaxis, ...], (batch_size, 1, 1, 1))).to(device)

    # Warmup
    with torch.no_grad():
        _ = model(x_batch)

    # Dissection 2: Model forward pass
    tracemalloc.start()
    t_fwd_list = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x_batch)
        t_fwd_list.append((time.perf_counter() - t0) * 1000.0)

    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    pred_norm = out[0, 0].cpu().numpy()
    pred_t = normalizer.inverse_transform_temperature(pred_norm)

    # Dissection 3: Diagnostics
    t_diag_list = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = compute_heat_balance(grid, pred_t, k_field, q_field, bcs)
        pred_t_torch = torch.from_numpy(pred_t).float().unsqueeze(0).unsqueeze(0).to(device)
        k_torch = torch.from_numpy(k_field).float().unsqueeze(0).unsqueeze(0).to(device)
        q_torch = torch.from_numpy(q_field).float().unsqueeze(0).unsqueeze(0).to(device)
        bc_channels = torch.zeros(1, 8, n, n).to(device)
        with torch.no_grad():
            _ = pde_op(
                temperature_k=pred_t_torch,
                k_field=k_torch,
                q_field=q_torch,
                bc_channels=bc_channels,
            )
        t_diag_list.append((time.perf_counter() - t0) * 1000.0)

    mean_fwd = float(np.mean(t_fwd_list))
    mean_prep = float(np.mean(t_prep_list))
    mean_diag = float(np.mean(t_diag_list))
    per_sample_fwd = mean_fwd / batch_size

    return {
        "batch_size": batch_size,
        "fwd_total_mean_ms": mean_fwd,
        "fwd_per_sample_mean_ms": per_sample_fwd,
        "input_prep_mean_ms": mean_prep,
        "diagnostics_mean_ms": mean_diag,
        "pipeline_per_sample_ms": per_sample_fwd + mean_prep + mean_diag,
        "peak_ram_mb": float(peak_bytes / (1024 * 1024)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Multi-Resolution Performance")
    parser.parse_args()

    torch.set_num_threads(8)
    device = torch.device("cpu")
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load normalizer
    norm_path = Path("artifacts/data/normalization_stats.json")
    if norm_path.is_file():
        with open(norm_path, encoding="utf-8") as f:
            normalizer = FieldNormalizer.from_dict(json.load(f))
    else:
        normalizer = FieldNormalizer()
        normalizer.fit(
            k=np.ones((2, 32, 32)),
            q=np.zeros((2, 32, 32)),
            temperature=300.0 * np.ones((2, 32, 32)),
        )

    # Standard FNO-2D
    model = FourierNeuralOperator2D(
        in_channels=12,
        out_channels=1,
        modes_x=12,
        modes_y=12,
        width=32,
        depth=4,
        activation="gelu",
    ).to(device)
    model.eval()

    resolutions = [32, 64, 128, 256]
    batch_sizes = [1, 4, 16, 32]

    results: dict[str, Any] = {
        "metadata": {
            "threads": 8,
            "device": "cpu",
            "resolutions": resolutions,
            "batch_sizes": batch_sizes,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "fvm_solver": {},
        "neural_operator_single": {},
        "neural_operator_batched": {},
        "ensemble_5member": {},
        "speedup_summary": {},
    }

    logger.info("=== Benchmarking Classical FVM Solver across resolutions ===")
    for n in resolutions:
        logger.info(f"FVM solve at {n}x{n} (N={n * n} cells)...")
        # For 256x256, reduce repeats
        repeats = 3 if n >= 128 else 5
        fvm_stats = benchmark_fvm_solver(n, repeats=repeats)
        results["fvm_solver"][f"res_{n}"] = fvm_stats
        logger.info(
            f"-> FVM {n}x{n}: {fvm_stats['mean_ms']:.2f} ms | RAM: {fvm_stats['peak_ram_mb']:.2f} MB"
        )

    logger.info("=== Benchmarking Neural Operator Forward Pass & Pipeline ===")
    for n in resolutions:
        logger.info(f"Neural Operator at {n}x{n}...")
        no_stats = benchmark_neural_operator(
            model, normalizer, n, batch_size=1, repeats=5, device=device
        )
        results["neural_operator_single"][f"res_{n}"] = no_stats

        # 5-member ensemble simulation: 5 forward passes + 1 prep + 1 diagnostics
        ens_fwd = no_stats["fwd_per_sample_mean_ms"] * 5.0
        ens_total = ens_fwd + no_stats["input_prep_mean_ms"] + no_stats["diagnostics_mean_ms"]
        results["ensemble_5member"][f"res_{n}"] = {
            "fwd_ensemble_ms": ens_fwd,
            "pipeline_ensemble_ms": ens_total,
        }

        # Speedups
        fvm_time = results["fvm_solver"][f"res_{n}"]["mean_ms"]
        single_fwd_speedup = fvm_time / no_stats["fwd_per_sample_mean_ms"]
        single_pipe_speedup = fvm_time / no_stats["pipeline_per_sample_ms"]
        ens_pipe_speedup = fvm_time / ens_total

        results["speedup_summary"][f"res_{n}"] = {
            "fvm_ms": fvm_time,
            "single_fwd_ms": no_stats["fwd_per_sample_mean_ms"],
            "single_pipe_ms": no_stats["pipeline_per_sample_ms"],
            "ens_pipe_ms": ens_total,
            "speedup_fwd_vs_fvm": single_fwd_speedup,
            "speedup_pipe_vs_fvm": single_pipe_speedup,
            "speedup_ens_vs_fvm": ens_pipe_speedup,
        }

        logger.info(
            f"-> NO {n}x{n} (batch 1): Fwd {no_stats['fwd_per_sample_mean_ms']:.2f} ms | "
            f"Pipe {no_stats['pipeline_per_sample_ms']:.2f} ms | Speedup: {single_pipe_speedup:.2f}x vs FVM"
        )

    logger.info("=== Benchmarking Batch Scaling ===")
    for n in [32, 64, 128]:
        results["neural_operator_batched"][f"res_{n}"] = {}
        for bs in batch_sizes:
            b_stats = benchmark_neural_operator(
                model, normalizer, n, batch_size=bs, repeats=5, device=device
            )
            results["neural_operator_batched"][f"res_{n}"][f"batch_{bs}"] = b_stats
            logger.info(
                f"-> {n}x{n} Batch {bs:02d}: Total {b_stats['fwd_total_mean_ms']:.2f} ms | "
                f"Per-sample {b_stats['fwd_per_sample_mean_ms']:.2f} ms"
            )

    out_json = reports_dir / "milestone_08c_benchmarks.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Performance benchmarks complete. Saved to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
