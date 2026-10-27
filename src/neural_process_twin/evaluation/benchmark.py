"""Inference latency and throughput benchmarking comparing surrogates against numerical reference solver.

Measures single-sample latency (batch size 1) and batched throughput with warmup passes,
high-resolution timers, and synchronization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from neural_process_twin.solvers.boundary import BoundaryConditions
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import solve_steady_heat_2d


@dataclass(frozen=True)
class BenchmarkReport:
    """Benchmark results for a model or solver.

    Attributes:
        name: Name identifier of the evaluated method.
        device: Device on which evaluation took place ('cpu', 'cuda').
        batch_size: Batch size used for throughput measurement.
        single_sample_latency_mean_ms: Mean single-sample latency in milliseconds.
        single_sample_latency_std_ms: Latency standard deviation in milliseconds.
        single_sample_latency_median_ms: Median latency in milliseconds.
        single_sample_latency_p95_ms: 95th percentile latency in milliseconds.
        throughput_samples_per_sec: Batched throughput in samples per second.
    """

    name: str
    device: str
    batch_size: int
    single_sample_latency_mean_ms: float
    single_sample_latency_std_ms: float
    single_sample_latency_median_ms: float
    single_sample_latency_p95_ms: float
    throughput_samples_per_sec: float


def _sync_device(device: torch.device) -> None:
    """Synchronize device for accurate timing."""
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_surrogate(
    model: nn.Module,
    input_shape: tuple[int, int, int, int],  # (1, in_channels, ny, nx)
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    warmup: int = 10,
    repeats: int = 50,
) -> BenchmarkReport:
    """Benchmark single-sample latency and batched throughput of a PyTorch surrogate.

    Args:
        model: Surrogate model to benchmark.
        input_shape: 4D shape tuple (1, in_channels, ny, nx).
        batch_size: Batch size for batched throughput benchmark.
        device: Device to execute benchmark on.
        warmup: Number of warmup iterations.
        repeats: Number of measured timing iterations.

    Returns:
        BenchmarkReport containing latency and throughput metrics.
    """
    dev = torch.device(device) if isinstance(device, str) else device
    model.to(dev)
    model.eval()

    _, c, ny, nx = input_shape
    x_single = torch.randn(1, c, ny, nx, device=dev)
    x_batch = torch.randn(batch_size, c, ny, nx, device=dev)

    # 1. Warmup single-sample
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x_single)
        _sync_device(dev)

        # 2. Measure single-sample latency
        latencies_ms = []
        for _ in range(repeats):
            _sync_device(dev)
            t0 = time.perf_counter()
            _ = model(x_single)
            _sync_device(dev)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        # 3. Warmup batched
        for _ in range(warmup // 2 + 1):
            _ = model(x_batch)
        _sync_device(dev)

        # 4. Measure batched throughput
        _sync_device(dev)
        t_batch_start = time.perf_counter()
        for _ in range(repeats):
            _ = model(x_batch)
        _sync_device(dev)
        t_batch_total = time.perf_counter() - t_batch_start

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    total_samples = repeats * batch_size
    throughput = float(total_samples / max(t_batch_total, 1e-9))

    return BenchmarkReport(
        name=type(model).__name__,
        device=dev.type,
        batch_size=batch_size,
        single_sample_latency_mean_ms=float(np.mean(lat_arr)),
        single_sample_latency_std_ms=float(np.std(lat_arr)),
        single_sample_latency_median_ms=float(np.median(lat_arr)),
        single_sample_latency_p95_ms=float(np.percentile(lat_arr, 95)),
        throughput_samples_per_sec=throughput,
    )


def benchmark_reference_solver(
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
    warmup: int = 3,
    repeats: int = 20,
) -> BenchmarkReport:
    """Benchmark single-sample latency of the numerical reference solver.

    Args:
        grid: Computational grid.
        k_field: 2D conductivity field.
        q_field: 2D heat source field.
        bcs: Prescribed BoundaryConditions.
        warmup: Number of warmup solves.
        repeats: Number of measured solve iterations.

    Returns:
        BenchmarkReport for the numerical reference solver.
    """
    # 1. Warmup
    for _ in range(warmup):
        _ = solve_steady_heat_2d(grid, k_field, q_field, bcs)

    # 2. Timing
    latencies_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = solve_steady_heat_2d(grid, k_field, q_field, bcs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    mean_lat_s = np.mean(lat_arr) / 1000.0
    throughput = float(1.0 / max(mean_lat_s, 1e-9))

    return BenchmarkReport(
        name="FiniteVolumeReferenceSolver",
        device="cpu",
        batch_size=1,
        single_sample_latency_mean_ms=float(np.mean(lat_arr)),
        single_sample_latency_std_ms=float(np.std(lat_arr)),
        single_sample_latency_median_ms=float(np.median(lat_arr)),
        single_sample_latency_p95_ms=float(np.percentile(lat_arr, 95)),
        throughput_samples_per_sec=throughput,
    )


def compare_benchmarks(
    solver_report: BenchmarkReport,
    surrogate_report: BenchmarkReport,
) -> dict[str, Any]:
    """Calculate speedup factors comparing surrogate against reference solver.

    Args:
        solver_report: Benchmark report for the reference numerical solver.
        surrogate_report: Benchmark report for the surrogate model.

    Returns:
        Dictionary with speedup metrics and comparison statistics.
    """
    solver_lat = solver_report.single_sample_latency_mean_ms
    surrogate_lat = surrogate_report.single_sample_latency_mean_ms
    single_speedup = solver_lat / max(surrogate_lat, 1e-9)

    solver_tput = solver_report.throughput_samples_per_sec
    surrogate_tput = surrogate_report.throughput_samples_per_sec
    throughput_speedup = surrogate_tput / max(solver_tput, 1e-9)

    return {
        "reference_solver_latency_ms": solver_lat,
        "surrogate_latency_ms": surrogate_lat,
        "single_sample_speedup": single_speedup,
        "reference_solver_throughput_sps": solver_tput,
        "surrogate_throughput_sps": surrogate_tput,
        "throughput_speedup": throughput_speedup,
    }
