"""Model evaluation, physical residual diagnostics, and inference benchmarking.

This package provides evaluation tools for Neural Process Twin surrogate models:
    - Physical error metrics in Kelvin: compute_relative_l2_error, compute_rmse, compute_mae,
      compute_max_absolute_error, compute_all_metrics, compute_aggregate_statistics.
    - Discrete physical PDE residuals and boundary flux balance diagnostics:
      compute_discrete_pde_residual, compute_boundary_flux_diagnostics, evaluate_physical_residuals.
    - Comprehensive partition evaluation: evaluate_surrogate, evaluate_all_splits, EvaluationResult.
    - Performance benchmarking: benchmark_surrogate, benchmark_reference_solver, compare_benchmarks,
      BenchmarkReport.
"""

from __future__ import annotations

from neural_process_twin.evaluation.benchmark import (
    BenchmarkReport,
    benchmark_reference_solver,
    benchmark_surrogate,
    compare_benchmarks,
)
from neural_process_twin.evaluation.evaluator import (
    EvaluationResult,
    evaluate_all_splits,
    evaluate_surrogate,
)
from neural_process_twin.evaluation.metrics import (
    compute_aggregate_statistics,
    compute_all_metrics,
    compute_mae,
    compute_max_absolute_error,
    compute_relative_l2_error,
    compute_rmse,
)
from neural_process_twin.evaluation.residual import (
    compute_boundary_flux_diagnostics,
    compute_discrete_pde_residual,
    evaluate_physical_residuals,
)

__all__ = [
    "BenchmarkReport",
    "EvaluationResult",
    "benchmark_reference_solver",
    "benchmark_surrogate",
    "compare_benchmarks",
    "compute_aggregate_statistics",
    "compute_all_metrics",
    "compute_boundary_flux_diagnostics",
    "compute_discrete_pde_residual",
    "compute_mae",
    "compute_max_absolute_error",
    "compute_relative_l2_error",
    "compute_rmse",
    "evaluate_all_splits",
    "evaluate_physical_residuals",
    "evaluate_surrogate",
]
