"""Unit tests for scientific evaluation metrics, physical residuals, and benchmarking."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_process_twin.evaluation.benchmark import (
    benchmark_reference_solver,
    benchmark_surrogate,
    compare_benchmarks,
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
from neural_process_twin.models.cnn import CNNBaseline2D
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import solve_steady_heat_2d


class TestPhysicalMetrics:
    """Test suite for physical evaluation metrics in Kelvin."""

    def test_hand_computed_relative_l2(self) -> None:
        y_true = np.array([[2.0, 2.0], [2.0, 2.0]])
        y_pred = np.array([[3.0, 3.0], [3.0, 3.0]])
        # diff norm = sqrt(4 * 1) = 2, ref norm = sqrt(4 * 4) = 4, rel = 0.5
        rel_l2 = compute_relative_l2_error(y_pred, y_true)
        assert pytest.approx(rel_l2, rel=1e-5) == 0.5

    def test_rmse_physical_units(self) -> None:
        y_true = np.full((10, 10), 300.0)  # 300 K
        y_pred = np.full((10, 10), 303.0)  # 303 K
        # RMSE should be exactly 3.0 K
        assert pytest.approx(compute_rmse(y_pred, y_true), rel=1e-5) == 3.0

    def test_mae_physical_units(self) -> None:
        y_true = np.array([300.0, 305.0])
        y_pred = np.array([301.0, 303.0])  # errors: 1.0, 2.0 -> MAE = 1.5 K
        assert pytest.approx(compute_mae(y_pred, y_true), rel=1e-5) == 1.5

    def test_max_absolute_error(self) -> None:
        y_true = np.array([[300.0, 300.0], [300.0, 300.0]])
        y_pred = np.array([[300.5, 300.2], [308.0, 300.1]])  # max error = 8.0 K
        assert pytest.approx(compute_max_absolute_error(y_pred, y_true), rel=1e-5) == 8.0

    def test_compute_all_metrics(self) -> None:
        y_true = torch.full((1, 8, 8), 300.0)
        y_pred = torch.full((1, 8, 8), 302.0)
        metrics = compute_all_metrics(y_pred, y_true)

        assert "relative_l2" in metrics
        assert "rmse_k" in metrics
        assert "mae_k" in metrics
        assert "max_error_k" in metrics
        assert pytest.approx(metrics["rmse_k"], rel=1e-5) == 2.0
        assert pytest.approx(metrics["mae_k"], rel=1e-5) == 2.0

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_rmse(np.zeros((4, 4)), np.zeros((4, 5)))

    def test_aggregate_statistics(self) -> None:
        samples = [
            {"relative_l2": 0.01, "rmse_k": 1.0},
            {"relative_l2": 0.03, "rmse_k": 3.0},
            {"relative_l2": 0.02, "rmse_k": 2.0},
        ]
        stats = compute_aggregate_statistics(samples)
        assert pytest.approx(stats["rmse_k"]["mean"], rel=1e-5) == 2.0
        assert pytest.approx(stats["rmse_k"]["min"], rel=1e-5) == 1.0
        assert pytest.approx(stats["rmse_k"]["max"], rel=1e-5) == 3.0
        assert pytest.approx(stats["relative_l2"]["mean"], rel=1e-5) == 0.02


class TestDiscreteResiduals:
    """Test suite for finite-volume PDE residuals and boundary flux balance."""

    @pytest.fixture
    def setup_problem(
        self,
    ) -> tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray]:
        grid = Grid2D(nx=16, ny=16, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
        k_field = np.full((16, 16), 2.5)
        q_field = np.full((16, 16), 500.0)
        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
        )
        sol = solve_steady_heat_2d(grid, k_field, q_field, bcs)
        return grid, k_field, q_field, bcs, sol.temperature

    def test_reference_solution_residual_near_zero(
        self,
        setup_problem: tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray],
    ) -> None:
        grid, k_field, q_field, bcs, t_exact = setup_problem
        res = compute_discrete_pde_residual(grid, k_field, q_field, bcs, t_exact)
        assert res["pde_residual_l2_w"] < 1e-8
        assert res["pde_residual_relative"] < 1e-8

    def test_perturbed_solution_residual_positive(
        self,
        setup_problem: tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray],
    ) -> None:
        grid, k_field, q_field, bcs, t_exact = setup_problem
        t_perturbed = t_exact + 10.0  # Constant shift violates Dirichlet boundaries
        res = compute_discrete_pde_residual(grid, k_field, q_field, bcs, t_perturbed)
        assert res["pde_residual_l2_w"] > 1.0

    def test_boundary_flux_diagnostics(
        self,
        setup_problem: tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray],
    ) -> None:
        grid, k_field, q_field, bcs, t_exact = setup_problem
        flux_diag = compute_boundary_flux_diagnostics(grid, k_field, q_field, bcs, t_exact)
        assert flux_diag["flux_imbalance_w"] < 1e-8

        # Combined function test
        all_res = evaluate_physical_residuals(grid, k_field, q_field, bcs, t_exact)
        assert "pde_residual_l2_w" in all_res
        assert "flux_imbalance_w" in all_res


class TestBenchmarkModule:
    """Test suite for latency and throughput benchmarking."""

    def test_surrogate_benchmark(self) -> None:
        model = CNNBaseline2D(in_channels=12, out_channels=1, width=8, depth=2)
        report = benchmark_surrogate(
            model=model,
            input_shape=(1, 12, 16, 16),
            batch_size=4,
            device="cpu",
            warmup=2,
            repeats=5,
        )
        assert report.single_sample_latency_mean_ms > 0.0
        assert report.throughput_samples_per_sec > 0.0

    def test_reference_solver_benchmark(self) -> None:
        grid = Grid2D(nx=8, ny=8, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
        k_field = np.ones((8, 8))
        q_field = np.zeros((8, 8))
        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
        )
        report = benchmark_reference_solver(
            grid=grid,
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            warmup=1,
            repeats=3,
        )
        assert report.single_sample_latency_mean_ms > 0.0

    def test_compare_benchmarks(self) -> None:
        from neural_process_twin.evaluation.benchmark import BenchmarkReport

        sol_rep = BenchmarkReport("Solver", "cpu", 1, 10.0, 0.5, 10.0, 11.0, 100.0)
        sur_rep = BenchmarkReport("Surrogate", "cpu", 32, 1.0, 0.05, 1.0, 1.1, 1000.0)
        comp = compare_benchmarks(sol_rep, sur_rep)

        assert pytest.approx(comp["single_sample_speedup"], rel=1e-5) == 10.0
        assert pytest.approx(comp["throughput_speedup"], rel=1e-5) == 10.0
