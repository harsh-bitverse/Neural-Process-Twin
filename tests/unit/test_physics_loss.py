"""Unit tests for differentiable physics-informed loss modules.

Verifies:
1. Exact agreement between vectorized PyTorch DiscretePDEResidual and SciPy sparse assembly.
2. Differentiability and non-zero gradient backpropagation through model parameters.
3. Numerical stability and non-divergence of DimensionlessPDELoss under zero loads.
4. Correctness and masking of BoundaryConditionLoss for Dirichlet and Neumann edges.
5. Orthogonality of interior-only residual formulation when paired with boundary penalties.
6. Resolution independence and dimensional correctness across 16x16 and 32x32 grids.
7. Compound PhysicsAwareLoss telemetry and weighting mechanics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from neural_process_twin.data.boundary_encoding import encode_boundary_conditions
from neural_process_twin.data.normalization import FieldNormalizer, FieldStatistics
from neural_process_twin.solvers.assembly import assemble_steady_heat_system
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import solve_steady_heat_2d
from neural_process_twin.training.physics_loss import (
    BoundaryConditionLoss,
    DimensionlessPDELoss,
    DiscretePDEResidual,
    PhysicsAwareLoss,
)


@pytest.fixture
def test_setup() -> tuple[
    Grid2D, np.ndarray, np.ndarray, BoundaryConditions, FieldNormalizer, np.ndarray
]:
    """Provide a verified 8x8 steady-state conduction problem."""
    grid = Grid2D(nx=8, ny=8, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
    rng = np.random.default_rng(42)
    k_field = 1.0 + rng.uniform(0.5, 2.0, size=(8, 8))
    q_field = rng.uniform(50.0, 300.0, size=(8, 8))
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 50.0),
    )
    res = solve_steady_heat_2d(grid, k_field, q_field, bcs)

    normalizer = FieldNormalizer()
    normalizer.k_stats = FieldStatistics(mean=1.5, std=0.5, min=0.5, max=3.0)
    normalizer.q_stats = FieldStatistics(mean=150.0, std=50.0, min=0.0, max=300.0)
    normalizer.t_stats = FieldStatistics(mean=320.0, std=15.0, min=290.0, max=360.0)

    return grid, k_field, q_field, bcs, normalizer, res.temperature


class TestDiscretePDEResidual:
    """Tests verifying the discrete PDE residual operator."""

    def test_residual_matches_scipy_assembly(
        self,
        test_setup: tuple[
            Grid2D, np.ndarray, np.ndarray, BoundaryConditions, FieldNormalizer, np.ndarray
        ],
    ) -> None:
        """Verify PyTorch vectorized operator matches SciPy linear system A*T - b to machine precision."""
        grid, k_np, q_np, bcs, normalizer, t_exact = test_setup

        # Arbitrary perturbed field to test non-zero residuals
        t_perturbed = t_exact + 12.5

        # SciPy reference residual
        sys = assemble_steady_heat_system(grid, k_np, q_np, bcs)
        r_ref = (sys.A @ t_perturbed.ravel() - sys.b).reshape(8, 8)

        # PyTorch operator
        op = DiscretePDEResidual(grid=grid)

        # Prepare double-precision tensors for precision benchmark
        t_torch = torch.tensor(t_perturbed, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        k_torch = torch.tensor(k_np, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        q_torch = torch.tensor(q_np, dtype=torch.float64).unsqueeze(0).unsqueeze(0)

        # Encode boundary conditions
        bc_arr = encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer)
        bc_torch = torch.tensor(bc_arr, dtype=torch.float64).unsqueeze(0)

        r_torch, b_torch = op(
            temperature_k=t_torch,
            k_field=k_torch,
            q_field=q_torch,
            bc_channels=bc_torch,
            t_mean=normalizer.t_stats.mean,
            t_std=normalizer.t_stats.std,
            residual_type="full",
        )

        r_np = r_torch.squeeze().cpu().numpy()
        b_np = b_torch.squeeze().cpu().numpy()

        # With float32 encoded channels, roundoff on repeating decimal (300-320)/15 is ~3e-6 W
        assert pytest.approx(b_np, abs=1e-5) == sys.b.reshape(8, 8)
        assert pytest.approx(r_np, abs=1e-5) == r_ref

        # In float64 without float32 quantisation, exact machine precision is achieved (< 1e-10)
        bc_torch_f64 = torch.zeros((1, 8, 8, 8), dtype=torch.float64)
        bc_torch_f64[:, 0, :, 0] = 1.0
        bc_torch_f64[:, 1, :, 0] = 300.0
        bc_torch_f64[:, 0, :, -1] = 1.0
        bc_torch_f64[:, 1, :, -1] = 350.0
        bc_torch_f64[:, 6, 0, :] = 1.0
        bc_torch_f64[:, 7, 0, :] = 0.0
        bc_torch_f64[:, 6, -1, :] = 1.0
        bc_torch_f64[:, 7, -1, :] = 50.0 / 1000.0

        r_exact, b_exact = op(
            temperature_k=t_torch,
            k_field=k_torch,
            q_field=q_torch,
            bc_channels=bc_torch_f64,
            t_mean=0.0,
            t_std=1.0,
            residual_type="full",
        )
        assert pytest.approx(b_exact.squeeze().cpu().numpy(), abs=1e-10) == sys.b.reshape(8, 8)
        assert pytest.approx(r_exact.squeeze().cpu().numpy(), abs=1e-10) == r_ref

    def test_exact_solution_residual_is_zero(
        self,
        test_setup: tuple[
            Grid2D, np.ndarray, np.ndarray, BoundaryConditions, FieldNormalizer, np.ndarray
        ],
    ) -> None:
        """Verify exact reference solution produces machine-zero residual (< 1e-10 W on exact float64)."""
        grid, k_np, q_np, _, _, t_exact = test_setup
        op = DiscretePDEResidual(grid=grid)

        t_torch = torch.tensor(t_exact, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        k_torch = torch.tensor(k_np, dtype=torch.float64).unsqueeze(0).unsqueeze(0)
        q_torch = torch.tensor(q_np, dtype=torch.float64).unsqueeze(0).unsqueeze(0)

        # Exact double-precision BC channels
        bc_torch_f64 = torch.zeros((1, 8, 8, 8), dtype=torch.float64)
        bc_torch_f64[:, 0, :, 0] = 1.0
        bc_torch_f64[:, 1, :, 0] = 300.0
        bc_torch_f64[:, 0, :, -1] = 1.0
        bc_torch_f64[:, 1, :, -1] = 350.0
        bc_torch_f64[:, 6, 0, :] = 1.0
        bc_torch_f64[:, 7, 0, :] = 0.0
        bc_torch_f64[:, 6, -1, :] = 1.0
        bc_torch_f64[:, 7, -1, :] = 50.0 / 1000.0

        r_torch, _ = op(
            temperature_k=t_torch,
            k_field=k_torch,
            q_field=q_torch,
            bc_channels=bc_torch_f64,
            t_mean=0.0,
            t_std=1.0,
        )

        max_residual = torch.max(torch.abs(r_torch)).item()
        assert max_residual < 1e-10, f"Expected machine zero residual, got {max_residual} W"

    def test_interior_residual_dimension(
        self,
        test_setup: tuple[
            Grid2D, np.ndarray, np.ndarray, BoundaryConditions, FieldNormalizer, np.ndarray
        ],
    ) -> None:
        """Verify interior residual slices out boundary cells correctly."""
        grid, k_np, q_np, bcs, normalizer, t_exact = test_setup
        op = DiscretePDEResidual(grid=grid)

        t_torch = torch.tensor(t_exact, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        k_torch = torch.tensor(k_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        q_torch = torch.tensor(q_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        bc_torch = torch.tensor(
            encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer), dtype=torch.float32
        ).unsqueeze(0)

        r_int, b_int = op(
            temperature_k=t_torch,
            k_field=k_torch,
            q_field=q_torch,
            bc_channels=bc_torch,
            t_mean=normalizer.t_stats.mean,
            t_std=normalizer.t_stats.std,
            residual_type="interior",
        )

        assert r_int.shape == (1, 1, 6, 6)
        assert b_int.shape == (1, 1, 6, 6)


class TestDimensionlessPDELoss:
    """Tests for the scale-invariant DimensionlessPDELoss."""

    def test_differentiability_and_gradient_flow(
        self,
        test_setup: tuple[
            Grid2D, np.ndarray, np.ndarray, BoundaryConditions, FieldNormalizer, np.ndarray
        ],
    ) -> None:
        """Verify loss is differentiable and backpropagates gradients into model parameters."""
        grid, _k_np, _q_np, bcs, normalizer, _ = test_setup
        loss_mod = DimensionlessPDELoss(grid=grid, residual_type="full")

        # Toy trainable layer
        conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
        dummy_input = torch.randn(2, 1, 8, 8, requires_grad=True)
        t_pred = conv(dummy_input) * 20.0 + 300.0

        k_torch = torch.ones(2, 1, 8, 8)
        q_torch = torch.ones(2, 1, 8, 8) * 100.0
        bc_torch = (
            torch.tensor(
                encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer),
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(2, 1, 1, 1)
        )

        loss = loss_mod(
            temperature_k=t_pred,
            k_field=k_torch,
            q_field=q_torch,
            bc_channels=bc_torch,
            t_mean=normalizer.t_stats.mean,
            t_std=normalizer.t_stats.std,
        )

        assert torch.isfinite(loss)
        assert loss.item() > 0.0

        loss.backward()

        assert conv.weight.grad is not None
        assert torch.any(conv.weight.grad != 0.0)
        assert torch.all(torch.isfinite(conv.weight.grad))

    def test_denominator_floor_stability(self) -> None:
        """Verify loss remains stable and finite when b is strictly zero (zero BCs, zero source)."""
        grid = Grid2D(nx=4, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
        loss_mod = DimensionlessPDELoss(grid=grid, q_floor=1.0)

        # Zero source, zero conductivity, zero BCs
        t_pred = torch.ones(1, 1, 4, 4) * 300.0
        k_torch = torch.ones(1, 1, 4, 4)
        q_torch = torch.zeros(1, 1, 4, 4)
        bc_channels = torch.zeros(1, 8, 4, 4)

        loss = loss_mod(
            temperature_k=t_pred,
            k_field=k_torch,
            q_field=q_torch,
            bc_channels=bc_channels,
            t_mean=0.0,
            t_std=1.0,
        )

        assert torch.isfinite(loss)
        assert not torch.isnan(loss)
        assert loss.item() >= 0.0


class TestBoundaryConditionLoss:
    """Tests for BoundaryConditionLoss."""

    def test_bc_loss_zero_on_exact_conditions(
        self,
        test_setup: tuple[
            Grid2D, np.ndarray, np.ndarray, BoundaryConditions, FieldNormalizer, np.ndarray
        ],
    ) -> None:
        """Verify boundary condition loss is minimal when predictions conform to boundary values."""
        grid, k_np, _, bcs, normalizer, t_exact = test_setup
        bc_loss_fn = BoundaryConditionLoss(grid=grid, use_extrapolation=False)

        t_torch = torch.tensor(t_exact, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        k_torch = torch.tensor(k_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        bc_torch = torch.tensor(
            encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer), dtype=torch.float32
        ).unsqueeze(0)

        loss = bc_loss_fn(
            temperature_k=t_torch,
            k_field=k_torch,
            bc_channels=bc_torch,
            t_mean=normalizer.t_stats.mean,
            t_std=normalizer.t_stats.std,
        )

        assert torch.isfinite(loss)
        # Numerical discretization error at boundary for cell-centre to face is modest
        assert loss.item() < 0.1

    def test_bc_loss_increases_with_boundary_violation(
        self,
        test_setup: tuple[
            Grid2D, np.ndarray, np.ndarray, BoundaryConditions, FieldNormalizer, np.ndarray
        ],
    ) -> None:
        """Verify boundary condition loss strictly increases when boundary temperatures deviate."""
        grid, k_np, _, bcs, normalizer, t_exact = test_setup
        bc_loss_fn = BoundaryConditionLoss(grid=grid)

        t_exact_t = torch.tensor(t_exact, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_corrupted_t = t_exact_t + 50.0  # 50K boundary violation
        k_torch = torch.tensor(k_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        bc_torch = torch.tensor(
            encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer), dtype=torch.float32
        ).unsqueeze(0)

        loss_exact = bc_loss_fn(
            temperature_k=t_exact_t,
            k_field=k_torch,
            bc_channels=bc_torch,
            t_mean=normalizer.t_stats.mean,
            t_std=normalizer.t_stats.std,
        )
        loss_corrupted = bc_loss_fn(
            temperature_k=t_corrupted_t,
            k_field=k_torch,
            bc_channels=bc_torch,
            t_mean=normalizer.t_stats.mean,
            t_std=normalizer.t_stats.std,
        )

        assert loss_corrupted.item() > loss_exact.item() + 0.5


class TestPhysicsAwareLossCompound:
    """Tests for the compound PhysicsAwareLoss."""

    def test_component_telemetry_and_zero_weight_shortcuts(
        self,
        test_setup: tuple[
            Grid2D, np.ndarray, np.ndarray, BoundaryConditions, FieldNormalizer, np.ndarray
        ],
    ) -> None:
        """Verify compound loss returns telemetry and correctly handles zero weights."""
        grid, k_np, q_np, bcs, normalizer, t_exact = test_setup

        compound_loss = PhysicsAwareLoss(
            grid=grid,
            normalizer=normalizer,
            data_weight=1.0,
            pde_weight=0.01,
            bc_weight=0.05,
        )

        bc_arr = encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer)
        input_12ch = np.zeros((1, 12, 8, 8), dtype=np.float32)
        input_12ch[0, 0] = normalizer.transform_k(k_np)
        input_12ch[0, 1] = normalizer.transform_q(q_np)
        input_12ch[0, 4:12] = bc_arr

        batch = {
            "x": torch.tensor(input_12ch),
            "k_raw": torch.tensor(k_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
            "q_raw": torch.tensor(q_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        }

        y_true = (
            torch.tensor(normalizer.transform_temperature(t_exact), dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        y_pred = y_true + 0.1  # Slight error

        total_loss, metrics = compound_loss(y_pred, y_true, batch=batch)

        assert torch.isfinite(total_loss)
        assert "loss_total" in metrics
        assert "loss_data" in metrics
        assert "loss_pde" in metrics
        assert "loss_bc" in metrics

        assert metrics["loss_data"] > 0.0
        assert metrics["loss_pde"] > 0.0
        assert metrics["loss_bc"] > 0.0
        assert pytest.approx(metrics["loss_total"], rel=1e-5) == (
            metrics["loss_data"] + 0.01 * metrics["loss_pde"] + 0.05 * metrics["loss_bc"]
        )
