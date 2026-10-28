"""Scientific verification and regression tests for Milestone 08F.

Tests:
1. LogSumExpMaxLoss: numerical stability, monotonicity, approximation to Chebyshev norm.
2. BoundaryLayerLoss: spatial masking, selectivity for boundary ring errors.
3. InterfaceGradientLoss: conductivity gradient weighting and scale invariance.
4. RefinedLocalizedLoss: composite loss calculation and gradient flow.
5. RefinedFNO2D: multi-resolution shape consistency, parameter counting, and warm-start residual property.
6. JacobiPhysicsSmoother: residual reduction on localized perturbations.
"""

from __future__ import annotations

import numpy as np
import torch

from neural_process_twin.models.refined_fno import RefinedFNO2D
from neural_process_twin.solvers.assembly import assemble_steady_heat_system
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.physics_smoother import JacobiPhysicsSmoother
from neural_process_twin.training.refined_loss import (
    BoundaryLayerLoss,
    InterfaceGradientLoss,
    LogSumExpMaxLoss,
    RefinedLocalizedLoss,
)


class TestRefinedLosses:
    """Tests for Milestone 08F localized loss components."""

    def test_log_sum_exp_max_loss(self) -> None:
        """Verify numerical stability and convergence of smooth-max to MaxAE."""
        loss_fn = LogSumExpMaxLoss(beta=5.0)

        pred = torch.zeros(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)
        # Introduce single localized spike of 20.0
        pred[0, 0, 8, 8] = 20.0

        loss = loss_fn(pred, target)
        val = loss.item()

        # Smooth-max should be between mean (20/256 = 0.078) and exact max (20.0), close to 20.0
        assert val > 15.0
        assert val <= 20.0 + 1e-4

        # Monotonicity test: doubling the spike must increase loss
        pred2 = torch.zeros(1, 1, 16, 16)
        pred2[0, 0, 8, 8] = 40.0
        loss2 = loss_fn(pred2, target)
        assert loss2.item() > val

    def test_boundary_layer_loss(self) -> None:
        """Verify boundary loss isolates perimeter cells."""
        loss_fn = BoundaryLayerLoss(ring_width=2)

        # 1. Error only in deep interior (cell 8, 8 in 16x16 grid)
        pred_int = torch.zeros(1, 1, 16, 16)
        target = torch.zeros(1, 1, 16, 16)
        pred_int[0, 0, 8, 8] = 10.0
        loss_int = loss_fn(pred_int, target)
        assert loss_int.item() == 0.0

        # 2. Error on boundary (cell 0, 5)
        pred_bnd = torch.zeros(1, 1, 16, 16)
        pred_bnd[0, 0, 0, 5] = 10.0
        loss_bnd = loss_fn(pred_bnd, target)
        assert loss_bnd.item() > 0.0

    def test_interface_gradient_loss(self) -> None:
        """Verify errors at material jumps receive higher loss weight."""
        loss_fn = InterfaceGradientLoss(alpha=3.0)

        # Create conductivity field with step jump along x = 8
        k_field = torch.ones(1, 1, 16, 16)
        k_field[:, :, :, 8:] = 10.0  # contrast = 10

        target = torch.zeros(1, 1, 16, 16)

        # Error at uniform region (x = 2)
        pred_uniform = torch.zeros(1, 1, 16, 16)
        pred_uniform[0, 0, 8, 2] = 5.0
        loss_uniform = loss_fn(pred_uniform, target, k_field=k_field)

        # Equal error at interface region (x = 8)
        pred_interface = torch.zeros(1, 1, 16, 16)
        pred_interface[0, 0, 8, 8] = 5.0
        loss_interface = loss_fn(pred_interface, target, k_field=k_field)

        assert loss_interface.item() > loss_uniform.item()

    def test_composite_refined_loss_backprop(self) -> None:
        """Verify composite loss backward pass computes non-zero gradients."""
        loss_fn = RefinedLocalizedLoss(lambda_max=0.05, lambda_bnd=0.10, lambda_grad=0.05)

        pred = torch.randn(2, 1, 16, 16, requires_grad=True)
        target = torch.randn(2, 1, 16, 16)
        k_field = torch.ones(2, 1, 16, 16)

        total_loss, metrics = loss_fn(pred, target, k_field=k_field)
        assert total_loss.item() > 0.0
        assert "loss_smooth_max" in metrics
        assert "loss_bnd" in metrics

        total_loss.backward()
        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))


class TestRefinedFNO2DArchitecture:
    """Tests for RefinedFNO2D hybrid architecture."""

    def test_shape_and_multiresolution(self) -> None:
        """Verify RefinedFNO2D processes different spatial resolutions without reconfiguration."""
        model = RefinedFNO2D(
            in_channels=12,
            out_channels=1,
            modes_x=8,
            modes_y=8,
            width=16,
            depth=2,
            refinement_hidden=16,
            refinement_layers=2,
        )
        model.eval()

        for res in [16, 32, 64]:
            x = torch.randn(1, 12, res, res)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 1, res, res)
            assert torch.all(torch.isfinite(out))

    def test_parameter_count(self) -> None:
        """Verify parameter counting."""
        model = RefinedFNO2D(
            in_channels=12,
            out_channels=1,
            modes_x=12,
            modes_y=12,
            width=32,
            depth=4,
            refinement_hidden=32,
            refinement_layers=3,
        )
        params = model.get_num_parameters()
        assert params > 100_000

    def test_warm_start_residual_initialization(self) -> None:
        """Verify initialized refinement block produces near-zero correction on forward pass."""
        model = RefinedFNO2D(
            in_channels=12,
            out_channels=1,
            modes_x=8,
            modes_y=8,
            width=16,
            depth=2,
        )
        model.eval()
        x = torch.randn(1, 12, 16, 16)

        with torch.no_grad():
            y_total, y_fno, delta_local = model.forward_decomposed(x)

        # Delta should be negligible due to 1e-3 weight init on final conv
        assert torch.max(torch.abs(delta_local)).item() < 0.05
        assert torch.allclose(y_total, y_fno, atol=0.05)


class TestJacobiPhysicsSmoother:
    """Tests for JacobiPhysicsSmoother."""

    def test_residual_reduction(self) -> None:
        """Verify Jacobi smoothing strictly reduces PDE residual on perturbed field."""
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, 32, 32, thickness=1.0)
        k_field = np.ones(grid.shape)
        q_field = np.zeros(grid.shape)
        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, 300.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, 300.0),
        )

        system = assemble_steady_heat_system(grid, k_field, q_field, bcs)

        # Linear initial guess with localized perturbation
        t_init = np.linspace(300.0, 350.0, 32).reshape(1, 32).repeat(32, axis=0)
        t_init[16, 16] += 25.0  # 25K hotspot perturbation

        # Initial residual L2
        r_init = np.linalg.norm(system.A @ t_init.reshape(-1) - system.b)

        smoother = JacobiPhysicsSmoother(omega=0.67, max_iters=3)
        t_smoothed = smoother.smooth(t_init, grid, k_field, q_field, bcs, num_iters=3)

        r_smooth = np.linalg.norm(system.A @ t_smoothed.reshape(-1) - system.b)

        # Residual must strictly decrease
        assert r_smooth < r_init
        # The 25K hotspot must be damped
        assert t_smoothed[16, 16] < t_init[16, 16]
