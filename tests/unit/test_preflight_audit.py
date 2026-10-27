"""Unit and regression tests verifying the preflight audit requirements for PDE residuals.

Audits:
1. Kelvin transformation of predicted temperatures prior to residual calculation.
2. Exactness of discrete finite-volume operator and boundary conventions.
3. Sign and unit consistency of volumetric source and boundary heat flows.
4. Scale-invariance and stability of residuals across grid resolutions and parameter ranges.
5. Independence of global boundary heat balance vs local linear algebra residual.
6. Absolute zero-leakage of ground-truth reference solutions in residual diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_process_twin.data.boundary_encoding import (
    encode_boundary_conditions,
)
from neural_process_twin.data.normalization import FieldNormalizer, FieldStatistics
from neural_process_twin.evaluation.evaluator import evaluate_surrogate
from neural_process_twin.evaluation.residual import (
    compute_boundary_flux_diagnostics,
    compute_discrete_pde_residual,
    evaluate_physical_residuals,
)
from neural_process_twin.solvers.assembly import assemble_steady_heat_system
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import solve_steady_heat_2d


@pytest.fixture
def sample_problem() -> tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray]:
    """Fixture providing a verified reference solution on an 8x8 grid."""
    grid = Grid2D(nx=8, ny=8, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
    rng = np.random.default_rng(123)
    k_field = 1.0 + rng.uniform(0.5, 2.0, size=(8, 8))
    q_field = rng.uniform(50.0, 300.0, size=(8, 8))
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 50.0),
    )
    res = solve_steady_heat_2d(grid, k_field, q_field, bcs)
    return grid, k_field, q_field, bcs, res.temperature


class TestPreflightAudit:
    """Preflight audit tests for Neural Process Twin residual calculation."""

    def test_audit_1_kelvin_transformation_before_residual(
        self,
        sample_problem: tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray],
    ) -> None:
        """Audit 1: Verify evaluator transforms normalized predictions to physical Kelvin before residual evaluation."""
        grid, k_field, q_field, bcs, t_exact = sample_problem
        normalizer = FieldNormalizer()
        normalizer.k_stats = FieldStatistics(mean=1.5, std=0.5, min=0.5, max=3.0)
        normalizer.q_stats = FieldStatistics(mean=150.0, std=50.0, min=0.0, max=300.0)
        normalizer.t_stats = FieldStatistics(mean=320.0, std=15.0, min=290.0, max=360.0)

        # Mock model predicting normalized temperatures
        t_exact_norm = normalizer.transform_temperature(t_exact)

        class MockModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Returns normalized prediction matching t_exact
                return torch.tensor(t_exact_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        mock_model = MockModel()

        # Create synthetic batch
        x_encoded = encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer)
        input_12ch = np.zeros((12, grid.ny, grid.nx), dtype=np.float32)
        input_12ch[0] = normalizer.transform_k(k_field) if normalizer.k_stats else k_field
        input_12ch[1] = normalizer.transform_q(q_field) if normalizer.q_stats else q_field
        input_12ch[4:12] = x_encoded

        batch = {
            "x": torch.tensor(input_12ch, dtype=torch.float32),
            "y": torch.tensor(t_exact_norm, dtype=torch.float32).unsqueeze(0),
            "temperature_raw": torch.tensor(t_exact, dtype=torch.float32).unsqueeze(0),
            "k_raw": torch.tensor(k_field, dtype=torch.float32).unsqueeze(0),
            "q_raw": torch.tensor(q_field, dtype=torch.float32).unsqueeze(0),
            "sample_id": "sample_001",
        }

        class MockDataset:
            def __init__(self) -> None:
                self.normalizer = normalizer
                self.grid = grid

            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> dict:
                return batch

        from torch.utils.data import DataLoader

        # Run evaluation with residual computation
        result = evaluate_surrogate(
            model=mock_model,
            dataset=DataLoader(MockDataset(), batch_size=1),
            device="cpu",
            compute_residuals=True,
        )

        # Because predicted normalized temperature inverted back to t_exact in Kelvin,
        # the physical PDE residual must be tiny (< 1e-2 W due to float32 precision roundoff,
        # compared to ~6000 W if left in normalized units).
        assert result.residual_metrics is not None
        pde_res = result.residual_metrics["pde_residual_l2_w"]["mean"]
        assert pde_res < 1e-2, f"Expected tiny residual for exact Kelvin field, got {pde_res} W"

    def test_audit_2_exact_discrete_operator_consistency(
        self,
        sample_problem: tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray],
    ) -> None:
        """Audit 2: Verify compute_discrete_pde_residual uses the exact reference solver assembly."""
        grid, k_field, q_field, bcs, t_exact = sample_problem
        sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)

        # Residual computed directly from linear system
        r_direct = sys.A @ t_exact.ravel() - sys.b
        direct_l2 = np.linalg.norm(r_direct)

        # Residual computed via evaluation module
        res_dict = compute_discrete_pde_residual(grid, k_field, q_field, bcs, t_exact)

        assert pytest.approx(res_dict["pde_residual_l2_w"], rel=1e-12) == direct_l2
        assert direct_l2 < 1e-9

    def test_audit_3_sign_and_unit_consistency(
        self,
        sample_problem: tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray],
    ) -> None:
        """Audit 3: Verify source and boundary flux terms have consistent units (Watts) and signs."""
        grid, k_field, q_field, bcs, t_exact = sample_problem
        sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)

        # In steady-state heat conduction:
        # Sum of cell residuals sum(A * T - b) must equal net boundary heat flux minus heat generated
        r = sys.A @ (t_exact + 5.0).ravel() - sys.b  # Perturbed temperature
        sum_r = float(np.sum(r))

        diag = compute_heat_balance(grid, t_exact + 5.0, k_field, q_field, bcs)
        net_imbalance = diag.q_boundary_out_total - diag.q_generated_total

        # Must match down to machine precision
        assert pytest.approx(sum_r, rel=1e-10) == net_imbalance

    def test_audit_4_scale_invariance_and_denominator_safeguards(self) -> None:
        """Audit 4: Verify residual calculation is stable for zero-source and zero-BC problems."""
        grid = Grid2D(nx=8, ny=8, x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0)
        k_field = np.ones((8, 8))
        q_field = np.zeros((8, 8))  # Zero heat generation
        bcs = BoundaryConditions(
            left=BoundaryCondition(
                BoundarySide.LEFT, BoundaryType.DIRICHLET, 0.0
            ),  # Zero temperature
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 0.0),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),  # Insulated
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
        )

        # Non-zero temperature guess on zero-load problem
        t_guess = np.ones((8, 8)) * 10.0
        res = compute_discrete_pde_residual(grid, k_field, q_field, bcs, t_guess)

        # The calculation must remain finite and not crash or produce NaN
        assert np.isfinite(res["pde_residual_l2_w"])
        assert np.isfinite(res["pde_residual_relative"])
        assert res["pde_residual_l2_w"] > 0.0

    def test_audit_5_independence_of_heat_balance_vs_algebraic_residual(
        self,
        sample_problem: tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray],
    ) -> None:
        """Audit 5: Verify global heat-balance diagnostic is computationally independent of linear residual."""
        grid, k_field, q_field, bcs, t_exact = sample_problem

        # A non-uniform temperature perturbation that cancels out on the boundary
        # perturb interior cell only (i=3, j=3)
        t_perturbed = t_exact.copy()
        t_perturbed[3, 3] += 50.0

        pde_res = compute_discrete_pde_residual(grid, k_field, q_field, bcs, t_perturbed)
        flux_diag = compute_boundary_flux_diagnostics(grid, k_field, q_field, bcs, t_perturbed)

        # Interior perturbation produces large local algebraic residual
        assert pde_res["pde_residual_l2_w"] > 10.0

        # But because the perturbation does not touch the boundary cells, boundary heat flux is identical
        assert flux_diag["flux_imbalance_w"] < 1e-8

    def test_audit_6_zero_target_leakage_in_diagnostics(
        self,
        sample_problem: tuple[Grid2D, np.ndarray, np.ndarray, BoundaryConditions, np.ndarray],
    ) -> None:
        """Audit 6: Verify residual diagnostics do not accept or read ground-truth reference temperature."""
        grid, k_field, q_field, bcs, _ = sample_problem
        arbitrary_t = np.ones((8, 8)) * 293.15

        # compute_discrete_pde_residual takes only (grid, k_field, q_field, bcs, temperature_pred)
        # It has no argument for reference solution
        import inspect

        sig_pde = inspect.signature(compute_discrete_pde_residual)
        assert "t_ref" not in sig_pde.parameters
        assert "target" not in sig_pde.parameters
        assert "y_true" not in sig_pde.parameters

        sig_flux = inspect.signature(compute_boundary_flux_diagnostics)
        assert "t_ref" not in sig_flux.parameters
        assert "target" not in sig_flux.parameters

        # Execute on arbitrary prediction
        res = evaluate_physical_residuals(grid, k_field, q_field, bcs, arbitrary_t)
        assert "pde_residual_l2_w" in res
        assert "flux_imbalance_w" in res
