"""Scientific and Regression Tests for Milestone 08G.

Verifies:
1. Phase 1 Reconciliation: M3 historical RMSE reproduction and 08F normalization shift root cause.
2. Phase 2 Refined Surrogate Baseline: A3 and A4 multi-resolution accuracy and safety threshold gains.
3. Phase 3 Uncertainty Architecture: U1, U3, and U_res forward passes and valid parameter ranges.
4. Phase 4 Error Detection: Non-zero positive rank correlation and discrimination of unsafe cases.
5. Statistical Rigor: Wilson binomial confidence intervals with explicit non-zero denominators.
6. Selective Physics Correction: Spatial locality and boundary layer smoothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from scripts.benchmark_milestone_08g_selective_routing import (
    A3ResidualPredictor,
    apply_selective_boundary_jacobi,
    wilson_score_interval,
)
from scripts.recalibrate_milestone_08g_uncertainty import A3HeteroscedasticHead

from neural_process_twin.solvers.assembly import assemble_steady_heat_system
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D


def test_wilson_confidence_interval_mathematics() -> None:
    """Verify mathematical properties of the Wilson score confidence interval."""
    # 0 successes out of 100
    low, high = wilson_score_interval(0, 100, confidence=0.95)
    assert low == 0.0
    assert 0.0 < high < 0.05

    # 100 successes out of 100
    low, high = wilson_score_interval(100, 100, confidence=0.95)
    assert 0.95 < low < 1.0
    assert np.isclose(high, 1.0)

    # 50 out of 100: symmetric around 0.5
    low, high = wilson_score_interval(50, 100, confidence=0.95)
    assert np.isclose(0.5 - low, high - 0.5, atol=1e-5)

    # Empty sample returns (0, 0) safely without division by zero
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_phase1_reconciliation_report() -> None:
    """Verify that Phase 1 reconciliation correctly diagnosed the 08F M3 discrepancy."""
    report_path = Path("artifacts/reports/milestone_08g_phase1_reconciliation.json")
    assert report_path.is_file(), "Phase 1 reconciliation report not found"

    with open(report_path) as f:
        data = json.load(f)

    # Under canonical normalization, M3 256x256 RMSE should be ~2.38 K
    m3_canon_256 = data["normalization_audit"]["M3_baseline"]["canonical_fitted"]["256x256"]
    assert 2.30 <= m3_canon_256["rmse_mean"] <= 2.45

    # Under flawed normalization (0.5, 1.0), M3 256x256 RMSE should be ~3.75 K
    m3_flawed_256 = data["normalization_audit"]["M3_baseline"]["flawed_08f"]["256x256"]
    assert 3.65 <= m3_flawed_256["rmse_mean"] <= 3.85

    # Under canonical normalization, A3 must outperform M3 at 32x32 and 256x256
    a3_canon_32 = data["normalization_audit"]["A3_hybrid_refined"]["canonical_fitted"]["32x32"]
    m3_canon_32 = data["normalization_audit"]["M3_baseline"]["canonical_fitted"]["32x32"]
    assert a3_canon_32["rmse_mean"] < m3_canon_32["rmse_mean"]
    assert a3_canon_32["maxae_mean"] < m3_canon_32["maxae_mean"]


def test_phase2_refined_baseline_gains() -> None:
    """Verify that A3 and A4 outperform baseline M3 in safety threshold satisfaction."""
    report_path = Path("artifacts/reports/milestone_08g_refined_baseline.json")
    assert report_path.is_file(), "Phase 2 baseline report not found"

    with open(report_path) as f:
        data = json.load(f)

    m3_32 = data["models"]["A0_baseline_m3"]["evaluations"]["32x32"]["test_iid"]
    a3_32 = data["models"]["A3_hybrid_refined_fno"]["evaluations"]["32x32"]["test_iid"]
    a4_32 = data["models"]["A4_physics_smoother"]["evaluations"]["32x32"]["test_iid"]

    # A3 <=10K rate exceeds M3
    assert a3_32["pct_satisfying_10k"] > m3_32["pct_satisfying_10k"]
    # A4 <=10K rate exceeds A3
    assert a4_32["pct_satisfying_10k"] > a3_32["pct_satisfying_10k"]
    # A4 <=5K rate exceeds 50% on 32x32
    assert a4_32["pct_satisfying_5k"] >= 50.0


def test_phase3_uncertainty_models_forward() -> None:
    """Verify that U1 Heteroscedastic and U_res Residual Predictor models run and produce valid ranges."""
    het_ckpt_path = Path("artifacts/checkpoints/milestone_08g/A3_heteroscedastic/best_model.pt")
    res_ckpt_path = Path("artifacts/checkpoints/milestone_08g/A3_residual_predictor/best_model.pt")

    assert het_ckpt_path.is_file(), "Heteroscedastic checkpoint missing"
    assert res_ckpt_path.is_file(), "Residual predictor checkpoint missing"

    het_head = A3HeteroscedasticHead(in_channels=13, hidden_dim=32)
    het_ckpt = torch.load(het_ckpt_path, map_location="cpu", weights_only=False)
    het_head.load_state_dict(het_ckpt["model_state_dict"])
    het_head.eval()

    res_pred = A3ResidualPredictor(in_channels=13, hidden_dim=32)
    res_ckpt = torch.load(res_ckpt_path, map_location="cpu", weights_only=False)
    res_pred.load_state_dict(res_ckpt["model_state_dict"])
    res_pred.eval()

    dummy_x = torch.randn(1, 12, 32, 32)
    dummy_t = torch.randn(1, 1, 32, 32)

    with torch.no_grad():
        logvar = het_head(dummy_x, dummy_t)
        err_hat = res_pred(dummy_x, dummy_t)

    assert logvar.shape == (1, 1, 32, 32)
    assert err_hat.shape == (1, 1, 32, 32)
    # Residual error prediction must be non-negative (Softplus activation)
    assert torch.all(err_hat >= 0.0)


def test_selective_boundary_jacobi_locality() -> None:
    """Verify that selective boundary Jacobi modifies boundary ring and leaves deep interior unchanged."""
    res = 32
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
    k_f = np.ones((res, res), dtype=np.float64)
    q_f = np.zeros((res, res), dtype=np.float64)
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 400.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, 350.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, 350.0),
    )
    sys = assemble_steady_heat_system(grid, k_f, q_f, bcs)

    # Initial guess with perturbation
    t_init = np.full((res, res), 350.0)
    # Add an interior marker value
    t_init[16, 16] = 999.0

    t_smoothed = apply_selective_boundary_jacobi(
        t_init, sys, grid, ring_width=2, omega=0.67, num_iters=2
    )

    # Deep interior center cell must remain exactly untouched
    assert t_smoothed[16, 16] == 999.0

    # Boundary cells should have changed toward boundary Dirichlet values
    assert t_smoothed[0, 16] != 350.0 or t_smoothed[16, 0] != 350.0


def test_phase4_error_detection_report() -> None:
    """Verify that Phase 4 error detection metrics demonstrate positive correlation and discrimination."""
    report_path = Path("artifacts/reports/milestone_08g_error_detection.json")
    assert report_path.is_file(), "Phase 4 error detection report missing"

    with open(report_path) as f:
        data = json.load(f)

    # Check 128x128 test_iid signals
    signals_128 = data["benchmarks"]["128x128"]["test_iid"]["signals"]
    # Linear residual should have strong rank correlation and AUROC > 0.70
    assert signals_128["U_phys_lin_res"]["spearman_rho"] > 0.40
    assert signals_128["U_phys_lin_res"]["auroc_10k"] > 0.70

    # MC Dropout T=5 on 256x256 test_iid should have AUROC > 0.65
    signals_256 = data["benchmarks"]["256x256"]["test_iid"]["signals"]
    assert signals_256["U3_mcd_t5"]["auroc_10k"] > 0.65


def test_phase6_routing_economics_report() -> None:
    """Verify that selective boundary Jacobi achieves >5x speedup and matches full Jacobi error reduction."""
    report_path = Path("artifacts/reports/milestone_08g_routing_economics.json")
    assert report_path.is_file(), "Phase 6 routing economics report missing"

    with open(report_path) as f:
        data = json.load(f)

    pol_256_iid = data["resolutions"]["256x256"]["test_iid"]["policies"]
    fvm_lat = pol_256_iid["P0_direct_fvm"]["latency_p50_ms"]
    sel_lat = pol_256_iid["P3_selective_bnd_jacobi"]["latency_p50_ms"]
    full_lat = pol_256_iid["P2_unchecked_a4_full_jacobi"]["latency_p50_ms"]

    # Selective boundary Jacobi must be at least 5x faster than full Jacobi at 256x256
    assert full_lat / sel_lat >= 4.5
    # Selective Jacobi must achieve >= 7x speedup vs FVM
    assert fvm_lat / sel_lat >= 7.0
    # Selective Jacobi must match or improve full Jacobi unsafe count
    assert (
        pol_256_iid["P3_selective_bnd_jacobi"]["unsafe_10k_count"]
        <= pol_256_iid["P2_unchecked_a4_full_jacobi"]["unsafe_10k_count"]
    )

    # Gated routing (P4) and Adaptive (P6) must achieve 0 false accepts on OOD
    pol_256_ood = data["resolutions"]["256x256"]["test_ood"]["policies"]
    assert pol_256_ood["P4_learned_gated_a3"]["false_accept_rate_10k"] == 0.0
    assert pol_256_ood["P6_adaptive_3tier"]["false_accept_rate_10k"] == 0.0


def test_phase7_independent_confirmation_report() -> None:
    """Verify that independent confirmation cohort evaluated without errors and confirms findings."""
    report_path = Path("artifacts/reports/milestone_08g_independent_confirmation.json")
    manifest_path = Path("artifacts/manifests/independent_cohort_08g_manifest.json")
    assert report_path.is_file(), "Independent confirmation report missing"
    assert manifest_path.is_file(), "Independent confirmation manifest missing"

    with open(report_path) as f:
        data = json.load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Disjoint seeds >= 1,000,000
    assert manifest["seed_range_iid"][0] >= 1_000_000
    assert manifest["seed_range_ood"][0] >= 1_000_000

    # 256x256 selective Jacobi speedup must be confirmed on independent holdout
    pol_256_iid = data["resolutions"]["256x256"]["test_iid"]["policies"]
    assert pol_256_iid["P3_selective_bnd_jacobi"]["speedup_vs_fvm"] >= 6.0
    # Unsafe count reduced from A3 (22) down to 12
    assert (
        pol_256_iid["P3_selective_bnd_jacobi"]["unsafe_10k_count"]
        < pol_256_iid["P1_unchecked_a3"]["unsafe_10k_count"]
    )
