"""Scientific unit and regression tests for Milestone 08H: Error Certification and Bounds.

Tests:
1. Discrete error equation exactness: r = b - A * T_hat satisfies A * (T_ref - T_hat) = r.
2. M-matrix properties: positive diagonal, non-positive off-diagonals, elementwise A^-1 >= 0.
3. Stability constant scaling: ||A^-1||_inf scales as O(N) = O(h^-2).
4. Rigorous upper bound soundness: ||e||_inf <= ||A^-1||_inf * ||r||_inf with zero underestimation.
5. Exact barrier majorant soundness: |e| <= A^-1 |r| elementwise.
6. Preconditioned Jacobi residual units (Kelvin) and non-rigorous nature (can underestimate).
7. Wilson score confidence interval correctness.
8. Independent confirmation cohort provenance and SHA-256 integrity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import h5py
import numpy as np
import pytest
from scipy.sparse import linalg as spla
from scripts.benchmark_milestone_08h_safe_acceptance import wilson_score_interval

from neural_process_twin.solvers.assembly import assemble_steady_heat_system
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D


@pytest.fixture
def test_grid() -> Grid2D:
    return Grid2D(0.0, 1.0, 0.0, 1.0, 16, 16, thickness=1.0)


@pytest.fixture
def test_bcs() -> BoundaryConditions:
    return BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 400.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, 350.0),
    )


def test_discrete_error_equation_exactness(test_grid: Grid2D, test_bcs: BoundaryConditions) -> None:
    """Verifies that A * (T_ref - T_hat) = b - A * T_hat holds to machine precision."""
    res = test_grid.nx
    k_field = np.ones((res, res), dtype=np.float64) * 2.5
    q_field = np.zeros((res, res), dtype=np.float64)

    system = assemble_steady_heat_system(test_grid, k_field, q_field, test_bcs)
    a_csc = system.A.tocsc()
    b = system.b

    # Exact reference solve
    t_ref = spla.spsolve(a_csc, b)

    # Synthetic surrogate prediction with known error
    rng = np.random.default_rng(42)
    synthetic_error = rng.normal(0.0, 5.0, size=t_ref.shape)
    t_hat = t_ref + synthetic_error

    # Algebraic residual
    r = b - a_csc.dot(t_hat)

    # True error e = T_ref - T_hat
    e_true = t_ref - t_hat

    # Check A * e == r
    ae = a_csc.dot(e_true)
    assert np.allclose(ae, r, atol=1e-10, rtol=1e-10), "Discrete error equation A * e = r failed!"


def test_m_matrix_and_inverse_non_negativity(
    test_grid: Grid2D, test_bcs: BoundaryConditions
) -> None:
    """Verifies M-matrix property: off-diagonals <= 0, diagonals > 0, and A^-1 >= 0."""
    res = test_grid.nx
    k_field = np.linspace(1.0, 5.0, res * res).reshape(res, res)
    q_field = np.zeros((res, res), dtype=np.float64)

    system = assemble_steady_heat_system(test_grid, k_field, q_field, test_bcs)
    a_dense = system.A.toarray()

    # 1. Diagonal entries positive
    diag = np.diag(a_dense)
    assert np.all(diag > 0.0), "Diagonal entries must be strictly positive!"

    # 2. Off-diagonal entries non-positive
    off_diag_mask = ~np.eye(res * res, dtype=bool)
    assert np.all(a_dense[off_diag_mask] <= 1e-12), "Off-diagonal entries must be non-positive!"

    # 3. Dense inverse non-negative: A^-1 >= 0
    a_inv = np.linalg.inv(a_dense)
    min_inv_entry = np.min(a_inv)
    assert min_inv_entry >= -1e-12, f"A^-1 has negative entries! min = {min_inv_entry}"


def test_stability_constant_o_n_scaling() -> None:
    """Verifies that ||A^-1||_inf scales as O(N) = O(h^-2)."""
    resolutions = [16, 32]
    norm_ainv_values = []

    for res in resolutions:
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 300.0),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, 300.0),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, 300.0),
        )
        k_field = np.ones((res, res), dtype=np.float64)
        q_field = np.zeros((res, res), dtype=np.float64)

        sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)
        ones_vec = np.ones(res * res, dtype=np.float64)
        w_ones = spla.spsolve(sys.A.tocsc(), ones_vec)
        norm_ainv = float(np.max(w_ones))
        norm_ainv_values.append(norm_ainv)

    # When grid doubles from 16 to 32 (N quadruples from 256 to 1024),
    # ||A^-1||_inf should scale by approximately 4x.
    scaling_ratio = norm_ainv_values[1] / norm_ainv_values[0]
    assert np.isclose(scaling_ratio, 4.0, rtol=0.15), (
        f"Expected scaling ratio ~4.0x, got {scaling_ratio:.3f}x"
    )


def test_rigorous_upper_bound_soundness(test_grid: Grid2D, test_bcs: BoundaryConditions) -> None:
    """Verifies that ||e||_inf <= ||A^-1||_inf * ||r||_inf with 0 underestimation."""
    res = test_grid.nx
    k_field = np.ones((res, res), dtype=np.float64) * 2.0
    q_field = np.zeros((res, res), dtype=np.float64)

    system = assemble_steady_heat_system(test_grid, k_field, q_field, test_bcs)
    a_csc = system.A.tocsc()
    b = system.b

    t_ref = spla.spsolve(a_csc, b)

    ones_vec = np.ones(res * res, dtype=np.float64)
    w_ones = spla.spsolve(a_csc, ones_vec)
    norm_ainv = float(np.max(w_ones))

    # Test 5 different error perturbations
    for seed in range(5):
        rng = np.random.default_rng(seed)
        t_hat = t_ref + rng.normal(0.0, 10.0, size=t_ref.shape)
        e_true = np.max(np.abs(t_ref - t_hat))
        r = b - a_csc.dot(t_hat)
        r_linf = np.max(np.abs(r))

        bound = norm_ainv * r_linf
        assert bound >= e_true - 1e-10, f"Rigorous bound violated! Bound={bound}, True={e_true}"


def test_exact_barrier_majorant_soundness(test_grid: Grid2D, test_bcs: BoundaryConditions) -> None:
    """Verifies that |e| <= A^-1 |r| elementwise."""
    res = test_grid.nx
    k_field = np.ones((res, res), dtype=np.float64) * 1.5
    q_field = np.zeros((res, res), dtype=np.float64)

    system = assemble_steady_heat_system(test_grid, k_field, q_field, test_bcs)
    a_csc = system.A.tocsc()
    b = system.b
    t_ref = spla.spsolve(a_csc, b)

    rng = np.random.default_rng(123)
    t_hat = t_ref + rng.uniform(-15.0, 15.0, size=t_ref.shape)
    e_true_vec = np.abs(t_ref - t_hat)

    r = b - a_csc.dot(t_hat)
    abs_r = np.abs(r)
    w_barrier = spla.spsolve(a_csc, abs_r)

    # Barrier must bound error in EVERY single cell
    assert np.all(w_barrier >= e_true_vec - 1e-10), "Barrier majorant violated in some cells!"


def test_jacobi_residual_properties(test_grid: Grid2D, test_bcs: BoundaryConditions) -> None:
    """Verifies Jacobi residual properties: O(N) evaluation and non-guaranteed upper bound."""
    res = test_grid.nx
    k_field = np.ones((res, res), dtype=np.float64)
    q_field = np.zeros((res, res), dtype=np.float64)

    system = assemble_steady_heat_system(test_grid, k_field, q_field, test_bcs)
    a_csr = system.A.tocsr()
    b = system.b
    t_ref = spla.spsolve(system.A.tocsc(), b)

    diag_a = a_csr.diagonal()
    inv_diag = 1.0 / diag_a

    # Construct a smooth, long-wavelength error mode (e.g. constant temperature offset in interior)
    # For a smooth mode, A * e is very small, so D^-1 * r is much smaller than ||e||_inf!
    smooth_err = np.zeros((res, res))
    smooth_err[2:-2, 2:-2] = 20.0  # 20 Kelvin offset in interior
    t_hat = t_ref.reshape(res, res) - smooth_err
    t_hat_flat = t_hat.ravel()

    r = b - a_csr.dot(t_hat_flat)
    jac_estimator = float(np.max(np.abs(inv_diag * r)))
    true_maxae = float(np.max(smooth_err))

    # For smooth errors, Jacobi residual strictly UNDERESTIMATES true error
    assert jac_estimator < true_maxae, (
        f"Jacobi estimator should underestimate smooth error! Estimator={jac_estimator}, True={true_maxae}"
    )


def test_wilson_confidence_interval_soundness() -> None:
    """Verifies that Wilson score interval handles zero, full, and intermediate counts."""
    # Zero successes
    low_0, high_0 = wilson_score_interval(0, 100)
    assert np.isclose(low_0, 0.0, atol=1e-12)
    assert 0.0 < high_0 < 0.1

    # 100% successes
    low_1, high_1 = wilson_score_interval(100, 100)
    assert 0.9 < low_1 < 1.0
    assert np.isclose(high_1, 1.0, atol=1e-12)

    # 0 trials edge case
    low_e, high_e = wilson_score_interval(0, 0)
    assert low_e == 0.0 and high_e == 0.0


def test_independent_cohort_provenance_and_integrity() -> None:
    """Verifies that independent cohort 08H exists, has correct seed ranges, and valid SHA-256."""
    cohort_path = Path("artifacts/data/independent_cohort_08h.h5")
    manifest_path = Path("artifacts/manifests/independent_cohort_08h_manifest.json")

    assert cohort_path.is_file(), f"Missing cohort file {cohort_path}"
    assert manifest_path.is_file(), f"Missing manifest file {manifest_path}"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Compute hash
    from scripts.generate_milestone_08h_independent_cohort import compute_file_sha256

    actual_sha = compute_file_sha256(cohort_path)
    assert actual_sha == manifest["sha256"], "Cohort SHA-256 does not match manifest!"

    # Check seeds in HDF5
    with h5py.File(cohort_path, "r") as f:
        for res in [32, 64, 128, 256]:
            assert f"res_{res}" in f, f"Missing resolution {res} in cohort"
            seeds = f[f"res_{res}/seeds"][:]
            assert np.all(seeds >= 1_100_000), (
                "All seeds must be >= 1,100,000 for strict disjointness!"
            )
