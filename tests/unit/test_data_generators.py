"""Unit tests for physical field generators (GRF, composite materials, heat sources, boundary conditions)."""

from __future__ import annotations

import numpy as np
import pytest

from neural_process_twin.data.generators import (
    BoundaryConditionSampler,
    CompositeMaterialGenerator,
    CovarianceKernel,
    GaussianRandomFieldGenerator,
    HeatSourceGenerator,
    InclusionGeometry,
    PositivityTransform,
)
from neural_process_twin.solvers.boundary import BoundaryType
from neural_process_twin.solvers.grid import Grid2D


@pytest.fixture
def test_grid() -> Grid2D:
    return Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=32, ny=32)


@pytest.mark.unit
def test_grf_reproducibility(test_grid: Grid2D) -> None:
    """Test that GaussianRandomFieldGenerator produces identical fields with the same seed."""
    gen = GaussianRandomFieldGenerator(
        grid=test_grid,
        kernel=CovarianceKernel.GAUSSIAN,
        correlation_length=0.2,
    )
    field_a1 = gen.generate(rng=42)
    field_a2 = gen.generate(rng=42)
    field_b = gen.generate(rng=43)

    assert np.array_equal(field_a1, field_a2)
    assert not np.array_equal(field_a1, field_b)


@pytest.mark.unit
def test_grf_matern_kernel(test_grid: Grid2D) -> None:
    """Test GRF generation with Matérn covariance."""
    gen = GaussianRandomFieldGenerator(
        grid=test_grid,
        kernel=CovarianceKernel.MATERN,
        correlation_length=0.15,
        nu=1.5,
    )
    field = gen.generate(rng=100)
    assert field.shape == (32, 32)
    assert np.all(np.isfinite(field))
    assert np.all(field > 0.0)


@pytest.mark.unit
def test_grf_positivity_and_bounds(test_grid: Grid2D) -> None:
    """Test that sigmoid transform strictly respects [k_min, k_max] and log-normal stays positive."""
    # Sigmoid transform
    k_min, k_max = 0.5, 8.0
    gen_sig = GaussianRandomFieldGenerator(
        grid=test_grid,
        positivity_transform=PositivityTransform.SIGMOID,
        k_min=k_min,
        k_max=k_max,
    )
    for seed in range(5):
        field = gen_sig.generate(rng=seed)
        assert np.all(field >= k_min)
        assert np.all(field <= k_max)
        assert np.all(field > 0.0)

    # Log-normal transform
    gen_ln = GaussianRandomFieldGenerator(
        grid=test_grid,
        positivity_transform=PositivityTransform.LOG_NORMAL,
        log_normal_mu=1.0,
        log_normal_sigma=0.5,
    )
    field_ln = gen_ln.generate(rng=99)
    assert np.all(field_ln > 0.0)
    assert np.all(np.isfinite(field_ln))


@pytest.mark.unit
def test_grf_invalid_parameters(test_grid: Grid2D) -> None:
    """Test validation of invalid generator arguments."""
    with pytest.raises(ValueError, match="correlation_length"):
        GaussianRandomFieldGenerator(grid=test_grid, correlation_length=-0.1)

    with pytest.raises(ValueError, match="Matérn"):
        GaussianRandomFieldGenerator(grid=test_grid, nu=0.0)

    with pytest.raises(ValueError, match="k_min"):
        GaussianRandomFieldGenerator(grid=test_grid, k_min=-1.0)

    with pytest.raises(ValueError, match="k_max"):
        GaussianRandomFieldGenerator(grid=test_grid, k_min=5.0, k_max=2.0)


@pytest.mark.unit
def test_composite_material_generator(test_grid: Grid2D) -> None:
    """Test composite material generation with known inclusions."""
    inc1 = InclusionGeometry(
        shape_type="circle",
        conductivity=50.0,
        center_x=0.5,
        center_y=0.5,
        radius=0.2,
    )
    inc2 = InclusionGeometry(
        shape_type="rectangle",
        conductivity=10.0,
        center_x=0.8,
        center_y=0.2,
        width=0.1,
        height=0.1,
    )
    gen = CompositeMaterialGenerator(
        grid=test_grid,
        matrix_conductivity=1.0,
        inclusions=[inc1, inc2],
    )
    field = gen.generate()

    assert field.shape == (32, 32)
    assert np.all(field > 0.0)
    # Center cell should have circular inclusion conductivity
    mid = test_grid.cell_index(16, 16)
    cx, cy = test_grid.cell_coords(mid)
    assert np.isclose(field[cy, cx], 50.0)
    # Corner cell (0, 0) should be matrix material
    assert np.isclose(field[0, 0], 1.0)


@pytest.mark.unit
def test_composite_material_random_synthesis(test_grid: Grid2D) -> None:
    """Test randomized composite inclusion placement."""
    gen = CompositeMaterialGenerator(grid=test_grid, matrix_conductivity=2.0)
    field = gen.generate(rng=123, num_inclusions=3, contrast_range=(15.0, 30.0))
    assert field.shape == (32, 32)
    assert np.all(field >= 2.0)
    assert np.max(field) > 10.0


@pytest.mark.unit
def test_heat_source_generator(test_grid: Grid2D) -> None:
    """Test zero, uniform, and Gaussian blob heat sources."""
    # Zero
    gen_zero = HeatSourceGenerator(grid=test_grid, distribution="zero")
    assert np.all(gen_zero.generate() == 0.0)

    # Uniform
    gen_uni = HeatSourceGenerator(grid=test_grid, distribution="uniform", q_min=50.0, q_max=50.0)
    assert np.allclose(gen_uni.generate(rng=1), 50.0)

    # Gaussian blobs
    gen_blobs = HeatSourceGenerator(
        grid=test_grid,
        distribution="gaussian_blobs",
        q_min=100.0,
        q_max=500.0,
        num_peaks=3,
    )
    q = gen_blobs.generate(rng=42)
    assert q.shape == (32, 32)
    assert np.all(q >= 0.0)
    assert np.max(q) > 100.0


@pytest.mark.unit
def test_boundary_condition_sampler() -> None:
    """Test BC sampler produces well-posed boundary conditions."""
    sampler = BoundaryConditionSampler(
        temperature_range=(300.0, 350.0),
        mode="mixed_dirichlet_neumann",
    )
    bcs = sampler.sample(rng=42)
    assert bcs.left.bc_type == BoundaryType.DIRICHLET
    assert bcs.right.bc_type == BoundaryType.DIRICHLET
    assert bcs.bottom.bc_type == BoundaryType.NEUMANN
    assert bcs.top.bc_type == BoundaryType.NEUMANN

    # Ensure at least one Dirichlet in random mode
    rand_sampler = BoundaryConditionSampler(mode="random")
    for seed in range(20):
        rand_bcs = rand_sampler.sample(rng=seed)
        assert not rand_bcs.is_pure_neumann
