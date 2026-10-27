"""Preflight verification tests for boundary condition encoding into model input tensors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neural_process_twin.data.boundary_encoding import (
    CANONICAL_CHANNEL_NAMES,
    NUM_CANONICAL_CHANNELS,
    decode_boundary_conditions,
    encode_boundary_conditions,
)
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D


@pytest.fixture
def test_grid() -> Grid2D:
    return Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=16, ny=16)


@pytest.fixture
def mock_normalizer() -> FieldNormalizer:
    norm = FieldNormalizer()
    k_dummy = np.full((16, 16), 1.0)
    q_dummy = np.full((16, 16), 100.0)
    t_dummy = np.linspace(300.0, 360.0, 16 * 16).reshape(16, 16)
    return norm.fit(k_dummy, q_dummy, t_dummy)


@pytest.mark.unit
def test_canonical_channel_count_and_names() -> None:
    """Verify canonical channel specifications."""
    assert NUM_CANONICAL_CHANNELS == 12
    assert len(CANONICAL_CHANNEL_NAMES) == 12
    assert CANONICAL_CHANNEL_NAMES[0] == "conductivity"
    assert CANONICAL_CHANNEL_NAMES[1] == "heat_source"
    assert CANONICAL_CHANNEL_NAMES[2] == "coord_x"
    assert CANONICAL_CHANNEL_NAMES[3] == "coord_y"
    assert "bc_dirichlet_mask_x" in CANONICAL_CHANNEL_NAMES
    assert "bc_neumann_val_y" in CANONICAL_CHANNEL_NAMES


@pytest.mark.unit
def test_distinct_bcs_produce_distinguishable_inputs(
    test_grid: Grid2D, mock_normalizer: FieldNormalizer
) -> None:
    """Requirement 1: Distinct BC configurations must produce distinguishable model inputs."""
    # Config A: Left 300K, Right 350K, Bottom adiabatic (g=0), Top adiabatic (g=0)
    bcs_a = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )

    # Config B: Same geometry, but Left 320K, Right 380K, Bottom adiabatic, Top adiabatic
    bcs_b = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 320.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 380.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )

    # Config C: Pure Dirichlet on all four boundaries
    bcs_c = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.DIRICHLET, 310.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, 340.0),
    )

    tensor_a = encode_boundary_conditions(bcs_a, test_grid, mock_normalizer)
    tensor_b = encode_boundary_conditions(bcs_b, test_grid, mock_normalizer)
    tensor_c = encode_boundary_conditions(bcs_c, test_grid, mock_normalizer)

    # Tensors must be different
    assert not np.array_equal(tensor_a, tensor_b)
    assert not np.array_equal(tensor_a, tensor_c)
    assert not np.array_equal(tensor_b, tensor_c)

    # Specific difference: tensor_a and tensor_b differ in Dirichlet values (channels 1 & 5)
    assert not np.allclose(tensor_a[1], tensor_b[1])  # bc_dirichlet_val_x
    # tensor_a has Neumann on bottom/top (channels 6 & 7 active), tensor_c has Dirichlet (channels 4 & 5 active)
    assert np.all(tensor_a[6, 0, :] == 1.0)  # Neumann mask active on bottom in A
    assert np.all(tensor_c[4, 0, :] == 1.0)  # Dirichlet mask active on bottom in C


@pytest.mark.unit
def test_repeated_reads_are_identical(test_grid: Grid2D, mock_normalizer: FieldNormalizer) -> None:
    """Requirement 2: The same physical problem generates identical input tensors across repeated calls."""
    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 305.5),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 345.5),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, -250.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 150.0),
    )

    read1 = encode_boundary_conditions(bcs, test_grid, mock_normalizer)
    read2 = encode_boundary_conditions(bcs, test_grid, mock_normalizer)
    read3 = encode_boundary_conditions(bcs, test_grid, mock_normalizer)

    assert np.array_equal(read1, read2)
    assert np.array_equal(read2, read3)


@pytest.mark.unit
def test_reconstruction_from_saved_metadata(
    test_grid: Grid2D, mock_normalizer: FieldNormalizer
) -> None:
    """Requirement 3: Model input representation can be reconstructed into physical metadata."""
    orig_bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 305.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 355.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, -120.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 350.0),
    )

    # Encode with normalizer
    encoded = encode_boundary_conditions(orig_bcs, test_grid, mock_normalizer, flux_scale=1000.0)

    # Decode back to physical units
    decoded_bcs = decode_boundary_conditions(encoded, test_grid, mock_normalizer, flux_scale=1000.0)

    # Verify sides and types
    assert decoded_bcs.left.bc_type == BoundaryType.DIRICHLET
    assert decoded_bcs.right.bc_type == BoundaryType.DIRICHLET
    assert decoded_bcs.bottom.bc_type == BoundaryType.NEUMANN
    assert decoded_bcs.top.bc_type == BoundaryType.NEUMANN

    # Verify physical values in Kelvin and W/m^2
    eval_orig = orig_bcs.evaluate_all(test_grid)
    eval_dec = decoded_bcs.evaluate_all(test_grid)

    assert np.allclose(eval_orig["left"], eval_dec["left"], atol=1e-4)
    assert np.allclose(eval_orig["right"], eval_dec["right"], atol=1e-4)
    assert np.allclose(eval_orig["bottom"], eval_dec["bottom"], atol=1e-4)
    assert np.allclose(eval_orig["top"], eval_dec["top"], atol=1e-4)


@pytest.mark.unit
def test_sign_and_unit_conventions(test_grid: Grid2D) -> None:
    """Requirement 4: Encoding strictly follows solver outward flux and Kelvin unit conventions."""
    # Outward positive heat flux (+500 W/m^2 leaving) and inward negative flux (-300 W/m^2 entering)
    pos_flux = 500.0
    neg_flux = -300.0
    adiabatic = 0.0

    bcs = BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.NEUMANN, neg_flux),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.NEUMANN, pos_flux),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, adiabatic),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.DIRICHLET, 310.0),
    )

    flux_scale = 1000.0
    encoded = encode_boundary_conditions(bcs, test_grid, normalizer=None, flux_scale=flux_scale)

    # Left Neumann outward flux: neg_flux / 1000 = -0.3
    assert np.allclose(encoded[3, :, 0], -0.3)
    # Right Neumann outward flux: pos_flux / 1000 = +0.5
    assert np.allclose(encoded[3, :, -1], +0.5)
    # Bottom adiabatic wall: flux = 0.0, but mask is 1.0 (unambiguous!)
    assert np.all(encoded[6, 0, :] == 1.0)
    assert np.all(encoded[7, 0, :] == 0.0)
    # Interior must be zero
    assert np.all(encoded[2, :, 1:-1] == 0.0)  # interior neumann mask is 0
    assert np.all(encoded[3, :, 1:-1] == 0.0)  # interior neumann val is 0


@pytest.mark.unit
def test_dataset_12_channel_integration() -> None:
    """Test HeatFieldDataset produces 12 channels when include_bcs=True."""
    dataset_path = Path("artifacts/data/smoke_dataset.h5")
    if not dataset_path.is_file():
        pytest.skip("smoke_dataset.h5 not generated yet")

    ds = HeatFieldDataset(dataset_path, split="train", include_coords=True, include_bcs=True)
    assert len(ds) > 0
    sample = ds[0]

    assert sample["x"].shape == (12, 16, 16)
    assert sample["y"].shape == (1, 16, 16)
    assert sample["x"].dtype == torch.float32
    assert "bc_raw" in sample

    # Verify that different samples have distinguishable inputs if their BCs or fields differ
    sample_1 = ds[1]
    assert not torch.equal(sample["x"], sample_1["x"])
