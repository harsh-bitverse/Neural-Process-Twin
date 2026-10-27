"""Unit tests for HDF5 archival storage and dataset inspection."""

from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from neural_process_twin.data.orchestration import (
    DatasetOrchestrator,
    DatasetPartitionPlan,
)
from neural_process_twin.data.storage import (
    DATASET_FORMAT_VERSION,
    HDF5DatasetWriter,
    StorageSchemaError,
    inspect_hdf5_dataset,
)
from neural_process_twin.solvers.grid import Grid2D


@pytest.fixture
def small_dataset(tmp_path: Path) -> Path:
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=8, ny=8)
    plan = DatasetPartitionPlan(
        base_seed=10,
        num_train=3,
        num_val=1,
        num_test_iid=1,
        num_test_ood=1,
    )
    orchestrator = DatasetOrchestrator(grid=grid, fail_fast=True)
    dataset, audit = orchestrator.generate_dataset(plan)

    file_path = tmp_path / "test_archive.h5"
    writer = HDF5DatasetWriter()
    writer.write(file_path, grid=grid, dataset=dataset, audit=audit)
    return file_path


@pytest.mark.unit
def test_hdf5_write_and_inspect(small_dataset: Path) -> None:
    """Test HDF5 write integrity and inspect summary content."""
    summary = inspect_hdf5_dataset(small_dataset)

    assert summary.format_version == DATASET_FORMAT_VERSION
    assert summary.grid_shape == (8, 8)
    assert summary.domain_bounds == (0.0, 1.0, 0.0, 1.0)
    assert summary.total_samples == 6
    assert set(summary.splits.keys()) == {"train", "validation", "test_iid", "test_ood"}
    assert summary.splits["train"].sample_count == 3
    assert summary.normalizer_method == "standard"
    assert summary.failed_samples_count == 0


@pytest.mark.unit
def test_hdf5_internal_structure(small_dataset: Path) -> None:
    """Verify internal groups, datasets, datatypes, and chunking."""
    with h5py.File(small_dataset, "r") as f:
        assert f.attrs["format_version"] == DATASET_FORMAT_VERSION
        assert "grid" in f
        assert f["grid/x_mesh"].shape == (8, 8)
        assert f["grid/y_mesh"].shape == (8, 8)

        # Train partition
        train_grp = f["train"]
        assert train_grp["k"].shape == (3, 8, 8)
        assert train_grp["q"].shape == (3, 8, 8)
        assert train_grp["temperature"].shape == (3, 8, 8)
        assert train_grp["k"].chunks == (1, 8, 8)
        assert train_grp["sample_id"].shape == (3,)

        # Boundary conditions
        bc_grp = train_grp["boundary_conditions"]
        assert "left_type" in bc_grp
        assert "left_value" in bc_grp
        assert bc_grp["left_value"].shape == (3, 8)

        # Diagnostics
        diag_grp = train_grp["diagnostics"]
        assert "relative_residual" in diag_grp
        assert "heat_imbalance" in diag_grp

        # Normalization
        assert "normalization" in f
        assert "k_stats" in f["normalization"]


@pytest.mark.unit
def test_inspect_missing_file() -> None:
    """Test inspect raises FileNotFoundError on non-existent file."""
    with pytest.raises(FileNotFoundError):
        inspect_hdf5_dataset("non_existent_file.h5")


@pytest.mark.unit
def test_inspect_corrupt_schema(tmp_path: Path) -> None:
    """Test inspect raises StorageSchemaError on corrupt HDF5 file lacking format_version."""
    corrupt_file = tmp_path / "corrupt.h5"
    with h5py.File(corrupt_file, "w") as f:
        f.create_dataset("dummy", data=[1, 2, 3])

    with pytest.raises(StorageSchemaError, match="format_version"):
        inspect_hdf5_dataset(corrupt_file)
