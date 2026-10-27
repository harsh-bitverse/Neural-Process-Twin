"""Unit tests for PyTorch HeatFieldDataset and DataLoader integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neural_process_twin.data.dataset import (
    HeatFieldDataset,
    create_dataloader,
)
from neural_process_twin.data.orchestration import (
    DatasetOrchestrator,
    DatasetPartitionPlan,
)
from neural_process_twin.data.splits import SplitName
from neural_process_twin.data.storage import HDF5DatasetWriter
from neural_process_twin.solvers.grid import Grid2D


@pytest.fixture
def h5_file(tmp_path: Path) -> Path:
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=16, ny=16)
    plan = DatasetPartitionPlan(
        base_seed=123,
        num_train=6,
        num_val=2,
        num_test_iid=2,
        num_test_ood=2,
    )
    orchestrator = DatasetOrchestrator(grid=grid, fail_fast=True)
    dataset, audit = orchestrator.generate_dataset(plan)

    file_path = tmp_path / "dataset.h5"
    HDF5DatasetWriter().write(file_path, grid=grid, dataset=dataset, audit=audit)
    return file_path


@pytest.mark.unit
def test_dataset_tensor_shapes_with_coords(h5_file: Path) -> None:
    """Test HeatFieldDataset produces correct 4-channel input tensors with spatial coords."""
    ds = HeatFieldDataset(h5_file, split=SplitName.TRAIN, include_coords=True)
    assert len(ds) == 6

    sample = ds[0]
    assert "x" in sample and "y" in sample
    # x has 4 channels: (k, q, x, y)
    assert sample["x"].shape == (4, 16, 16)
    assert sample["x"].dtype == torch.float32
    # y has 1 channel: (temperature)
    assert sample["y"].shape == (1, 16, 16)
    assert sample["y"].dtype == torch.float32

    # Check raw fields
    assert sample["k_raw"].shape == (1, 16, 16)
    assert sample["temperature_raw"].shape == (1, 16, 16)


@pytest.mark.unit
def test_dataset_tensor_shapes_without_coords(h5_file: Path) -> None:
    """Test HeatFieldDataset produces 2-channel input tensors when include_coords=False."""
    ds = HeatFieldDataset(h5_file, split=SplitName.TRAIN, include_coords=False)
    sample = ds[0]
    # x has 2 channels: (k, q)
    assert sample["x"].shape == (2, 16, 16)
    assert sample["y"].shape == (1, 16, 16)


@pytest.mark.unit
def test_dataset_index_bounds(h5_file: Path) -> None:
    """Test dataset bounds checking."""
    ds = HeatFieldDataset(h5_file, split=SplitName.TRAIN)
    with pytest.raises(IndexError):
        _ = ds[-1]
    with pytest.raises(IndexError):
        _ = ds[len(ds)]


@pytest.mark.unit
def test_dataloader_batching_and_multiworker(h5_file: Path) -> None:
    """Test DataLoader batching with multi-worker execution."""
    ds = HeatFieldDataset(h5_file, split=SplitName.TRAIN)
    loader = create_dataloader(ds, batch_size=3, shuffle=False, num_workers=2)

    batches = list(loader)
    assert len(batches) == 2
    for b in batches:
        assert b["x"].shape == (3, 4, 16, 16)
        assert b["y"].shape == (3, 1, 16, 16)
