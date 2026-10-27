"""Integration tests for end-to-end dataset generation, archiving, and PyTorch ingestion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neural_process_twin.data import (
    DatasetGenerationError,
    DatasetOrchestrator,
    DatasetPartitionPlan,
    HDF5DatasetWriter,
    HeatFieldDataset,
    SplitName,
    create_dataloader,
    inspect_hdf5_dataset,
    verify_ood_criteria,
)
from neural_process_twin.solvers.grid import Grid2D


@pytest.mark.integration
def test_end_to_end_data_synthesis_and_ingestion(tmp_path: Path) -> None:
    """Test full integration pipeline: solve -> verify -> archive -> inspect -> PyTorch load."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=16, ny=16)
    plan = DatasetPartitionPlan(
        base_seed=42,
        num_train=8,
        num_val=2,
        num_test_iid=2,
        num_test_ood=2,
    )
    orchestrator = DatasetOrchestrator(grid=grid, fail_fast=True)
    dataset, audit = orchestrator.generate_dataset(plan)

    assert audit.successful_samples == 14
    assert audit.failed_samples == 0

    # Verify physical OOD contrast separation
    k_train = [s.conductivity for s in dataset[SplitName.TRAIN]]
    k_ood = [s.conductivity for s in dataset[SplitName.TEST_OOD]]
    ood_metrics = verify_ood_criteria(k_train, k_ood)
    assert ood_metrics["mean_ood_contrast"] > ood_metrics["mean_iid_contrast"]

    # Archive to HDF5
    h5_path = tmp_path / "integrated_dataset.h5"
    writer = HDF5DatasetWriter()
    writer.write(h5_path, grid=grid, dataset=dataset, audit=audit)

    # Inspect archive
    summary = inspect_hdf5_dataset(h5_path)
    assert summary.total_samples == 14
    assert summary.grid_shape == (16, 16)
    assert summary.splits["train"].sample_count == 8

    # Ingest into PyTorch HeatFieldDataset
    train_ds = HeatFieldDataset(h5_path, split=SplitName.TRAIN, include_coords=True)
    assert len(train_ds) == 8
    assert train_ds.normalizer is not None
    assert train_ds.normalizer.is_fitted

    ood_ds = HeatFieldDataset(h5_path, split=SplitName.TEST_OOD, normalizer=train_ds.normalizer)
    assert len(ood_ds) == 2

    # DataLoader iteration
    train_loader = create_dataloader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(train_loader))
    assert batch["x"].shape == (4, 4, 16, 16)
    assert batch["y"].shape == (4, 1, 16, 16)

    # Test inverse transformation on predicted/target field
    y_norm = batch["y"].squeeze(1).numpy()  # shape (4, 16, 16)
    t_recon = train_ds.normalizer.inverse_transform_temperature(y_norm)
    t_raw = batch["temperature_raw"].squeeze(1).numpy()
    assert np.allclose(t_recon, t_raw, atol=1e-4)


@pytest.mark.integration
def test_orchestrator_fail_fast_policy() -> None:
    """Test that fail-fast policy immediately aborts with DatasetGenerationError when tolerance is violated."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=16, ny=16)
    # Set an unachievable residual tolerance
    orchestrator = DatasetOrchestrator(
        grid=grid,
        fail_fast=True,
        residual_tolerance=1e-30,
    )
    with pytest.raises(DatasetGenerationError, match="Algebraic relative residual"):
        orchestrator.generate_sample(
            sample_id="test_fail_001",
            seed=42,
            split=SplitName.TRAIN,
        )


@pytest.mark.integration
def test_orchestrator_skip_and_audit_policy() -> None:
    """Test that skip-and-audit policy logs failures and retries until success."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=16, ny=16)
    # Using an overly strict tolerance where some samples might fail or we test retry mechanism
    orchestrator = DatasetOrchestrator(
        grid=grid,
        fail_fast=False,
        residual_tolerance=1e-6,
        heat_balance_tolerance=1e-3,
    )
    plan = DatasetPartitionPlan(
        base_seed=999,
        num_train=2,
        num_val=0,
        num_test_iid=0,
        num_test_ood=0,
    )
    dataset, audit = orchestrator.generate_dataset(plan)
    assert len(dataset[SplitName.TRAIN]) == 2
    assert audit.successful_samples == 2
