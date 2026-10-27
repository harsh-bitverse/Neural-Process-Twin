"""Generate dedicated independent calibration dataset for Milestone 05B.

Synthesizes 200 independent IID problem samples with conservative finite-volume reference solves,
rigorous residual and heat conservation diagnostics, and writes to artifacts/data/calibration_dataset.h5
with normalization metadata preserved from scaled_dataset.h5.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np

from neural_process_twin.data import (
    DatasetOrchestrator,
    DatasetPartitionPlan,
    HDF5DatasetWriter,
    SplitName,
    inspect_hdf5_dataset,
    verify_partitions_disjoint,
)
from neural_process_twin.data.normalization import FieldNormalizer, FieldStatistics
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.utils.config import load_data_config


def read_normalizer_from_h5(h5_path: Path) -> FieldNormalizer:
    """Extract fitted normalizer from existing scaled dataset archive."""
    with h5py.File(h5_path, "r") as f:
        if "normalization" not in f:
            raise ValueError(f"No normalization group found in {h5_path}")
        norm_grp = f["normalization"]
        method = str(norm_grp.attrs.get("method", "standard"))
        eps = float(norm_grp.attrs.get("eps", 1e-8))
        norm = FieldNormalizer(method=method, eps=eps)

        def read_stats(grp_name: str) -> FieldStatistics | None:
            if grp_name in norm_grp:
                sg = norm_grp[grp_name]
                return FieldStatistics(
                    mean=float(sg.attrs["mean"]),
                    std=float(sg.attrs["std"]),
                    min=float(sg.attrs["min"]),
                    max=float(sg.attrs["max"]),
                )
            return None

        norm.k_stats = read_stats("k_stats")
        norm.q_stats = read_stats("q_stats")
        norm.t_stats = read_stats("t_stats")
        if not norm.is_fitted:
            raise ValueError("Extracted normalizer is not marked as fitted.")
        return norm


def generate_calibration_dataset(
    config_path: Path = Path("configs/data_scaled.yaml"),
    reference_dataset_path: Path = Path("artifacts/data/scaled_dataset.h5"),
    output_path: Path = Path("artifacts/data/calibration_dataset.h5"),
    num_cal: int = 200,
    base_seed: int = 42,
) -> Path:
    """Generate independent calibration dataset with 200 samples."""
    print("=" * 80)
    print("MILESTONE 05B: GENERATING INDEPENDENT CALIBRATION DATASET")
    print("=" * 80)
    cfg = load_data_config(config_path)

    grid = Grid2D(
        x_min=cfg.domain.x_min,
        x_max=cfg.domain.x_max,
        y_min=cfg.domain.y_min,
        y_max=cfg.domain.y_max,
        nx=cfg.grid.nx,
        ny=cfg.grid.ny,
    )

    plan = DatasetPartitionPlan(
        base_seed=base_seed,
        num_train=1,  # dummy placeholder
        num_val=0,
        num_test_iid=0,
        num_test_ood=0,
        num_cal=num_cal,
    )

    # Verify seed disjointness against existing scaled dataset splits
    existing_plan = DatasetPartitionPlan(
        base_seed=base_seed,
        num_train=cfg.dataset.num_train,
        num_val=cfg.dataset.num_val,
        num_test_iid=cfg.dataset.num_test_iid,
        num_test_ood=cfg.dataset.num_test_ood,
    )
    all_seeds = {
        "train": existing_plan.get_split_seeds(SplitName.TRAIN),
        "validation": existing_plan.get_split_seeds(SplitName.VALIDATION),
        "test_iid": existing_plan.get_split_seeds(SplitName.TEST_IID),
        "test_ood": existing_plan.get_split_seeds(SplitName.TEST_OOD),
        "calibration": plan.get_split_seeds(SplitName.CALIBRATION),
    }
    verify_partitions_disjoint(all_seeds)
    print(
        f"Verified all partition seeds disjoint! Calibration seeds: {all_seeds['calibration'][0]} .. {all_seeds['calibration'][-1]}"
    )

    # Load frozen reference normalizer
    normalizer = read_normalizer_from_h5(reference_dataset_path)
    print(f"Loaded frozen normalizer from {reference_dataset_path}:")
    print(f"  k_mean={normalizer.k_stats.mean:.4f}, k_std={normalizer.k_stats.std:.4f}")
    print(f"  q_mean={normalizer.q_stats.mean:.4f}, q_std={normalizer.q_stats.std:.4f}")
    print(f"  t_mean={normalizer.t_stats.mean:.4f}, t_std={normalizer.t_stats.std:.4f}")

    orchestrator = DatasetOrchestrator(
        grid=grid,
        fail_fast=True,
        residual_tolerance=1e-6,
        heat_balance_tolerance=1e-3,
    )

    cal_seeds = plan.get_split_seeds(SplitName.CALIBRATION)
    from neural_process_twin.data.orchestration import GenerationAudit

    audit = GenerationAudit()

    t0 = time.perf_counter()
    print(f"Synthesizing {num_cal} independent calibration samples (IID regime)...")
    cal_samples = orchestrator.generate_split(
        split=SplitName.CALIBRATION,
        seeds=cal_seeds,
        audit=audit,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"Generated {len(cal_samples)} samples in {elapsed:.2f}s ({len(cal_samples) / max(elapsed, 1e-6):.1f} samples/s)."
    )
    print(
        f"Audit: attempted={audit.attempted_samples}, successful={audit.successful_samples}, failed={audit.failed_samples}"
    )

    # Audit physical distributions
    k_vals = np.stack([s.conductivity for s in cal_samples])
    q_vals = np.stack([s.source for s in cal_samples])
    t_vals = np.stack([s.solver_result.temperature for s in cal_samples])
    res_vals = [s.solver_result.diagnostics.relative_residual for s in cal_samples]
    bal_vals = [s.solver_result.diagnostics.energy_balance.relative_imbalance for s in cal_samples]

    k_min_arr = np.min(k_vals, axis=(1, 2))
    k_max_arr = np.max(k_vals, axis=(1, 2))
    contrasts = k_max_arr / np.maximum(k_min_arr, 1e-6)
    q_max_arr = np.max(q_vals, axis=(1, 2))

    print("\nCalibration Parameter Distributions Audit:")
    print(
        f"  Conductivity k: min={np.min(k_vals):.4f}, mean={np.mean(k_vals):.4f}, max={np.max(k_vals):.4f} W/(m K)"
    )
    print(
        f"  Contrast ratio: min={np.min(contrasts):.2f}, mean={np.mean(contrasts):.2f}, max={np.max(contrasts):.2f}"
    )
    print(
        f"  Heat source q:  min={np.min(q_vals):.2f}, mean={np.mean(q_vals):.2f}, max={np.max(q_vals):.2f} W/m^3"
    )
    print(
        f"  q_max per smp:  min={np.min(q_max_arr):.2f}, mean={np.mean(q_max_arr):.2f}, max={np.max(q_max_arr):.2f} W/m^3"
    )
    print(
        f"  Temperature T:  min={np.min(t_vals):.2f}, mean={np.mean(t_vals):.2f}, max={np.max(t_vals):.2f} K"
    )
    print(f"  Max Solver Res: {np.max(res_vals):.2e} (tolerance: 1e-6)")
    print(f"  Max Heat Imbal: {np.max(bal_vals):.2e} (tolerance: 1e-3)")

    assert np.min(t_vals) > 0.0, "Physical temperature positivity violation"
    assert np.max(res_vals) <= 1e-6, "Algebraic solver residual tolerance violation"
    assert np.max(bal_vals) <= 1e-3, "Physical heat conservation tolerance violation"
    assert np.mean(contrasts) < 6.0, "Calibration contrast not in IID range"

    # Write to HDF5
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = HDF5DatasetWriter()
    cal_dataset = {SplitName.CALIBRATION: cal_samples}
    writer.write(
        output_path=output_path,
        grid=grid,
        dataset=cal_dataset,
        normalizer=normalizer,
        audit=audit,
    )
    print(f"\nSuccessfully archived independent calibration dataset to {output_path}!")

    summary = inspect_hdf5_dataset(output_path)
    print(summary)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate independent calibration dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_scaled.yaml"))
    parser.add_argument(
        "--reference-dataset", type=Path, default=Path("artifacts/data/scaled_dataset.h5")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/data/calibration_dataset.h5")
    )
    parser.add_argument("--num-cal", type=int, default=200)
    args = parser.parse_args()

    generate_calibration_dataset(
        config_path=args.config,
        reference_dataset_path=args.reference_dataset,
        output_path=args.output,
        num_cal=args.num_cal,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
