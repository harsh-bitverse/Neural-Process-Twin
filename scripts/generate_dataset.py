#!/usr/bin/env python3
"""Dataset generation CLI for Neural Process Twin.

Synthesizes reference solution ensembles for 2D steady-state heat conduction,
orchestrating conservative FVM numerical reference solves, physical heat balance
diagnostics, HDF5 chunked compression, and dataset partitioning into
train, validation, test_iid, and test_ood splits.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from neural_process_twin.data import (
    DatasetOrchestrator,
    DatasetPartitionPlan,
    HDF5DatasetWriter,
    inspect_hdf5_dataset,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.utils.config import load_data_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic reference dataset for Neural Process Twin."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to YAML configuration file (default: configs/data.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/data/heat_conduction_dataset.h5"),
        help="Destination HDF5 archive path (default: artifacts/data/heat_conduction_dataset.h5).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a rapid smoke-test generation with reduced grid resolution and sample count.",
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail fast on any solver error or diagnostic violation (default: true).",
    )
    parser.add_argument("--num-train", type=int, default=None, help="Override train sample count.")
    parser.add_argument("--num-val", type=int, default=None, help="Override val sample count.")
    parser.add_argument(
        "--num-test-iid", type=int, default=None, help="Override test_iid sample count."
    )
    parser.add_argument(
        "--num-test-ood", type=int, default=None, help="Override test_ood sample count."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Loading data configuration from: {args.config}")
    cfg = load_data_config(args.config)

    # Configure grid
    if args.smoke_test:
        print(
            "[SMOKE-TEST] Overriding grid resolution to 16x16 and sample counts for verification."
        )
        grid = Grid2D(
            x_min=cfg.domain.x_min,
            x_max=cfg.domain.x_max,
            y_min=cfg.domain.y_min,
            y_max=cfg.domain.y_max,
            nx=16,
            ny=16,
        )
        plan = DatasetPartitionPlan(
            base_seed=cfg.dataset.seed,
            num_train=args.num_train if args.num_train is not None else 8,
            num_val=args.num_val if args.num_val is not None else 2,
            num_test_iid=args.num_test_iid if args.num_test_iid is not None else 2,
            num_test_ood=args.num_test_ood if args.num_test_ood is not None else 2,
        )
    else:
        grid = Grid2D(
            x_min=cfg.domain.x_min,
            x_max=cfg.domain.x_max,
            y_min=cfg.domain.y_min,
            y_max=cfg.domain.y_max,
            nx=cfg.grid.nx,
            ny=cfg.grid.ny,
        )
        plan = DatasetPartitionPlan(
            base_seed=cfg.dataset.seed,
            num_train=args.num_train if args.num_train is not None else cfg.dataset.num_train,
            num_val=args.num_val if args.num_val is not None else cfg.dataset.num_val,
            num_test_iid=args.num_test_iid
            if args.num_test_iid is not None
            else cfg.dataset.num_test_iid,
            num_test_ood=args.num_test_ood
            if args.num_test_ood is not None
            else cfg.dataset.num_test_ood,
        )

    print(
        f"Partition plan: train={plan.num_train}, val={plan.num_val}, "
        f"test_iid={plan.num_test_iid}, test_ood={plan.num_test_ood} "
        f"(Total: {plan.total_samples})"
    )

    orchestrator = DatasetOrchestrator(
        grid=grid,
        fail_fast=args.fail_fast,
    )

    t0 = time.perf_counter()
    print("Beginning reference solution synthesis...")
    dataset, audit = orchestrator.generate_dataset(plan)
    elapsed = time.perf_counter() - t0
    print(
        f"Generated {audit.successful_samples} verified samples in {elapsed:.2f}s "
        f"({audit.successful_samples / max(elapsed, 1e-6):.1f} samples/s). "
        f"Failed attempts: {audit.failed_samples}."
    )

    # Save to HDF5
    writer = HDF5DatasetWriter()
    print(f"Archiving dataset to {args.output}...")
    writer.write(
        output_path=args.output,
        grid=grid,
        dataset=dataset,
        audit=audit,
    )

    # Inspect written dataset
    summary = inspect_hdf5_dataset(args.output)
    print("\nDataset generation and archiving complete! Archive summary:")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
