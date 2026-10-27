#!/usr/bin/env python3
"""Dataset inspection CLI for Neural Process Twin.

Inspects an existing HDF5 dataset archive, displaying metadata, schema version,
partition sample counts, field ranges, conservation residual limits, and audit logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from neural_process_twin.data import inspect_hdf5_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect an archived HDF5 dataset for Neural Process Twin."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("artifacts/data/heat_conduction_dataset.h5"),
        help="Path to HDF5 dataset archive (default: artifacts/data/heat_conduction_dataset.h5).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.path.is_file():
        print(f"Error: dataset file does not exist at '{args.path}'", file=sys.stderr)
        return 1

    summary = inspect_hdf5_dataset(args.path)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
