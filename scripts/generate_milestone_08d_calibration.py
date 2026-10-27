#!/usr/bin/env python3
"""Generate Dedicated Calibration Partition for Milestone 08D.

Generates 200 independent multi-resolution physical samples (seeds 11150 - 11349)
sampled at 32x32 and 64x64 resolutions for fitting uncertainty calibration thresholds,
conformal prediction quantiles, and safety routing gates.

Disjointness Invariant:
- train: seeds 10000 - 10999
- validation: seeds 11000 - 11149
- calibration: seeds 11150 - 11349 (THIS DATASET)
- test_iid: seeds 12000 - 12199
- test_ood: seeds 13000 - 13199
- historical confirmation: seeds 500000 - 501199

Outputs artifacts/data/multires_calibration.h5.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import h5py

from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_confirmation_dataset import discretize_continuous_fields
from scripts.generate_multires_dataset import (
    synthesize_continuous_iid,
    synthesize_continuous_ood,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_milestone_08d_calibration")


def build_calibration_dataset(
    output_path: Path,
    num_iid: int = 100,
    num_ood: int = 100,
    resolutions: list[int] | None = None,
) -> None:
    """Generate and persist multi-resolution calibration dataset."""
    if resolutions is None:
        resolutions = [32, 64]

    logger.info(
        f"Synthesizing {num_iid} IID + {num_ood} OOD calibration samples across {resolutions}..."
    )
    t0 = time.perf_counter()
    solver = HeatConductionSolver2D()

    # Continuous specs
    iid_specs = []
    base_seed_iid = 11150
    for i in range(num_iid):
        seed = base_seed_iid + i
        k_spec, q_spec, bc_spec = synthesize_continuous_iid(seed)
        iid_specs.append(
            {
                "sample_id": f"calib_iid_{i:04d}",
                "seed": seed,
                "subgroup": "iid_smooth_grf",
                "k_spec": k_spec,
                "q_spec": q_spec,
                "bc_spec": bc_spec,
            }
        )

    ood_specs = []
    base_seed_ood = 11250
    for i in range(num_ood):
        seed = base_seed_ood + i
        k_spec, q_spec, bc_spec, subgroup = synthesize_continuous_ood(seed)
        ood_specs.append(
            {
                "sample_id": f"calib_ood_{i:04d}",
                "seed": seed,
                "subgroup": subgroup,
                "k_spec": k_spec,
                "q_spec": q_spec,
                "bc_spec": bc_spec,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["description"] = "Milestone 08D Dedicated Calibration Partition"
        f.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # Store continuous specs
        spec_grp = f.create_group("continuous_specs")
        spec_grp.create_dataset("iid", data=json.dumps(iid_specs))
        spec_grp.create_dataset("ood", data=json.dumps(ood_specs))

        for res in resolutions:
            logger.info(f"Discretizing and solving calibration cohort at {res}x{res}...")
            res_grp = f.create_group(f"res_{res}")
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)

            for split_name, specs in [
                ("calibration_iid", iid_specs),
                ("calibration_ood", ood_specs),
            ]:
                split_grp = res_grp.create_group(split_name)
                n = len(specs)

                k_ds = split_grp.create_dataset("k", shape=(n, res, res), dtype="f4")
                q_ds = split_grp.create_dataset("q", shape=(n, res, res), dtype="f4")
                t_ds = split_grp.create_dataset("temperature", shape=(n, res, res), dtype="f4")
                seed_ds = split_grp.create_dataset("seed", shape=(n,), dtype="i4")
                subgroup_ds = split_grp.create_dataset(
                    "subgroup", shape=(n,), dtype=h5py.string_dtype("utf-8")
                )

                bc_grp = split_grp.create_group("boundary_conditions")
                l_val_ds = bc_grp.create_dataset("left_value", shape=(n, res), dtype="f4")
                r_val_ds = bc_grp.create_dataset("right_value", shape=(n, res), dtype="f4")
                b_val_ds = bc_grp.create_dataset("bottom_value", shape=(n, res), dtype="f4")
                t_val_ds = bc_grp.create_dataset("top_value", shape=(n, res), dtype="f4")
                l_type_ds = bc_grp.create_dataset(
                    "left_type", shape=(n,), dtype=h5py.string_dtype("utf-8")
                )
                r_type_ds = bc_grp.create_dataset(
                    "right_type", shape=(n,), dtype=h5py.string_dtype("utf-8")
                )
                b_type_ds = bc_grp.create_dataset(
                    "bottom_type", shape=(n,), dtype=h5py.string_dtype("utf-8")
                )
                t_type_ds = bc_grp.create_dataset(
                    "top_type", shape=(n,), dtype=h5py.string_dtype("utf-8")
                )

                for idx, spec in enumerate(specs):
                    k_f, q_f, bcs = discretize_continuous_fields(
                        spec["k_spec"], spec["q_spec"], spec["bc_spec"], res, res
                    )
                    res_solve = solver.solve(grid, k_f, q_f, bcs)

                    k_ds[idx] = k_f.astype("f4")
                    q_ds[idx] = q_f.astype("f4")
                    t_ds[idx] = res_solve.temperature.astype("f4")
                    seed_ds[idx] = spec["seed"]
                    subgroup_ds[idx] = spec["subgroup"]

                    bc_eval = bcs.evaluate_all(grid)
                    l_val_ds[idx] = bc_eval["left"].astype("f4")
                    r_val_ds[idx] = bc_eval["right"].astype("f4")
                    b_val_ds[idx] = bc_eval["bottom"].astype("f4")
                    t_val_ds[idx] = bc_eval["top"].astype("f4")

                    l_type_ds[idx] = bcs.left.bc_type.value
                    r_type_ds[idx] = bcs.right.bc_type.value
                    b_type_ds[idx] = bcs.bottom.bc_type.value
                    t_type_ds[idx] = bcs.top.bc_type.value

    elapsed = time.perf_counter() - t0
    logger.info(f"Generated {output_path} in {elapsed:.2f} s")


if __name__ == "__main__":
    out_file = Path("artifacts/data/multires_calibration.h5")
    build_calibration_dataset(out_file, num_iid=100, num_ood=100, resolutions=[32, 64])
