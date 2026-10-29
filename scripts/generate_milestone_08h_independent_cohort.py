"""Generate Independent Confirmation Evaluation Cohort for Milestone 08H (Phase 8).

Generates a brand-new, genuinely independent evaluation cohort with disjoint seeds >= 1,100,000:
- seeds 1,100,000 - 1,100,029: 30 unseen continuous IID realizations
- seeds 1,110,000 - 1,110,029: 30 unseen continuous OOD realizations (high-contrast + short-corr GRF)
- Discretized and solved via high-precision FVM across resolutions: 32x32, 64x64, 128x128, 256x256
- Persists SHA-256 hash manifest to artifacts/manifests/independent_cohort_08h_manifest.json.

GOVERNANCE INVARIANT:
This cohort is synthesized with disjoint seeds >= 1,100,000 and evaluated strictly after
all estimators and threshold policies have been frozen.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_confirmation_dataset import (
    compute_continuous_iid_sample,
    compute_continuous_ood_sample,
    discretize_continuous_fields,
)

from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_independent_cohort_08h")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_independent_cohort_08h(
    output_path: Path,
    manifest_path: Path,
    num_iid: int = 30,
    num_ood: int = 30,
    resolutions: list[int] | None = None,
) -> dict[str, Any]:
    """Generate independent confirmation cohort across resolutions."""
    if resolutions is None:
        resolutions = [32, 64, 128, 256]

    logger.info(
        f"Synthesizing {num_iid} IID + {num_ood} OOD independent evaluation samples across {resolutions}..."
    )
    t0 = time.perf_counter()
    solver = HeatConductionSolver2D()

    # 1. Synthesize continuous specifications with disjoint seeds (1,100,000+)
    iid_specs = []
    base_seed_iid = 1_100_000
    for i in range(num_iid):
        seed = base_seed_iid + i
        k_spec, q_spec, bc_spec = compute_continuous_iid_sample(seed)
        iid_specs.append(
            {
                "sample_id": f"eval_iid_{i:04d}",
                "seed": seed,
                "subgroup": "iid_smooth_grf",
                "k_spec": k_spec,
                "q_spec": q_spec,
                "bc_spec": bc_spec,
            }
        )

    ood_specs = []
    base_seed_ood = 1_110_000
    for i in range(num_ood):
        seed = base_seed_ood + i
        k_spec, q_spec, bc_spec, subgroup = compute_continuous_ood_sample(seed)
        ood_specs.append(
            {
                "sample_id": f"eval_ood_{i:04d}",
                "seed": seed,
                "subgroup": subgroup,
                "k_spec": k_spec,
                "q_spec": q_spec,
                "bc_spec": bc_spec,
            }
        )

    all_specs = iid_specs + ood_specs
    n_total = len(all_specs)

    # 2. Build HDF5 dataset structure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        meta_grp = f.create_group("metadata")
        meta_grp.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta_grp.attrs["purpose"] = "Milestone 08H Independent Confirmation Evaluation"
        meta_grp.attrs["base_seed_iid"] = base_seed_iid
        meta_grp.attrs["base_seed_ood"] = base_seed_ood
        meta_grp.attrs["num_iid"] = num_iid
        meta_grp.attrs["num_ood"] = num_ood
        meta_grp.attrs["resolutions"] = resolutions

        for res in resolutions:
            logger.info(f"Discretizing and solving {n_total} samples at {res}x{res}...")
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
            res_grp = f.create_group(f"res_{res}")

            k_ds = res_grp.create_dataset(
                "k", shape=(n_total, res, res), dtype=np.float32, compression="gzip"
            )
            q_ds = res_grp.create_dataset(
                "q", shape=(n_total, res, res), dtype=np.float32, compression="gzip"
            )
            t_ds = res_grp.create_dataset(
                "temperature", shape=(n_total, res, res), dtype=np.float32, compression="gzip"
            )

            bc_grp = res_grp.create_group("boundary_conditions")
            left_type = bc_grp.create_dataset(
                "left_type", shape=(n_total,), dtype=h5py.string_dtype()
            )
            right_type = bc_grp.create_dataset(
                "right_type", shape=(n_total,), dtype=h5py.string_dtype()
            )
            bottom_type = bc_grp.create_dataset(
                "bottom_type", shape=(n_total,), dtype=h5py.string_dtype()
            )
            top_type = bc_grp.create_dataset(
                "top_type", shape=(n_total,), dtype=h5py.string_dtype()
            )

            left_val = bc_grp.create_dataset("left_value", shape=(n_total, res), dtype=np.float32)
            right_val = bc_grp.create_dataset("right_value", shape=(n_total, res), dtype=np.float32)
            bottom_val = bc_grp.create_dataset(
                "bottom_value", shape=(n_total, res), dtype=np.float32
            )
            top_val = bc_grp.create_dataset("top_value", shape=(n_total, res), dtype=np.float32)

            sample_ids = res_grp.create_dataset(
                "sample_ids", shape=(n_total,), dtype=h5py.string_dtype()
            )
            subgroups = res_grp.create_dataset(
                "subgroups", shape=(n_total,), dtype=h5py.string_dtype()
            )
            seeds_ds = res_grp.create_dataset("seeds", shape=(n_total,), dtype=np.int64)

            for idx, spec in enumerate(all_specs):
                k_field, q_field, bcs = discretize_continuous_fields(
                    spec["k_spec"], spec["q_spec"], spec["bc_spec"], nx=res, ny=res
                )
                t_sol = solver.solve(grid, k_field, q_field, bcs)

                k_ds[idx] = k_field.astype(np.float32)
                q_ds[idx] = q_field.astype(np.float32)
                t_ds[idx] = t_sol.temperature.astype(np.float32)

                left_type[idx] = bcs.left.bc_type.value
                right_type[idx] = bcs.right.bc_type.value
                bottom_type[idx] = bcs.bottom.bc_type.value
                top_type[idx] = bcs.top.bc_type.value

                left_val[idx] = (
                    np.full(res, bcs.left.value, dtype=np.float32)
                    if np.isscalar(bcs.left.value)
                    else bcs.left.value.astype(np.float32)
                )
                right_val[idx] = (
                    np.full(res, bcs.right.value, dtype=np.float32)
                    if np.isscalar(bcs.right.value)
                    else bcs.right.value.astype(np.float32)
                )
                bottom_val[idx] = (
                    np.full(res, bcs.bottom.value, dtype=np.float32)
                    if np.isscalar(bcs.bottom.value)
                    else bcs.bottom.value.astype(np.float32)
                )
                top_val[idx] = (
                    np.full(res, bcs.top.value, dtype=np.float32)
                    if np.isscalar(bcs.top.value)
                    else bcs.top.value.astype(np.float32)
                )

                sample_ids[idx] = spec["sample_id"]
                subgroups[idx] = spec["subgroup"]
                seeds_ds[idx] = spec["seed"]

    total_time = time.perf_counter() - t0
    file_sha256 = compute_file_sha256(output_path)
    file_size_bytes = output_path.stat().st_size

    manifest = {
        "filename": str(output_path.name),
        "filepath": str(output_path),
        "sha256": file_sha256,
        "size_bytes": file_size_bytes,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Milestone 08H Pristine Independent Confirmation Evaluation Cohort",
        "provenance": {
            "disjoint_seed_ranges": {
                "iid": [base_seed_iid, base_seed_iid + num_iid - 1],
                "ood": [base_seed_ood, base_seed_ood + num_ood - 1],
            },
            "num_samples_per_resolution": n_total,
            "resolutions": resolutions,
            "synthesis_wall_clock_seconds": total_time,
        },
        "sample_manifest": [
            {
                "index": i,
                "sample_id": s["sample_id"],
                "seed": s["seed"],
                "subgroup": s["subgroup"],
            }
            for i, s in enumerate(all_specs)
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Independent cohort successfully created at {output_path} (SHA-256: {file_sha256})"
    )
    logger.info(f"Manifest written to {manifest_path}")

    return manifest


def main() -> None:
    output_path = Path("artifacts/data/independent_cohort_08h.h5")
    manifest_path = Path("artifacts/manifests/independent_cohort_08h_manifest.json")

    generate_independent_cohort_08h(
        output_path=output_path,
        manifest_path=manifest_path,
        num_iid=30,
        num_ood=30,
        resolutions=[32, 64, 128, 256],
    )


if __name__ == "__main__":
    main()
