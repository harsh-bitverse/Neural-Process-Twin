"""Generate Independent Confirmation Evaluation Cohort for Milestone 08G (Phase 7).

Generates a brand-new, genuinely independent evaluation cohort with disjoint seeds >= 1,000,000:
- seeds 1,000,000 - 1,000,029: 30 unseen continuous IID realizations
- seeds 1,010,000 - 1,010,029: 30 unseen continuous OOD realizations (high-contrast + short-corr GRF)
- Discretized and solved via high-precision FVM across resolutions: 32x32, 64x64, 128x128, 256x256
- Persists SHA-256 hash manifest to artifacts/manifests/independent_cohort_08g_manifest.json.

GOVERNANCE INVARIANT:
This cohort is frozen and held untouched during model development, calibration,
and threshold optimization. It is evaluated strictly once in Phase 7 for final independent confirmation.
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
logger = logging.getLogger("generate_independent_cohort_08g")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_independent_cohort_08g(
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

    # 1. Synthesize continuous specifications with disjoint seeds (1,000,000+)
    iid_specs = []
    base_seed_iid = 1_000_000
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
    base_seed_ood = 1_010_000
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["cohort_version"] = "milestone_08g_sealed_confirmation"
        h5f.attrs["creation_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h5f.attrs["num_iid"] = num_iid
        h5f.attrs["num_ood"] = num_ood
        h5f.attrs["seed_range_iid"] = f"{base_seed_iid}..{base_seed_iid + num_iid - 1}"
        h5f.attrs["seed_range_ood"] = f"{base_seed_ood}..{base_seed_ood + num_ood - 1}"

        for res in resolutions:
            logger.info(f"Discretizing and solving for resolution {res}x{res}...")
            res_grp = h5f.create_group(f"res_{res}")
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)

            for split_name, specs in [("test_iid", iid_specs), ("test_ood", ood_specs)]:
                split_grp = res_grp.create_group(split_name)
                n = len(specs)

                k_ds = split_grp.create_dataset("k", shape=(n, res, res), dtype="f4")
                q_ds = split_grp.create_dataset("q", shape=(n, res, res), dtype="f4")
                t_ds = split_grp.create_dataset("temperature", shape=(n, res, res), dtype="f4")

                bc_grp = split_grp.create_group("boundary_conditions")
                left_t = bc_grp.create_dataset("left_type", shape=(n,), dtype=h5py.string_dtype())
                right_t = bc_grp.create_dataset("right_type", shape=(n,), dtype=h5py.string_dtype())
                bottom_t = bc_grp.create_dataset(
                    "bottom_type", shape=(n,), dtype=h5py.string_dtype()
                )
                top_t = bc_grp.create_dataset("top_type", shape=(n,), dtype=h5py.string_dtype())

                left_v = bc_grp.create_dataset("left_value", shape=(n, res), dtype="f4")
                right_v = bc_grp.create_dataset("right_value", shape=(n, res), dtype="f4")
                bottom_v = bc_grp.create_dataset("bottom_value", shape=(n, res), dtype="f4")
                top_v = bc_grp.create_dataset("top_value", shape=(n, res), dtype="f4")

                subgroup_ds = split_grp.create_dataset(
                    "subgroup", shape=(n,), dtype=h5py.string_dtype()
                )
                seed_ds = split_grp.create_dataset("seed", shape=(n,), dtype="i4")

                for idx, spec in enumerate(specs):
                    k_f, q_f, bcs = discretize_continuous_fields(
                        spec["k_spec"], spec["q_spec"], spec["bc_spec"], res, res
                    )
                    res_solver = solver.solve(grid, k_f, q_f, bcs)
                    t_field = res_solver.temperature

                    k_ds[idx] = k_f.astype("f4")
                    q_ds[idx] = q_f.astype("f4")
                    t_ds[idx] = t_field.astype("f4")

                    left_t[idx] = bcs.left.bc_type.name.lower()
                    right_t[idx] = bcs.right.bc_type.name.lower()
                    bottom_t[idx] = bcs.bottom.bc_type.name.lower()
                    top_t[idx] = bcs.top.bc_type.name.lower()

                    left_v[idx] = (
                        np.full(res, bcs.left.value)
                        if np.isscalar(bcs.left.value)
                        else bcs.left.value
                    ).astype("f4")
                    right_v[idx] = (
                        np.full(res, bcs.right.value)
                        if np.isscalar(bcs.right.value)
                        else bcs.right.value
                    ).astype("f4")
                    bottom_v[idx] = (
                        np.full(res, bcs.bottom.value)
                        if np.isscalar(bcs.bottom.value)
                        else bcs.bottom.value
                    ).astype("f4")
                    top_v[idx] = (
                        np.full(res, bcs.top.value) if np.isscalar(bcs.top.value) else bcs.top.value
                    ).astype("f4")

                    subgroup_ds[idx] = spec["subgroup"]
                    seed_ds[idx] = spec["seed"]

    elapsed = time.perf_counter() - t0
    cohort_sha256 = compute_file_sha256(output_path)
    logger.info(f"Generated independent cohort in {elapsed:.1f}s. SHA-256: {cohort_sha256}")

    manifest = {
        "milestone": "08G",
        "phase": 7,
        "title": "Independent Confirmation Cohort Manifest",
        "dataset_path": str(output_path),
        "dataset_sha256": cohort_sha256,
        "num_iid": num_iid,
        "num_ood": num_ood,
        "seed_range_iid": [base_seed_iid, base_seed_iid + num_iid - 1],
        "seed_range_ood": [base_seed_ood, base_seed_ood + num_ood - 1],
        "resolutions": resolutions,
        "generation_time_seconds": elapsed,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to {manifest_path}")
    return manifest


def main() -> None:
    output_path = Path("artifacts/data/independent_cohort_08g.h5")
    manifest_path = Path("artifacts/manifests/independent_cohort_08g_manifest.json")
    generate_independent_cohort_08g(output_path, manifest_path)


if __name__ == "__main__":
    main()
