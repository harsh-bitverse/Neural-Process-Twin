"""Generate Independent Confirmation Evaluation Cohort for Milestone 08E (Phase 7).

Generates a brand-new, genuinely independent, previously unused evaluation cohort:
- seeds 800,000 - 800,099: 100 unseen continuous IID realizations
- seeds 810,000 - 810,099: 100 unseen continuous OOD realizations (high-contrast + short-corr GRF)
- Evaluated across resolutions: 32x32, 64x64, 128x128, 256x256
- Persists SHA-256 hash to artifacts/data/independent_cohort_08e_manifest.json.

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
logger = logging.getLogger("generate_independent_cohort_08e")


def compute_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_independent_cohort_08e(
    output_path: Path,
    manifest_path: Path,
    num_iid: int = 100,
    num_ood: int = 100,
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

    # 1. Synthesize continuous specifications with disjoint seeds (800,000+)
    iid_specs = []
    base_seed_iid = 800_000
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
    base_seed_ood = 810_000
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
    with h5py.File(output_path, "w") as f:
        f.attrs["description"] = "Milestone 08E Independent Evaluation Cohort"
        f.attrs["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        spec_grp = f.create_group("continuous_specs")
        spec_grp.create_dataset("iid", data=json.dumps(iid_specs))
        spec_grp.create_dataset("ood", data=json.dumps(ood_specs))

        for res in resolutions:
            logger.info(f"Discretizing and solving independent cohort at {res}x{res}...")
            res_grp = f.create_group(f"res_{res}")
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)

            for split_name, specs in [("eval_iid", iid_specs), ("eval_ood", ood_specs)]:
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

                for idx, item in enumerate(specs):
                    k_f, q_f, bcs = discretize_continuous_fields(
                        item["k_spec"], item["q_spec"], item["bc_spec"], nx=res, ny=res
                    )
                    res_sol = solver.solve(grid, k_f, q_f, bcs)

                    k_ds[idx] = k_f.astype("f4")
                    q_ds[idx] = q_f.astype("f4")
                    t_ds[idx] = res_sol.temperature.astype("f4")
                    seed_ds[idx] = item["seed"]
                    subgroup_ds[idx] = item["subgroup"]

                    evals = bcs.evaluate_all(grid)
                    l_val_ds[idx] = evals["left"].astype("f4")
                    r_val_ds[idx] = evals["right"].astype("f4")
                    b_val_ds[idx] = evals["bottom"].astype("f4")
                    t_val_ds[idx] = evals["top"].astype("f4")

                    l_type_ds[idx] = bcs.left.bc_type.value
                    r_type_ds[idx] = bcs.right.bc_type.value
                    b_type_ds[idx] = bcs.bottom.bc_type.value
                    t_type_ds[idx] = bcs.top.bc_type.value

    elapsed = time.perf_counter() - t0
    sha256_hash = compute_file_sha256(output_path)
    logger.info(f"Cohort generation complete in {elapsed:.1f}s. SHA-256: {sha256_hash}")

    manifest = {
        "dataset_filename": output_path.name,
        "dataset_path": str(output_path),
        "sha256": sha256_hash,
        "creation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "governance_status": "LOCKED_INDEPENDENT_EVALUATION_COHORT_08E",
        "num_iid_samples": num_iid,
        "num_ood_samples": num_ood,
        "iid_seed_range": [base_seed_iid, base_seed_iid + num_iid - 1],
        "ood_seed_range": [base_seed_ood, base_seed_ood + num_ood - 1],
        "resolutions": resolutions,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    return manifest


if __name__ == "__main__":
    out_file = Path("artifacts/data/independent_cohort_08e.h5")
    manifest_file = Path("artifacts/data/independent_cohort_08e_manifest.json")
    manifest = generate_independent_cohort_08e(out_file, manifest_file)
    print(f"\nLocked cohort created: {out_file} (SHA256: {manifest['sha256']})")
