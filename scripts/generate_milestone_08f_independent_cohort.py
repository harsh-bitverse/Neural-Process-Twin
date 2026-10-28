"""Generate Independent Confirmation Evaluation Cohort for Milestone 08F (Phase 9).

Generates a brand-new, genuinely independent, previously unused evaluation cohort:
- seeds 900,000 - 900,049: 50 unseen continuous IID realizations
- seeds 910,000 - 910,049: 50 unseen continuous OOD realizations (high-contrast + short-corr GRF)
- Discretized and solved via high-precision FVM across resolutions: 32x32, 64x64, 128x128, 256x256
- Persists SHA-256 hash manifest to artifacts/manifests/independent_cohort_08f_manifest.json.

GOVERNANCE INVARIANT:
This cohort is frozen and held untouched during model development, calibration,
and threshold optimization. It is evaluated strictly once in Phase 9 for final independent confirmation.
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
logger = logging.getLogger("generate_independent_cohort_08f")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_independent_cohort_08f(
    output_path: Path,
    manifest_path: Path,
    num_iid: int = 50,
    num_ood: int = 50,
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

    # 1. Synthesize continuous specifications with disjoint seeds (900,000+)
    iid_specs = []
    base_seed_iid = 900_000
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
    base_seed_ood = 910_000
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
        meta_grp = h5f.create_group("metadata")
        meta_grp.attrs["num_iid"] = num_iid
        meta_grp.attrs["num_ood"] = num_ood
        meta_grp.attrs["resolutions"] = resolutions
        meta_grp.attrs["base_seed_iid"] = base_seed_iid
        meta_grp.attrs["base_seed_ood"] = base_seed_ood
        meta_grp.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta_grp.attrs["purpose"] = "Milestone 08F Independent Confirmation Evaluation Cohort"

        for res in resolutions:
            logger.info(
                f"Discretizing and solving FVM reference solutions for resolution {res}x{res}..."
            )
            res_grp = h5f.create_group(f"res_{res}")
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)

            for split_name, specs in [("test_iid", iid_specs), ("test_ood", ood_specs)]:
                split_grp = res_grp.create_group(split_name)
                n_samples = len(specs)

                k_ds = split_grp.create_dataset("k", shape=(n_samples, res, res), dtype="f4")
                q_ds = split_grp.create_dataset("q", shape=(n_samples, res, res), dtype="f4")
                t_ds = split_grp.create_dataset(
                    "temperature", shape=(n_samples, res, res), dtype="f4"
                )

                bc_grp = split_grp.create_group("boundary_conditions")
                left_type_ds = bc_grp.create_dataset(
                    "left_type", shape=(n_samples,), dtype=h5py.string_dtype()
                )
                right_type_ds = bc_grp.create_dataset(
                    "right_type", shape=(n_samples,), dtype=h5py.string_dtype()
                )
                bottom_type_ds = bc_grp.create_dataset(
                    "bottom_type", shape=(n_samples,), dtype=h5py.string_dtype()
                )
                top_type_ds = bc_grp.create_dataset(
                    "top_type", shape=(n_samples,), dtype=h5py.string_dtype()
                )

                left_val_ds = bc_grp.create_dataset(
                    "left_value", shape=(n_samples, res), dtype="f4"
                )
                right_val_ds = bc_grp.create_dataset(
                    "right_value", shape=(n_samples, res), dtype="f4"
                )
                bottom_val_ds = bc_grp.create_dataset(
                    "bottom_value", shape=(n_samples, res), dtype="f4"
                )
                top_val_ds = bc_grp.create_dataset("top_value", shape=(n_samples, res), dtype="f4")

                meta_sub_grp = split_grp.create_group("sample_metadata")
                seed_ds = meta_sub_grp.create_dataset("seed", shape=(n_samples,), dtype="i8")
                subgroup_ds = meta_sub_grp.create_dataset(
                    "subgroup", shape=(n_samples,), dtype=h5py.string_dtype()
                )
                sample_id_ds = meta_sub_grp.create_dataset(
                    "sample_id", shape=(n_samples,), dtype=h5py.string_dtype()
                )

                for idx, spec in enumerate(specs):
                    k_disc, q_disc, bcs_disc = discretize_continuous_fields(
                        spec["k_spec"], spec["q_spec"], spec["bc_spec"], res, res
                    )
                    sol_res = solver.solve(grid, k_disc, q_disc, bcs_disc)

                    k_ds[idx] = k_disc.astype("f4")
                    q_ds[idx] = q_disc.astype("f4")
                    t_ds[idx] = sol_res.temperature.astype("f4")

                    left_type_ds[idx] = bcs_disc.left.bc_type.value
                    right_type_ds[idx] = bcs_disc.right.bc_type.value
                    bottom_type_ds[idx] = bcs_disc.bottom.bc_type.value
                    top_type_ds[idx] = bcs_disc.top.bc_type.value

                    evals = bcs_disc.evaluate_all(grid)
                    left_val_ds[idx] = evals["left"].astype("f4")
                    right_val_ds[idx] = evals["right"].astype("f4")
                    bottom_val_ds[idx] = evals["bottom"].astype("f4")
                    top_val_ds[idx] = evals["top"].astype("f4")

                    seed_ds[idx] = spec["seed"]
                    subgroup_ds[idx] = spec["subgroup"]
                    sample_id_ds[idx] = spec["sample_id"]

    elapsed = time.perf_counter() - t0
    sha256_hash = compute_file_sha256(output_path)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    manifest = {
        "dataset_name": "independent_cohort_08f.h5",
        "sha256": sha256_hash,
        "file_size_mb": round(file_size_mb, 2),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generation_time_seconds": round(elapsed, 2),
        "num_iid_samples": num_iid,
        "num_ood_samples": num_ood,
        "resolutions": resolutions,
        "base_seed_iid": base_seed_iid,
        "base_seed_ood": base_seed_ood,
        "iid_seeds": [base_seed_iid + i for i in range(num_iid)],
        "ood_seeds": [base_seed_ood + i for i in range(num_ood)],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Independent cohort successfully generated: {output_path} ({file_size_mb:.2f} MB, SHA: {sha256_hash[:12]}...) in {elapsed:.1f}s"
    )
    return manifest


if __name__ == "__main__":
    out_file = Path("artifacts/data/independent_cohort_08f.h5")
    man_file = Path("artifacts/data/independent_cohort_08f_manifest.json")
    generate_independent_cohort_08f(out_file, man_file, num_iid=50, num_ood=50)
