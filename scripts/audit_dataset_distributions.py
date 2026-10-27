"""Audit dataset distributions, partition disjointness, and solver convergence."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from neural_process_twin.data.splits import verify_partitions_disjoint


def audit_dataset(dataset_path: str | Path) -> dict[str, dict]:
    path = Path(dataset_path)
    report = {}

    with h5py.File(path, "r") as f:
        splits = ["train", "validation", "test_iid", "test_ood"]
        split_seeds: dict[str, list[int]] = {}

        for split in splits:
            grp = f[split]
            k = grp["k"][:]
            q = grp["q"][:]
            t = grp["temperature"][:]
            sample_ids = [
                s.decode("utf-8") if isinstance(s, bytes) else s for s in grp["sample_id"][:]
            ]
            seeds = [int(s) for s in grp["seed"][:]]
            split_seeds[split] = seeds

            # Diagnostics
            rel_res = grp["diagnostics"]["relative_residual"][:]
            imbalance = grp["diagnostics"]["heat_imbalance"][:]

            # Boundary conditions
            bc_grp = grp["boundary_conditions"]
            left_v = bc_grp["left_value"][:]
            right_v = bc_grp["right_value"][:]
            bottom_v = bc_grp["bottom_value"][:]
            top_v = bc_grp["top_value"][:]

            # Contrast ratio per sample
            contrasts = [float(np.max(k[i]) / (np.min(k[i]) + 1e-12)) for i in range(len(k))]

            report[split] = {
                "count": len(sample_ids),
                "k_min": float(np.min(k)),
                "k_max": float(np.max(k)),
                "k_mean": float(np.mean(k)),
                "k_std": float(np.std(k)),
                "contrast_min": float(np.min(contrasts)),
                "contrast_max": float(np.max(contrasts)),
                "contrast_mean": float(np.mean(contrasts)),
                "q_min": float(np.min(q)),
                "q_max": float(np.max(q)),
                "q_mean": float(np.mean(q)),
                "t_min": float(np.min(t)),
                "t_max": float(np.max(t)),
                "t_mean": float(np.mean(t)),
                "t_std": float(np.std(t)),
                "max_relative_residual": float(np.max(rel_res)),
                "max_heat_imbalance": float(np.max(imbalance)),
                "bc_left_mean": float(np.mean(left_v)),
                "bc_right_mean": float(np.mean(right_v)),
                "bc_bottom_mean": float(np.mean(bottom_v)),
                "bc_top_mean": float(np.mean(top_v)),
            }

        # Check disjointness
        verify_partitions_disjoint(split_seeds)
        report["partitions_disjoint"] = True

        # Check normalization
        if "normalization" in f:
            norm_grp = f["normalization"]
            t_mean_norm = norm_grp["t_stats"].attrs["mean"]
            t_std_norm = norm_grp["t_stats"].attrs["std"]
            t_train_mean = report["train"]["t_mean"]
            t_train_std = report["train"]["t_std"]
            report["normalization_matches_train"] = bool(
                np.isclose(t_mean_norm, t_train_mean, atol=1e-3)
                and np.isclose(t_std_norm, t_train_std, atol=1e-3)
            )

    return report


if __name__ == "__main__":
    rep = audit_dataset("artifacts/data/scaled_dataset.h5")
    print(json.dumps(rep, indent=2))
