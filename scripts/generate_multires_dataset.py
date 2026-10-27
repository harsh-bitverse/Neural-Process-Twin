#!/usr/bin/env python3
"""Generate the Multi-Resolution Heat Conduction Dataset for Milestone 08C.

Milestone 08C Phase 2: Resolution-Consistent Dataset Generation.
Generates physically equivalent heat-conduction problems sampled across multiple resolutions:
- 32x32 (standard resolution)
- 64x64 (fine resolution)
- 128x128 (super-resolution extrapolation)
- 256x256 (ultra-high resolution evaluation on subset)

For each physical realization, continuous conductivity, heat-source, and boundary-condition
functions are synthesized first, then sampled onto each grid. The underlying continuous
realizations are identical across resolutions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_multires_dataset")


def synthesize_continuous_iid(seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Synthesize continuous IID problem specification."""
    rng = np.random.default_rng(seed)

    # Continuous Gaussian Fourier GRF (Gaussian kernel)
    corr_length = float(rng.uniform(0.15, 0.30))
    k_min, k_max = 0.2, 5.0
    num_modes = 20

    kx = np.arange(-num_modes, num_modes + 1)
    ky = np.arange(-num_modes, num_modes + 1)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_sq = (2.0 * np.pi * kx_grid) ** 2 + (2.0 * np.pi * ky_grid) ** 2

    psd = 2.0 * np.pi * (corr_length**2) * np.exp(-0.5 * (corr_length**2) * k_sq)
    amp = np.sqrt(psd)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=kx_grid.shape)
    weights = rng.standard_normal(kx_grid.shape) * amp

    k_spec = {
        "type": "spectral_grf",
        "corr_length": corr_length,
        "k_min": k_min,
        "k_max": k_max,
        "kx": kx_grid.tolist(),
        "ky": ky_grid.tolist(),
        "weights": weights.tolist(),
        "phases": phases.tolist(),
    }

    # Heat source: 1 to 3 Gaussian peaks
    num_peaks = int(rng.integers(1, 4))
    peaks = []
    for _ in range(num_peaks):
        amp_p = float(rng.uniform(200.0, 1000.0))
        spread_p = float(rng.uniform(0.06, 0.18))
        cx = float(rng.uniform(0.1, 0.9))
        cy = float(rng.uniform(0.1, 0.9))
        peaks.append({"amplitude": amp_p, "spread": spread_p, "cx": cx, "cy": cy})
    q_spec = {"type": "gaussian_blobs", "peaks": peaks}

    # Boundary conditions
    t_left = float(rng.uniform(290.0, 340.0))
    t_right = float(rng.uniform(310.0, 360.0))
    b_is_dir = rng.random() < 0.5
    t_is_dir = rng.random() < 0.5
    b_val = float(rng.uniform(290.0, 350.0)) if b_is_dir else float(rng.uniform(-500.0, 500.0))
    t_val = float(rng.uniform(290.0, 350.0)) if t_is_dir else float(rng.uniform(-500.0, 500.0))

    bc_spec = {
        "left": {"type": "dirichlet", "value": t_left},
        "right": {"type": "dirichlet", "value": t_right},
        "bottom": {"type": "dirichlet" if b_is_dir else "neumann", "value": b_val},
        "top": {"type": "dirichlet" if t_is_dir else "neumann", "value": t_val},
    }

    return k_spec, q_spec, bc_spec


def synthesize_continuous_ood(
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Synthesize continuous OOD problem specification."""
    rng = np.random.default_rng(seed)

    if rng.random() < 0.7:
        subgroup = "high_contrast_inclusions"
        num_inc = int(rng.integers(2, 5))
        matrix_k = 1.0
        inclusions = []
        for _ in range(num_inc):
            contrast = float(rng.uniform(20.0, 50.0))
            k_inc = matrix_k * contrast
            radius = float(rng.uniform(0.06, 0.16))
            cx = float(rng.uniform(radius + 0.05, 1.0 - radius - 0.05))
            cy = float(rng.uniform(radius + 0.05, 1.0 - radius - 0.05))
            inclusions.append({"cx": cx, "cy": cy, "radius": radius, "k_inc": k_inc})
        k_spec = {
            "type": "composite_inclusions",
            "matrix_k": matrix_k,
            "inclusions": inclusions,
        }
    else:
        subgroup = "short_correlation_grf"
        corr_length = float(rng.uniform(0.04, 0.08))
        k_min, k_max = 0.5, 25.0
        num_modes = 28

        kx = np.arange(-num_modes, num_modes + 1)
        ky = np.arange(-num_modes, num_modes + 1)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        k_sq = (2.0 * np.pi * kx_grid) ** 2 + (2.0 * np.pi * ky_grid) ** 2

        # Matérn nu=1.5
        nu = 1.5
        c_factor = 4.0 * np.pi * nu * (2.0 * nu / (corr_length**2)) ** nu
        psd = c_factor * (2.0 * nu / (corr_length**2) + k_sq) ** (-(nu + 1.0))
        amp = np.sqrt(psd)
        phases = rng.uniform(0.0, 2.0 * np.pi, size=kx_grid.shape)
        weights = rng.standard_normal(kx_grid.shape) * amp

        k_spec = {
            "type": "spectral_grf",
            "corr_length": corr_length,
            "k_min": k_min,
            "k_max": k_max,
            "kx": kx_grid.tolist(),
            "ky": ky_grid.tolist(),
            "weights": weights.tolist(),
            "phases": phases.tolist(),
        }

    # Elevated heat source
    num_peaks = int(rng.integers(2, 5))
    peaks = []
    for _ in range(num_peaks):
        amp_p = float(rng.uniform(500.0, 2000.0))
        spread_p = float(rng.uniform(0.04, 0.12))
        cx = float(rng.uniform(0.1, 0.9))
        cy = float(rng.uniform(0.1, 0.9))
        peaks.append({"amplitude": amp_p, "spread": spread_p, "cx": cx, "cy": cy})
    q_spec = {"type": "gaussian_blobs", "peaks": peaks}

    t_left = float(rng.uniform(280.0, 350.0))
    t_right = float(rng.uniform(320.0, 380.0))
    b_is_dir = rng.random() < 0.5
    t_is_dir = rng.random() < 0.5
    b_val = float(rng.uniform(280.0, 380.0)) if b_is_dir else float(rng.uniform(-1000.0, 1000.0))
    t_val = float(rng.uniform(280.0, 380.0)) if t_is_dir else float(rng.uniform(-1000.0, 1000.0))

    bc_spec = {
        "left": {"type": "dirichlet", "value": t_left},
        "right": {"type": "dirichlet", "value": t_right},
        "bottom": {"type": "dirichlet" if b_is_dir else "neumann", "value": b_val},
        "top": {"type": "dirichlet" if t_is_dir else "neumann", "value": t_val},
    }

    return k_spec, q_spec, bc_spec, subgroup


def discretize_fields(
    k_spec: dict[str, Any],
    q_spec: dict[str, Any],
    bc_spec: dict[str, Any],
    grid: Grid2D,
) -> tuple[np.ndarray, np.ndarray, BoundaryConditions]:
    """Sample continuous functions onto grid."""
    ny, nx = grid.ny, grid.nx
    xc, yc = grid.x_mesh, grid.y_mesh

    if k_spec["type"] == "composite_inclusions":
        k = np.full((ny, nx), k_spec["matrix_k"], dtype=np.float64)
        for inc in k_spec["inclusions"]:
            dist_sq = (xc - inc["cx"]) ** 2 + (yc - inc["cy"]) ** 2
            k[dist_sq <= inc["radius"] ** 2] = inc["k_inc"]
    elif k_spec["type"] == "spectral_grf":
        kx = np.array(k_spec["kx"]).flatten()
        ky = np.array(k_spec["ky"]).flatten()
        weights = np.array(k_spec["weights"]).flatten()
        phases = np.array(k_spec["phases"]).flatten()

        xi = np.zeros((ny, nx), dtype=np.float64)
        for idx in range(len(weights)):
            w = weights[idx]
            if abs(w) < 1e-12:
                continue
            arg = 2.0 * np.pi * (kx[idx] * xc + ky[idx] * yc) + phases[idx]
            xi += w * np.cos(arg)

        xi_std = float(np.std(xi))
        xi_norm = (xi - float(np.mean(xi))) / xi_std if xi_std > 1e-12 else xi
        sig = 1.0 / (1.0 + np.exp(-np.clip(xi_norm, -30.0, 30.0)))
        k = k_spec["k_min"] + (k_spec["k_max"] - k_spec["k_min"]) * sig
    else:
        raise ValueError(k_spec["type"])

    q = np.zeros((ny, nx), dtype=np.float64)
    for p in q_spec["peaks"]:
        r_sq = (xc - p["cx"]) ** 2 + (yc - p["cy"]) ** 2
        q += p["amplitude"] * np.exp(-r_sq / (2.0 * p["spread"] ** 2))

    def make_bc(side: BoundarySide, spec: dict[str, Any]) -> BoundaryCondition:
        b_type = BoundaryType.DIRICHLET if spec["type"] == "dirichlet" else BoundaryType.NEUMANN
        return BoundaryCondition(side=side, bc_type=b_type, value=spec["value"])

    bcs = BoundaryConditions(
        left=make_bc(BoundarySide.LEFT, bc_spec["left"]),
        right=make_bc(BoundarySide.RIGHT, bc_spec["right"]),
        bottom=make_bc(BoundarySide.BOTTOM, bc_spec["bottom"]),
        top=make_bc(BoundarySide.TOP, bc_spec["top"]),
    )

    return k, q, bcs


def build_multires_dataset(
    output_path: Path,
    num_train: int = 1000,
    num_val: int = 150,
    num_test_iid: int = 200,
    num_test_ood: int = 200,
    num_extra_256: int = 50,
) -> dict[str, Any]:
    """Build multi-resolution dataset HDF5 archive across 32, 64, 128, and 256."""
    logger.info("Commencing Multi-Resolution Dataset Synthesis (Milestone 08C Phase 2)...")
    t0 = time.perf_counter()
    solver = HeatConductionSolver2D()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Synthesize all continuous specifications
    logger.info("Synthesizing continuous specifications for Train, Val, Test IID, and Test OOD...")
    train_specs = []
    for i in range(num_train):
        k_s, q_s, bc_s = synthesize_continuous_iid(10_000 + i)
        train_specs.append(
            {"sample_id": f"train_{i:05d}", "seed": 10_000 + i, "k": k_s, "q": q_s, "bc": bc_s}
        )

    val_specs = []
    for i in range(num_val):
        k_s, q_s, bc_s = synthesize_continuous_iid(11_000 + i)
        val_specs.append(
            {"sample_id": f"val_{i:05d}", "seed": 11_000 + i, "k": k_s, "q": q_s, "bc": bc_s}
        )

    test_iid_specs = []
    for i in range(num_test_iid):
        k_s, q_s, bc_s = synthesize_continuous_iid(12_000 + i)
        test_iid_specs.append(
            {"sample_id": f"test_iid_{i:05d}", "seed": 12_000 + i, "k": k_s, "q": q_s, "bc": bc_s}
        )

    test_ood_specs = []
    for i in range(num_test_ood):
        k_s, q_s, bc_s, sub = synthesize_continuous_ood(13_000 + i)
        test_ood_specs.append(
            {
                "sample_id": f"test_ood_{i:05d}",
                "seed": 13_000 + i,
                "k": k_s,
                "q": q_s,
                "bc": bc_s,
                "subgroup": sub,
            }
        )

    splits_config = [
        ("train", train_specs, [32, 64]),
        ("validation", val_specs, [32, 64]),
        ("test_iid", test_iid_specs, [32, 64, 128, 256]),
        ("test_ood", test_ood_specs, [32, 64, 128, 256]),
    ]

    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["format_version"] = "2.0.0"
        h5f.attrs["generator"] = "Neural Process Twin — Milestone 08C Multi-Resolution Pipeline"
        h5f.attrs["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for split_name, specs, res_list in splits_config:
            logger.info(
                f"Processing split '{split_name}' (N={len(specs)}) across resolutions {res_list}..."
            )
            for res in res_list:
                limit = num_extra_256 if res == 256 else len(specs)
                sub_specs = specs[:limit]

                grp_path = f"res_{res}/{split_name}"
                grp = h5f.create_group(grp_path)
                grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)

                # Store grid metadata
                grp.attrs["nx"] = res
                grp.attrs["ny"] = res
                grp.attrs["num_samples"] = limit

                k_ds = grp.create_dataset("k", shape=(limit, res, res), dtype=np.float32)
                q_ds = grp.create_dataset("q", shape=(limit, res, res), dtype=np.float32)
                t_ds = grp.create_dataset("temperature", shape=(limit, res, res), dtype=np.float32)

                sample_ids = grp.create_dataset(
                    "sample_id", shape=(limit,), dtype=h5py.string_dtype()
                )
                seeds = grp.create_dataset("seed", shape=(limit,), dtype=np.int64)
                contrasts = grp.create_dataset("contrast", shape=(limit,), dtype=np.float32)

                if "subgroup" in sub_specs[0]:
                    subgroups = grp.create_dataset(
                        "subgroup", shape=(limit,), dtype=h5py.string_dtype()
                    )
                    subgroup_ids = grp.create_dataset("subgroup_id", shape=(limit,), dtype=np.int32)

                bc_grp = grp.create_group("boundary_conditions")
                l_type_ds = bc_grp.create_dataset(
                    "left_type", shape=(limit,), dtype=h5py.string_dtype()
                )
                r_type_ds = bc_grp.create_dataset(
                    "right_type", shape=(limit,), dtype=h5py.string_dtype()
                )
                b_type_ds = bc_grp.create_dataset(
                    "bottom_type", shape=(limit,), dtype=h5py.string_dtype()
                )
                t_type_ds = bc_grp.create_dataset(
                    "top_type", shape=(limit,), dtype=h5py.string_dtype()
                )

                l_val_ds = bc_grp.create_dataset("left_value", shape=(limit, res), dtype=np.float32)
                r_val_ds = bc_grp.create_dataset(
                    "right_value", shape=(limit, res), dtype=np.float32
                )
                b_val_ds = bc_grp.create_dataset(
                    "bottom_value", shape=(limit, res), dtype=np.float32
                )
                t_val_ds = bc_grp.create_dataset("top_value", shape=(limit, res), dtype=np.float32)

                # Solve and fill
                t_res_start = time.perf_counter()
                for i, spec in enumerate(sub_specs):
                    k_np, q_np, bcs = discretize_fields(spec["k"], spec["q"], spec["bc"], grid)
                    res_sol = solver.solve(grid, k_np, q_np, bcs)

                    k_ds[i] = k_np.astype(np.float32)
                    q_ds[i] = q_np.astype(np.float32)
                    t_ds[i] = res_sol.temperature.astype(np.float32)

                    sample_ids[i] = spec["sample_id"]
                    seeds[i] = spec["seed"]
                    contrasts[i] = float(np.max(k_np) / max(np.min(k_np), 1e-12))

                    if "subgroup" in spec:
                        subgroups[i] = spec["subgroup"]
                        subgroup_ids[i] = 1 if spec["subgroup"] == "short_correlation_grf" else 0

                    evals = bcs.evaluate_all(grid)
                    l_type_ds[i] = bcs.left.bc_type.value
                    r_type_ds[i] = bcs.right.bc_type.value
                    b_type_ds[i] = bcs.bottom.bc_type.value
                    t_type_ds[i] = bcs.top.bc_type.value

                    l_val_ds[i] = evals["left"].astype(np.float32)
                    r_val_ds[i] = evals["right"].astype(np.float32)
                    b_val_ds[i] = evals["bottom"].astype(np.float32)
                    t_val_ds[i] = evals["top"].astype(np.float32)

                t_res_elapsed = time.perf_counter() - t_res_start
                logger.info(f"  Solved {grp_path} ({limit} samples) in {t_res_elapsed:.2f} s")

    elapsed = time.perf_counter() - t0
    sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
    file_size_bytes = output_path.stat().st_size
    logger.info(
        f"Multi-Resolution Dataset synthesized in {elapsed:.1f} s | Size: {file_size_bytes} bytes | SHA-256: {sha256}"
    )

    manifest = {
        "dataset_name": "multires_dataset",
        "file_path": str(output_path),
        "file_size_bytes": file_size_bytes,
        "sha256": sha256,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolutions": [32, 64, 128, 256],
        "splits": {
            "train": {"num_samples": num_train, "resolutions": [32, 64]},
            "validation": {"num_samples": num_val, "resolutions": [32, 64]},
            "test_iid": {"num_samples": num_test_iid, "resolutions": [32, 64, 128, 256]},
            "test_ood": {"num_samples": num_test_ood, "resolutions": [32, 64, 128, 256]},
        },
    }

    manifest_path = output_path.parent / "multires_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    return manifest


if __name__ == "__main__":
    out_file = Path("artifacts/data/multires_dataset.h5")
    build_multires_dataset(out_file)
