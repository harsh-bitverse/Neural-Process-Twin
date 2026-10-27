#!/usr/bin/env python3
"""Generate the Touchstone Confirmation Dataset for Milestone 08C.

Milestone 08C Phase 1: Touchstone Confirmation Dataset.
Generates an independent, versioned confirmation dataset with:
- Completely disjoint random seeds (>= 500,000)
- 200 independent IID samples (seeds 500,000 - 500,199)
- 200 independent OOD samples (seeds 501,000 - 501,199)
- Explicit per-sample OOD subgroup identification (high_contrast_inclusions vs short_correlation_grf)
- Solved at both 32x32 and 64x64 resolutions using the reference FVM solver
- Persists SHA-256 hash and manifest to artifacts/data/confirmation_manifest.json

CRITICAL GOVERNANCE INVARIANT:
This dataset is held in escrow. It must NOT be accessed for model development,
training, or hyperparameter selection. It is evaluated exactly once in Phase 7.
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
logger = logging.getLogger("generate_confirmation_dataset")


def compute_continuous_iid_sample(
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Synthesize continuous representation for an IID problem.

    IID Specification:
    - Conductivity: Continuous Gaussian Fourier modes with Gaussian covariance kernel,
      correlation length ell in [0.15, 0.30] m, scaled via sigmoid to [0.2, 5.0] W/(m K).
    - Source: 1 to 3 Gaussian peaks with amplitude in [200, 1000] W/m^3, spread in [0.06, 0.18] m.
    - Boundary Conditions: Mixed Dirichlet-Neumann with T_dir in [290, 360] K, j_neu in [-500, 500] W/m^2.
    """
    rng = np.random.default_rng(seed)

    # 1. Continuous Gaussian Random Field parameters (spectral representation)
    corr_length = float(rng.uniform(0.15, 0.30))
    k_min, k_max = 0.2, 5.0
    num_fourier_modes = 24

    # Precompute random Fourier coefficients for continuous evaluation
    # ell defines Gaussian spectral envelope: S(kx, ky) = 2 * pi * ell^2 * exp(-0.5 * ell^2 * (kx^2 + ky^2))
    kx_vals = np.arange(-num_fourier_modes, num_fourier_modes + 1)
    ky_vals = np.arange(-num_fourier_modes, num_fourier_modes + 1)
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals, indexing="ij")
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

    # 2. Continuous Gaussian blobs for heat source
    num_peaks = int(rng.integers(1, 4))
    peaks = []
    for _ in range(num_peaks):
        amp_p = float(rng.uniform(200.0, 1000.0))
        spread_p = float(rng.uniform(0.06, 0.18))
        cx = float(rng.uniform(0.1, 0.9))
        cy = float(rng.uniform(0.1, 0.9))
        peaks.append({"amplitude": amp_p, "spread": spread_p, "cx": cx, "cy": cy})

    q_spec = {"type": "gaussian_blobs", "peaks": peaks}

    # 3. Continuous boundary conditions
    # Left: Dirichlet, Right: Dirichlet, Bottom/Top: Neumann or Dirichlet
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


def compute_continuous_ood_sample(
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Synthesize continuous representation for an OOD problem.

    OOD Specification:
    - 70% probability: High-Contrast Inclusions (contrast ratio 20:1 to 50:1, radius 0.06 to 0.16 m)
    - 30% probability: Short-Correlation GRF (Matérn nu=1.5, ell in [0.04, 0.08] m, k in [0.5, 25.0])
    - Source: 2 to 4 Gaussian peaks with elevated amplitude in [500, 2000] W/m^3.
    - Boundary Conditions: Mixed with T_dir in [280, 380] K, j_neu in [-1000, 1000] W/m^2.
    """
    rng = np.random.default_rng(seed)

    if rng.random() < 0.7:
        subgroup = "high_contrast_inclusions"
        num_inclusions = int(rng.integers(2, 5))
        matrix_k = 1.0
        inclusions = []
        for _ in range(num_inclusions):
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
        num_fourier_modes = 32

        kx_vals = np.arange(-num_fourier_modes, num_fourier_modes + 1)
        ky_vals = np.arange(-num_fourier_modes, num_fourier_modes + 1)
        kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals, indexing="ij")
        k_sq = (2.0 * np.pi * kx_grid) ** 2 + (2.0 * np.pi * ky_grid) ** 2

        # Matérn PSD (nu = 1.5): S(k) proportional to (2*nu/ell^2 + |k|^2)^(-(nu + 1))
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

    # Elevated OOD heat source
    num_peaks = int(rng.integers(2, 5))
    peaks = []
    for _ in range(num_peaks):
        amp_p = float(rng.uniform(500.0, 2000.0))
        spread_p = float(rng.uniform(0.04, 0.12))
        cx = float(rng.uniform(0.1, 0.9))
        cy = float(rng.uniform(0.1, 0.9))
        peaks.append({"amplitude": amp_p, "spread": spread_p, "cx": cx, "cy": cy})

    q_spec = {"type": "gaussian_blobs", "peaks": peaks}

    # OOD boundary conditions
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


def discretize_continuous_fields(
    k_spec: dict[str, Any],
    q_spec: dict[str, Any],
    bc_spec: dict[str, Any],
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray, BoundaryConditions]:
    """Sample continuous physical specifications onto an nx by ny Cartesian grid."""
    grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=nx, ny=ny, thickness=1.0)
    xc, yc = grid.x_mesh, grid.y_mesh

    # 1. Evaluate k(x, y)
    if k_spec["type"] == "composite_inclusions":
        k = np.full((ny, nx), k_spec["matrix_k"], dtype=np.float64)
        for inc in k_spec["inclusions"]:
            dist_sq = (xc - inc["cx"]) ** 2 + (yc - inc["cy"]) ** 2
            mask = dist_sq <= inc["radius"] ** 2
            k[mask] = inc["k_inc"]
    elif k_spec["type"] == "spectral_grf":
        kx = np.array(k_spec["kx"])
        ky = np.array(k_spec["ky"])
        weights = np.array(k_spec["weights"])
        phases = np.array(k_spec["phases"])

        # Continuous Fourier evaluation
        # xi(x, y) = sum W_mn * cos(2*pi*(kx*x + ky*y) + phi_mn)
        xi = np.zeros((ny, nx), dtype=np.float64)
        # Vectorized evaluation over modes
        flat_kx = kx.flatten()
        flat_ky = ky.flatten()
        flat_w = weights.flatten()
        flat_p = phases.flatten()

        for idx in range(len(flat_w)):
            w = flat_w[idx]
            if abs(w) < 1e-12:
                continue
            k_x = flat_kx[idx]
            k_y = flat_ky[idx]
            phi = flat_p[idx]
            arg = 2.0 * np.pi * (k_x * xc + k_y * yc) + phi
            xi += w * np.cos(arg)

        # Normalize to zero mean unit variance
        xi_std = float(np.std(xi))
        xi_norm = (xi - float(np.mean(xi))) / xi_std if xi_std > 1e-12 else xi

        # Apply sigmoid transform to strictly map inside (k_min, k_max)
        k_min = k_spec["k_min"]
        k_max = k_spec["k_max"]
        sig = 1.0 / (1.0 + np.exp(-np.clip(xi_norm, -30.0, 30.0)))
        k = k_min + (k_max - k_min) * sig
    else:
        raise ValueError(f"Unknown k_spec type: {k_spec['type']}")

    # 2. Evaluate q(x, y)
    q = np.zeros((ny, nx), dtype=np.float64)
    for p in q_spec["peaks"]:
        r_sq = (xc - p["cx"]) ** 2 + (yc - p["cy"]) ** 2
        q += p["amplitude"] * np.exp(-r_sq / (2.0 * p["spread"] ** 2))

    # 3. Formulate discrete BoundaryConditions
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


def build_confirmation_dataset(
    output_path: Path,
    num_iid: int = 200,
    num_ood: int = 200,
) -> dict[str, Any]:
    """Generate and persist the Touchstone Confirmation Dataset across 32x32 and 64x64."""
    logger.info(f"Generating Confirmation Dataset: {num_iid} IID + {num_ood} OOD samples...")
    t0 = time.perf_counter()
    solver = HeatConductionSolver2D()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Specs storage for continuous provenance
    iid_specs: list[dict[str, Any]] = []
    ood_specs: list[dict[str, Any]] = []

    # 1. Synthesize continuous specifications
    logger.info("Synthesizing continuous physical problem specifications...")
    base_seed_iid = 500_000
    for idx in range(num_iid):
        seed = base_seed_iid + idx
        k_spec, q_spec, bc_spec = compute_continuous_iid_sample(seed)
        iid_specs.append(
            {
                "sample_id": f"confirm_iid_{idx:05d}",
                "seed": seed,
                "subgroup": "iid_smooth_grf",
                "k_spec": k_spec,
                "q_spec": q_spec,
                "bc_spec": bc_spec,
            }
        )

    base_seed_ood = 501_000
    for idx in range(num_ood):
        seed = base_seed_ood + idx
        k_spec, q_spec, bc_spec, subgroup = compute_continuous_ood_sample(seed)
        ood_specs.append(
            {
                "sample_id": f"confirm_ood_{idx:05d}",
                "seed": seed,
                "subgroup": subgroup,
                "k_spec": k_spec,
                "q_spec": q_spec,
                "bc_spec": bc_spec,
            }
        )

    # Count OOD subgroups
    hc_count = sum(1 for s in ood_specs if s["subgroup"] == "high_contrast_inclusions")
    grf_count = sum(1 for s in ood_specs if s["subgroup"] == "short_correlation_grf")
    logger.info(
        f"OOD Subgroups generated: {hc_count} High-Contrast Inclusions, {grf_count} Short-Correlation GRF"
    )

    # 2. Write HDF5 archive
    logger.info(f"Writing HDF5 archive to {output_path}...")
    with h5py.File(output_path, "w") as h5f:
        h5f.attrs["format_version"] = "2.0.0"
        h5f.attrs["generator"] = (
            "Neural Process Twin — Milestone 08C Touchstone Confirmation Dataset"
        )
        h5f.attrs["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        h5f.attrs["num_iid"] = num_iid
        h5f.attrs["num_ood"] = num_ood
        h5f.attrs["ood_subgroups"] = json.dumps(
            {"high_contrast_inclusions": hc_count, "short_correlation_grf": grf_count}
        )

        # Save continuous problem specifications in JSON string dataset
        h5f.create_dataset("continuous_specs/iid", data=json.dumps(iid_specs))
        h5f.create_dataset("continuous_specs/ood", data=json.dumps(ood_specs))

        # Discretize and solve at 32x32 and 64x64
        for res in [32, 64]:
            logger.info(f"Discretizing and solving FVM reference solutions at {res}x{res}...")
            grid = Grid2D(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=res, ny=res, thickness=1.0)
            res_grp = h5f.create_group(f"res_{res}")

            # Grid coordinates
            res_grp.create_dataset("grid/x_mesh", data=grid.x_mesh)
            res_grp.create_dataset("grid/y_mesh", data=grid.y_mesh)
            res_grp.create_dataset("grid/x_centers", data=grid.x_centers)
            res_grp.create_dataset("grid/y_centers", data=grid.y_centers)

            for split_key, specs in [
                ("confirmation_iid", iid_specs),
                ("confirmation_ood", ood_specs),
            ]:
                grp = res_grp.create_group(split_key)
                n_samples = len(specs)

                k_ds = grp.create_dataset("k", shape=(n_samples, res, res), dtype=np.float32)
                q_ds = grp.create_dataset("q", shape=(n_samples, res, res), dtype=np.float32)
                t_ds = grp.create_dataset(
                    "temperature", shape=(n_samples, res, res), dtype=np.float32
                )

                sample_ids = grp.create_dataset(
                    "sample_id", shape=(n_samples,), dtype=h5py.string_dtype()
                )
                seeds = grp.create_dataset("seed", shape=(n_samples,), dtype=np.int64)
                subgroups = grp.create_dataset(
                    "subgroup", shape=(n_samples,), dtype=h5py.string_dtype()
                )
                subgroup_ids = grp.create_dataset("subgroup_id", shape=(n_samples,), dtype=np.int32)
                contrasts = grp.create_dataset("contrast", shape=(n_samples,), dtype=np.float32)

                # BC channels
                bc_grp = grp.create_group("boundary_conditions")
                l_type_ds = bc_grp.create_dataset(
                    "left_type", shape=(n_samples,), dtype=h5py.string_dtype()
                )
                r_type_ds = bc_grp.create_dataset(
                    "right_type", shape=(n_samples,), dtype=h5py.string_dtype()
                )
                b_type_ds = bc_grp.create_dataset(
                    "bottom_type", shape=(n_samples,), dtype=h5py.string_dtype()
                )
                t_type_ds = bc_grp.create_dataset(
                    "top_type", shape=(n_samples,), dtype=h5py.string_dtype()
                )

                l_val_ds = bc_grp.create_dataset(
                    "left_value", shape=(n_samples, res), dtype=np.float32
                )
                r_val_ds = bc_grp.create_dataset(
                    "right_value", shape=(n_samples, res), dtype=np.float32
                )
                b_val_ds = bc_grp.create_dataset(
                    "bottom_value", shape=(n_samples, res), dtype=np.float32
                )
                t_val_ds = bc_grp.create_dataset(
                    "top_value", shape=(n_samples, res), dtype=np.float32
                )

                # Solve each sample
                for i, spec in enumerate(specs):
                    k_np, q_np, bcs = discretize_continuous_fields(
                        spec["k_spec"], spec["q_spec"], spec["bc_spec"], nx=res, ny=res
                    )
                    res_sol = solver.solve(grid, k_np, q_np, bcs)

                    k_ds[i] = k_np.astype(np.float32)
                    q_ds[i] = q_np.astype(np.float32)
                    t_ds[i] = res_sol.temperature.astype(np.float32)

                    sample_ids[i] = spec["sample_id"]
                    seeds[i] = spec["seed"]
                    subgroups[i] = spec["subgroup"]
                    subgroup_ids[i] = 1 if spec["subgroup"] == "short_correlation_grf" else 0

                    c_val = float(np.max(k_np) / max(np.min(k_np), 1e-12))
                    contrasts[i] = c_val

                    evals = bcs.evaluate_all(grid)
                    l_type_ds[i] = bcs.left.bc_type.value
                    r_type_ds[i] = bcs.right.bc_type.value
                    b_type_ds[i] = bcs.bottom.bc_type.value
                    t_type_ds[i] = bcs.top.bc_type.value

                    l_val_ds[i] = evals["left"].astype(np.float32)
                    r_val_ds[i] = evals["right"].astype(np.float32)
                    b_val_ds[i] = evals["bottom"].astype(np.float32)
                    t_val_ds[i] = evals["top"].astype(np.float32)

    elapsed = time.perf_counter() - t0
    sha256_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()
    file_size_bytes = output_path.stat().st_size
    logger.info(
        f"Confirmation Dataset created in {elapsed:.1f} s | Size: {file_size_bytes} bytes | SHA-256: {sha256_hash}"
    )

    # Build manifest
    manifest: dict[str, Any] = {
        "dataset_name": "touchstone_confirmation_dataset",
        "file_path": str(output_path),
        "file_size_bytes": file_size_bytes,
        "sha256": sha256_hash,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolutions": [32, 64],
        "splits": {
            "confirmation_iid": {
                "num_samples": num_iid,
                "seed_range": [base_seed_iid, base_seed_iid + num_iid - 1],
                "description": "200 unseen in-distribution smooth GRF conductivity fields",
            },
            "confirmation_ood": {
                "num_samples": num_ood,
                "seed_range": [base_seed_ood, base_seed_ood + num_ood - 1],
                "subgroups": {
                    "high_contrast_inclusions": hc_count,
                    "short_correlation_grf": grf_count,
                },
                "description": f"200 unseen out-of-distribution physical fields ({hc_count} High-Contrast Inclusions + {grf_count} Short-Correlation GRF)",
            },
        },
        "governance_status": "ESCROW_LOCKED",
        "inaccessible_for_training": True,
        "inaccessible_for_hyperparameters": True,
    }

    manifest_path = output_path.parent / "confirmation_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")

    return manifest


if __name__ == "__main__":
    out_file = Path("artifacts/data/confirmation_dataset.h5")
    build_confirmation_dataset(out_file, num_iid=200, num_ood=200)
