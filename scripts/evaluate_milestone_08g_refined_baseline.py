"""Milestone 08G Phase 2: Refined-Model Multi-Resolution Baseline.

Evaluates frozen models (A0 baseline M3, A1 loss-refined, A3 hybrid refined, A4 physics smoother)
under exact canonical normalization across:
- Resolutions: 32x32, 64x64, 128x128, 256x256
- Splits: test_iid, test_ood
- Sub-regions: Boundary (2 cells), Interior, Interface (sharp k jumps)
- Safety thresholds: % satisfying MaxAE <= 5 K, % satisfying MaxAE <= 10 K
- Diagnostics: Discrete heat balance / flux imbalance

Outputs: artifacts/reports/milestone_08g_refined_baseline.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from neural_process_twin.data.continuous_encoding import encode_continuous_physical_inputs
from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.models.refined_fno import RefinedFNO2D
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.physics_smoother import JacobiPhysicsSmoother

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_08g_baseline")


def compute_file_sha256(path: Path) -> str:
    if not path.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_bcs_row(grp: h5py.Group, idx: int) -> BoundaryConditions:
    bc_grp = grp["boundary_conditions"]
    l_type = bc_grp["left_type"][idx]
    r_type = bc_grp["right_type"][idx]
    b_type = bc_grp["bottom_type"][idx]
    t_type = bc_grp["top_type"][idx]

    l_val = bc_grp["left_value"][idx]
    r_val = bc_grp["right_value"][idx]
    b_val = bc_grp["bottom_value"][idx]
    t_val = bc_grp["top_value"][idx]

    def _to_str(val: Any) -> str:
        return val.decode("utf-8").lower() if isinstance(val, bytes) else str(val).lower()

    def _side_bc(side: BoundarySide, btype_val: Any, val_arr: np.ndarray) -> BoundaryCondition:
        t_str = _to_str(btype_val)
        b_type = BoundaryType.DIRICHLET if t_str == "dirichlet" else BoundaryType.NEUMANN
        if np.allclose(val_arr, val_arr[0]):
            return BoundaryCondition(side=side, bc_type=b_type, value=float(val_arr[0]))
        return BoundaryCondition(side=side, bc_type=b_type, value=val_arr.copy())

    return BoundaryConditions(
        left=_side_bc(BoundarySide.LEFT, l_type, l_val),
        right=_side_bc(BoundarySide.RIGHT, r_type, r_val),
        bottom=_side_bc(BoundarySide.BOTTOM, b_type, b_val),
        top=_side_bc(BoundarySide.TOP, t_type, t_val),
    )


def load_all_candidates(device: torch.device) -> dict[str, Any]:
    m3_path = Path("artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt")
    a1_path = Path("artifacts/checkpoints/milestone_08f/A1_loss_refined_fno/best_model.pt")
    a3_path = Path("artifacts/checkpoints/milestone_08f/A3_hybrid_refined_fno/best_model.pt")

    # M3
    m3_ckpt = torch.load(m3_path, map_location=device, weights_only=False)
    normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])
    m3_model = FourierNeuralOperator2D(
        in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4, activation="gelu"
    ).to(device)
    m3_model.load_state_dict(m3_ckpt.get("model_state_dict", m3_ckpt))
    m3_model.eval()

    # A1
    a1_ckpt = torch.load(a1_path, map_location=device, weights_only=False)
    a1_model = FourierNeuralOperator2D(
        in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4, activation="gelu"
    ).to(device)
    a1_model.load_state_dict(a1_ckpt.get("model_state_dict", a1_ckpt))
    a1_model.eval()

    # A3
    a3_ckpt = torch.load(a3_path, map_location=device, weights_only=False)
    a3_model = RefinedFNO2D(
        in_channels=12,
        out_channels=1,
        modes_x=12,
        modes_y=12,
        width=32,
        depth=4,
        refinement_hidden=32,
        refinement_layers=3,
        activation="gelu",
    ).to(device)
    a3_model.load_state_dict(a3_ckpt.get("model_state_dict", a3_ckpt))
    a3_model.eval()

    # Shared fitted log_k stats
    log_k_mean = float(m3_ckpt["log_k_mean"])
    log_k_std = float(m3_ckpt["log_k_std"])

    return {
        "A0_baseline_m3": {"model": m3_model, "type": "fno", "path": str(m3_path)},
        "A1_loss_refined_fno": {"model": a1_model, "type": "fno", "path": str(a1_path)},
        "A3_hybrid_refined_fno": {"model": a3_model, "type": "hybrid", "path": str(a3_path)},
        "A4_physics_smoother": {"model": a3_model, "type": "smoother", "path": str(a3_path)},
        "normalizer": normalizer,
        "log_k_mean": log_k_mean,
        "log_k_std": log_k_std,
    }


def evaluate_candidate_split(
    model_name: str,
    candidate_info: dict[str, Any],
    normalizer: FieldNormalizer,
    log_k_mean: float,
    log_k_std: float,
    h5_path: Path,
    resolution: int,
    split: str,
    device: torch.device,
    smoother: JacobiPhysicsSmoother,
) -> dict[str, Any]:
    model = candidate_info["model"]
    model_type = candidate_info["type"]
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, resolution, resolution, thickness=1.0)

    with h5py.File(h5_path, "r") as f:
        grp = f[f"res_{resolution}/{split}"]
        n_samples = grp["temperature"].shape[0]
        k_all = grp["k"][:]
        q_all = grp["q"][:]
        t_all = grp["temperature"][:]
        bcs_all = [parse_bcs_row(grp, i) for i in range(n_samples)]

    rmses: list[float] = []
    maxaes: list[float] = []
    rel_l2s: list[float] = []
    bnd_maxaes: list[float] = []
    int_maxaes: list[float] = []
    iface_maxaes: list[float] = []
    flux_imbalances: list[float] = []
    rel_flux_imbalances: list[float] = []
    latencies_ms: list[float] = []

    # Boundary ring (2 cells)
    rw = min(2, resolution // 2)
    bnd_mask = np.zeros((resolution, resolution), dtype=bool)
    bnd_mask[:rw, :] = True
    bnd_mask[-rw:, :] = True
    bnd_mask[:, :rw] = True
    bnd_mask[:, -rw:] = True
    int_mask = ~bnd_mask

    for i in range(n_samples):
        k_field = k_all[i]
        q_field = q_all[i]
        t_true = t_all[i]
        bcs = bcs_all[i]

        # Interface mask: where |grad(log(k))| is elevated
        log_k = np.log(np.clip(k_field, 1e-4, None))
        grad_y, grad_x = np.gradient(log_k)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        iface_mask = grad_mag > np.percentile(grad_mag, 85)

        x_in = encode_continuous_physical_inputs(
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            grid=grid,
            normalizer=normalizer,
            use_log_k=True,
            log_k_mean=log_k_mean,
            log_k_std=log_k_std,
        )
        x_tensor = torch.from_numpy(x_in).unsqueeze(0).to(device)

        t_start = time.perf_counter()
        with torch.no_grad():
            out_norm = model(x_tensor)
            t_pred = normalizer.inverse_transform_temperature(out_norm.cpu().numpy()[0, 0])

            if model_type == "smoother":
                t_pred = smoother.smooth(
                    t_init=t_pred,
                    grid=grid,
                    k_field=k_field,
                    q_field=q_field,
                    bcs=bcs,
                    num_iters=2,
                )
        t_end = time.perf_counter()
        latencies_ms.append((t_end - t_start) * 1000.0)

        err_map = np.abs(t_pred - t_true)
        rmse = float(np.sqrt(np.mean((t_pred - t_true) ** 2)))
        maxae = float(np.max(err_map))
        rel_l2 = float(np.linalg.norm(t_pred - t_true) / np.linalg.norm(t_true))

        bnd_maxae = float(np.max(err_map[bnd_mask]))
        int_maxae = float(np.max(err_map[int_mask])) if np.any(int_mask) else bnd_maxae
        iface_maxae = float(np.max(err_map[iface_mask])) if np.any(iface_mask) else 0.0

        diag = compute_heat_balance(grid, t_pred, k_field, q_field, bcs)

        rmses.append(rmse)
        maxaes.append(maxae)
        rel_l2s.append(rel_l2)
        bnd_maxaes.append(bnd_maxae)
        int_maxaes.append(int_maxae)
        iface_maxaes.append(iface_maxae)
        flux_imbalances.append(float(diag.flux_imbalance))
        rel_flux_imbalances.append(float(diag.relative_imbalance))

    return {
        "n_samples": n_samples,
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "maxae_mean": float(np.mean(maxaes)),
        "maxae_p90": float(np.percentile(maxaes, 90)),
        "maxae_max": float(np.max(maxaes)),
        "rel_l2_mean": float(np.mean(rel_l2s)),
        "pct_satisfying_5k": float(np.mean([m <= 5.0 for m in maxaes]) * 100.0),
        "pct_satisfying_10k": float(np.mean([m <= 10.0 for m in maxaes]) * 100.0),
        "bnd_maxae_mean": float(np.mean(bnd_maxaes)),
        "int_maxae_mean": float(np.mean(int_maxaes)),
        "iface_maxae_mean": float(np.mean(iface_maxaes)),
        "flux_imbalance_mean_w": float(np.mean(np.abs(flux_imbalances))),
        "rel_flux_imbalance_mean": float(np.mean(np.abs(rel_flux_imbalances))),
        "latency_mean_ms": float(np.mean(latencies_ms)),
        "latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "latency_p95_ms": float(np.percentile(latencies_ms, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 08G Refined Baseline Evaluation")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/multires_dataset.h5"),
        help="Path to multires dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/milestone_08g_refined_baseline.json"),
        help="Path to output baseline report",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    logger.info(f"Using compute device: {device}")

    candidates = load_all_candidates(device)
    normalizer = candidates["normalizer"]
    log_k_mean = candidates["log_k_mean"]
    log_k_std = candidates["log_k_std"]
    smoother = JacobiPhysicsSmoother(omega=0.67, max_iters=2)

    resolutions = [32, 64, 128, 256]
    splits = ["test_iid", "test_ood"]

    results: dict[str, Any] = {
        "milestone": "08G",
        "phase": 2,
        "title": "Milestone 08G Refined Surrogate Baseline Evaluation",
        "normalization": {
            "use_log_k": True,
            "log_k_mean": log_k_mean,
            "log_k_std": log_k_std,
            "source": "Fitted from M3 checkpoint metadata (canonical)",
        },
        "models": {},
    }

    model_keys = [
        "A0_baseline_m3",
        "A1_loss_refined_fno",
        "A3_hybrid_refined_fno",
        "A4_physics_smoother",
    ]

    for model_name in model_keys:
        logger.info(f"--- Evaluating {model_name} ---")
        results["models"][model_name] = {
            "path": candidates[model_name]["path"],
            "sha256": compute_file_sha256(Path(candidates[model_name]["path"])),
            "type": candidates[model_name]["type"],
            "evaluations": {},
        }

        for res in resolutions:
            results["models"][model_name]["evaluations"][f"{res}x{res}"] = {}
            for split in splits:
                logger.info(f"Evaluating {model_name} on {res}x{res} {split}...")
                metrics = evaluate_candidate_split(
                    model_name=model_name,
                    candidate_info=candidates[model_name],
                    normalizer=normalizer,
                    log_k_mean=log_k_mean,
                    log_k_std=log_k_std,
                    h5_path=args.dataset,
                    resolution=res,
                    split=split,
                    device=device,
                    smoother=smoother,
                )
                results["models"][model_name]["evaluations"][f"{res}x{res}"][split] = metrics
                logger.info(
                    f"{model_name:22s} | {res:3d}x{res:3d} {split:8s} | "
                    f"RMSE: {metrics['rmse_mean']:.3f} K | MaxAE: {metrics['maxae_mean']:.3f} K | "
                    f"<=5K: {metrics['pct_satisfying_5k']:.1f}% | <=10K: {metrics['pct_satisfying_10k']:.1f}% | "
                    f"BndMaxAE: {metrics['bnd_maxae_mean']:.2f} K | Latency p50: {metrics['latency_p50_ms']:.2f} ms"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Baseline results saved to {args.output}")


if __name__ == "__main__":
    main()
