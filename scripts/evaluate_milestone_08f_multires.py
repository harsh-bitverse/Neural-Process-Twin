"""Comprehensive Multi-Resolution Evaluation for Milestone 08F Candidates.

Evaluates candidates A0, A1, A2, A3, and A4 across:
- Resolutions: 32x32, 64x64, 128x128, 256x256
- Splits: test_iid, test_ood
- Metrics: RMSE, MaxAE, % satisfying <=5K and <=10K, Boundary MaxAE,
  Interior MaxAE, Interface MaxAE, Flux Imbalance, and Inference Latency.
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
from neural_process_twin.models.boundary_aware_fno import DiscreteBoundaryCorrection
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
logger = logging.getLogger("evaluate_milestone_08f")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    if not path.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_bcs_row(grp: h5py.Group, idx: int) -> BoundaryConditions:
    """Parse BoundaryConditions from an HDF5 group row."""
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


def load_candidate_models(device: torch.device) -> dict[str, Any]:
    """Load all candidate models and wrappers."""
    m3_path = Path("artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt")
    a1_path = Path("artifacts/checkpoints/milestone_08f/A1_loss_refined_fno/best_model.pt")
    a3_path = Path("artifacts/checkpoints/milestone_08f/A3_hybrid_refined_fno/best_model.pt")

    # 1. A0: Baseline M3
    m3_ckpt = torch.load(m3_path, map_location=device, weights_only=False)
    normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])
    m3_model = FourierNeuralOperator2D(
        in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4, activation="gelu"
    ).to(device)
    m3_model.load_state_dict(m3_ckpt.get("model_state_dict", m3_ckpt))
    m3_model.eval()

    # 2. A1: Loss-refined FNO
    a1_model = FourierNeuralOperator2D(
        in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4, activation="gelu"
    ).to(device)
    if a1_path.is_file():
        a1_ckpt = torch.load(a1_path, map_location=device, weights_only=False)
        a1_model.load_state_dict(a1_ckpt.get("model_state_dict", a1_ckpt))
    else:
        logger.warning(f"A1 checkpoint not found at {a1_path}. Using initial weights.")
    a1_model.eval()

    # 3. A3: Hybrid Refined FNO
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
    if a3_path.is_file():
        a3_ckpt = torch.load(a3_path, map_location=device, weights_only=False)
        a3_model.load_state_dict(a3_ckpt.get("model_state_dict", a3_ckpt))
    else:
        logger.warning(f"A3 checkpoint not found at {a3_path}. Using initial weights.")
    a3_model.eval()

    return {
        "A0_baseline_m3": {"model": m3_model, "type": "fno", "path": str(m3_path)},
        "A1_loss_refined_fno": {"model": a1_model, "type": "fno", "path": str(a1_path)},
        "A2_boundary_corrected": {"model": a1_model, "type": "bnd_corr", "path": str(a1_path)},
        "A3_hybrid_refined_fno": {"model": a3_model, "type": "hybrid", "path": str(a3_path)},
        "A4_physics_smoother": {"model": a3_model, "type": "smoother", "path": str(a3_path)},
        "normalizer": normalizer,
    }


def evaluate_split(
    model_id: str,
    model_info: dict[str, Any],
    normalizer: FieldNormalizer,
    h5_path: Path,
    resolution: int,
    split: str,
    device: torch.device,
    smoother: JacobiPhysicsSmoother,
) -> dict[str, Any]:
    """Evaluate a single candidate on a single resolution and split."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, resolution, resolution, thickness=1.0)
    model = model_info["model"]
    model_type = model_info["type"]

    bnd_correction = DiscreteBoundaryCorrection(grid=grid, correction_mode="hard")

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

    # Boundary ring mask (2 cells)
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

        x_in = encode_continuous_physical_inputs(
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            grid=grid,
            normalizer=normalizer,
            use_log_k=True,
            log_k_mean=0.5,
            log_k_std=1.0,
        )
        x_tensor = torch.from_numpy(x_in).unsqueeze(0).to(device)

        # Inference timing
        t_start = time.perf_counter()
        with torch.no_grad():
            out_norm = model(x_tensor)
            t_pred = normalizer.inverse_transform_temperature(out_norm.cpu().numpy()[0, 0])

            if model_type == "bnd_corr":
                # Apply discrete boundary correction
                t_pred_t = torch.from_numpy(t_pred).unsqueeze(0).unsqueeze(0).float()
                k_t = torch.from_numpy(k_field).unsqueeze(0).unsqueeze(0).float()
                # bc_channels: use x_tensor[:, 4:12]
                bc_ch = x_tensor[:, 4:12]
                t_corr_t = bnd_correction._apply_fvm_balance(
                    temperature_k=t_pred_t,
                    k_field=k_t,
                    bc_channels=bc_ch,
                    q_field=torch.from_numpy(q_field).unsqueeze(0).unsqueeze(0).float(),
                )
                t_pred = t_corr_t.numpy()[0, 0]

            elif model_type == "smoother":
                # Apply 2 iterations of Jacobi physics smoothing
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
        maxae = float(np.max(err_map))
        rmse = float(np.sqrt(np.mean(err_map**2)))
        rel_l2 = float(np.linalg.norm(err_map) / (np.linalg.norm(t_true) + 1e-8)) * 100.0

        bnd_maxae = float(np.max(err_map[bnd_mask]))
        int_maxae = float(np.max(err_map[int_mask])) if np.any(int_mask) else bnd_maxae

        # Interface error: top 20% gradient cells
        grad_ky, grad_kx = np.gradient(k_field, grid.dy, grid.dx)
        grad_mag = np.sqrt(grad_kx**2 + grad_ky**2)
        iface_thresh = np.percentile(grad_mag, 80)
        iface_mask = grad_mag >= iface_thresh
        iface_maxae = float(np.max(err_map[iface_mask])) if np.any(iface_mask) else maxae

        # Heat balance
        hb = compute_heat_balance(grid, t_pred, k_field, q_field, bcs)

        rmses.append(rmse)
        maxaes.append(maxae)
        rel_l2s.append(rel_l2)
        bnd_maxaes.append(bnd_maxae)
        int_maxaes.append(int_maxae)
        iface_maxaes.append(iface_maxae)
        flux_imbalances.append(hb.flux_imbalance)
        rel_flux_imbalances.append(hb.relative_imbalance * 100.0)

    # Discard first latency measurement as warm-up
    clean_latencies = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms

    return {
        "n_samples": n_samples,
        "mean_rmse_k": float(np.mean(rmses)),
        "std_rmse_k": float(np.std(rmses, ddof=1)) if n_samples > 1 else 0.0,
        "mean_rel_l2_pct": float(np.mean(rel_l2s)),
        "maxae_distribution": {
            "min": float(np.min(maxaes)),
            "p25": float(np.percentile(maxaes, 25)),
            "median": float(np.median(maxaes)),
            "p75": float(np.percentile(maxaes, 75)),
            "p90": float(np.percentile(maxaes, 90)),
            "p95": float(np.percentile(maxaes, 95)),
            "max": float(np.max(maxaes)),
        },
        "pct_under_5k": float(np.mean([m <= 5.0 for m in maxaes]) * 100.0),
        "pct_under_10k": float(np.mean([m <= 10.0 for m in maxaes]) * 100.0),
        "mean_bnd_maxae_k": float(np.mean(bnd_maxaes)),
        "mean_int_maxae_k": float(np.mean(int_maxaes)),
        "mean_iface_maxae_k": float(np.mean(iface_maxaes)),
        "median_flux_imbalance_w": float(np.median(flux_imbalances)),
        "mean_rel_flux_imbalance_pct": float(np.mean(rel_flux_imbalances)),
        "mean_latency_ms": float(np.mean(clean_latencies)),
        "p95_latency_ms": float(np.percentile(clean_latencies, 95)),
    }


def main() -> int:
    """Execute multi-resolution evaluation across all candidates."""
    parser = argparse.ArgumentParser(description="Evaluate Milestone 08F candidates")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/multires_dataset.h5"),
        help="Path to multiresolution HDF5 dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/milestone_08f_multires_evaluation.json"),
        help="Path to output JSON report",
    )
    args = parser.parse_args()

    torch.set_num_threads(8)
    device = torch.device("cpu")
    models_dict = load_candidate_models(device)
    normalizer: FieldNormalizer = models_dict["normalizer"]

    candidate_ids = [
        "A0_baseline_m3",
        "A1_loss_refined_fno",
        "A2_boundary_corrected",
        "A3_hybrid_refined_fno",
        "A4_physics_smoother",
    ]

    resolutions = [32, 64, 128, 256]
    splits = ["test_iid", "test_ood"]
    smoother = JacobiPhysicsSmoother(omega=0.67, max_iters=2)

    results: dict[str, Any] = {
        "dataset_path": str(args.dataset),
        "dataset_sha256": compute_file_sha256(args.dataset),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolutions": resolutions,
        "splits": splits,
        "candidates": {},
    }

    for cid in candidate_ids:
        logger.info(f"=== Evaluating Candidate: {cid} ===")
        minfo = models_dict[cid]
        results["candidates"][cid] = {
            "checkpoint_path": minfo["path"],
            "model_type": minfo["type"],
            "evaluations": {},
        }

        for res in resolutions:
            for split in splits:
                eval_key = f"res_{res}/{split}"
                logger.info(f"[{cid}] Evaluating {eval_key}...")
                eval_res = evaluate_split(
                    model_id=cid,
                    model_info=minfo,
                    normalizer=normalizer,
                    h5_path=args.dataset,
                    resolution=res,
                    split=split,
                    device=device,
                    smoother=smoother,
                )
                results["candidates"][cid]["evaluations"][eval_key] = eval_res
                logger.info(
                    f"[{cid}] {eval_key} -> RMSE: {eval_res['mean_rmse_k']:.2f} K | "
                    f"Median MaxAE: {eval_res['maxae_distribution']['median']:.2f} K | "
                    f"Max MaxAE: {eval_res['maxae_distribution']['max']:.2f} K | "
                    f"<=10K: {eval_res['pct_under_10k']:.1f}% | "
                    f"Latency: {eval_res['mean_latency_ms']:.2f} ms"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete. Authoritative report saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
