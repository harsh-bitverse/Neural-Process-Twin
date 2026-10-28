"""Independent Confirmation Evaluation on Locked 08F Cohort (Phase 9).

Evaluates candidates A0, A1, A2, A3, and A4 on the pristine, previously unseen
independent confirmation cohort:
- artifacts/data/independent_cohort_08f.h5 (seeds 900,000-900,049 IID, 910,000-910,049 OOD)
- Evaluated across: 32x32, 64x64, 128x128, 256x256
- Outputs: artifacts/reports/milestone_08f_independent_confirmation.json
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
logger = logging.getLogger("eval_08f_independent")


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


def evaluate_cohort(
    cohort_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Execute independent confirmation evaluation across all models on sealed cohort."""
    logger.info(f"Loading independent confirmation cohort: {cohort_path}")
    cohort_sha = compute_file_sha256(cohort_path)

    torch.set_num_threads(8)
    device = torch.device("cpu")

    m3_path = Path("artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt")
    a1_path = Path("artifacts/checkpoints/milestone_08f/A1_loss_refined_fno/best_model.pt")
    a3_path = Path("artifacts/checkpoints/milestone_08f/A3_hybrid_refined_fno/best_model.pt")

    # Load checkpoints
    m3_ckpt = torch.load(m3_path, map_location=device, weights_only=False)
    normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])

    m3_model = FourierNeuralOperator2D(
        in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4, activation="gelu"
    ).to(device)
    m3_model.load_state_dict(m3_ckpt.get("model_state_dict", m3_ckpt))
    m3_model.eval()

    a1_model = FourierNeuralOperator2D(
        in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4, activation="gelu"
    ).to(device)
    a1_ckpt = torch.load(a1_path, map_location=device, weights_only=False)
    a1_model.load_state_dict(a1_ckpt.get("model_state_dict", a1_ckpt))
    a1_model.eval()

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
    a3_ckpt = torch.load(a3_path, map_location=device, weights_only=False)
    a3_model.load_state_dict(a3_ckpt.get("model_state_dict", a3_ckpt))
    a3_model.eval()

    models = {
        "A0_baseline_m3": {"model": m3_model, "type": "fno"},
        "A1_loss_refined_fno": {"model": a1_model, "type": "fno"},
        "A2_boundary_corrected": {"model": a1_model, "type": "bnd_corr"},
        "A3_hybrid_refined_fno": {"model": a3_model, "type": "hybrid"},
        "A4_physics_smoother": {"model": a3_model, "type": "smoother"},
    }

    resolutions = [32, 64, 128, 256]
    splits = ["test_iid", "test_ood"]
    smoother = JacobiPhysicsSmoother(omega=0.67, max_iters=2)

    results: dict[str, Any] = {
        "cohort_path": str(cohort_path),
        "cohort_sha256": cohort_sha,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resolutions": resolutions,
        "splits": splits,
        "candidates": {},
    }

    for cid, minfo in models.items():
        logger.info(f"=== Evaluating Candidate on Locked Holdout: {cid} ===")
        results["candidates"][cid] = {"evaluations": {}}
        model = minfo["model"]
        mtype = minfo["type"]

        for res in resolutions:
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
            bnd_corr = DiscreteBoundaryCorrection(grid=grid, correction_mode="hard")

            rw = min(2, res // 2)
            bnd_mask = np.zeros((res, res), dtype=bool)
            bnd_mask[:rw, :] = True
            bnd_mask[-rw:, :] = True
            bnd_mask[:, :rw] = True
            bnd_mask[:, -rw:] = True
            int_mask = ~bnd_mask

            for split in splits:
                eval_key = f"res_{res}/{split}"
                with h5py.File(cohort_path, "r") as f:
                    grp = f[eval_key]
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
                flux_imb: list[float] = []
                rel_flux_imb: list[float] = []

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

                    with torch.no_grad():
                        out_norm = model(x_tensor)
                        t_pred = normalizer.inverse_transform_temperature(
                            out_norm.cpu().numpy()[0, 0]
                        )

                        if mtype == "bnd_corr":
                            t_pred_t = torch.from_numpy(t_pred).unsqueeze(0).unsqueeze(0).float()
                            k_t = torch.from_numpy(k_field).unsqueeze(0).unsqueeze(0).float()
                            bc_ch = x_tensor[:, 4:12]
                            t_corr_t = bnd_corr._apply_fvm_balance(
                                temperature_k=t_pred_t,
                                k_field=k_t,
                                bc_channels=bc_ch,
                                q_field=torch.from_numpy(q_field).unsqueeze(0).unsqueeze(0).float(),
                            )
                            t_pred = t_corr_t.numpy()[0, 0]

                        elif mtype == "smoother":
                            t_pred = smoother.smooth(
                                t_init=t_pred,
                                grid=grid,
                                k_field=k_field,
                                q_field=q_field,
                                bcs=bcs,
                                num_iters=2,
                            )

                    err_map = np.abs(t_pred - t_true)
                    maxae = float(np.max(err_map))
                    rmse = float(np.sqrt(np.mean(err_map**2)))
                    rel_l2 = (
                        float(np.linalg.norm(err_map) / (np.linalg.norm(t_true) + 1e-8)) * 100.0
                    )

                    bnd_maxae = float(np.max(err_map[bnd_mask]))
                    int_maxae = float(np.max(err_map[int_mask])) if np.any(int_mask) else bnd_maxae

                    hb = compute_heat_balance(grid, t_pred, k_field, q_field, bcs)

                    rmses.append(rmse)
                    maxaes.append(maxae)
                    rel_l2s.append(rel_l2)
                    bnd_maxaes.append(bnd_maxae)
                    int_maxaes.append(int_maxae)
                    flux_imb.append(hb.flux_imbalance)
                    rel_flux_imb.append(hb.relative_imbalance * 100.0)

                eval_summary = {
                    "n_samples": n_samples,
                    "mean_rmse_k": float(np.mean(rmses)),
                    "std_rmse_k": float(np.std(rmses, ddof=1)) if n_samples > 1 else 0.0,
                    "mean_rel_l2_pct": float(np.mean(rel_l2s)),
                    "median_maxae_k": float(np.median(maxaes)),
                    "p95_maxae_k": float(np.percentile(maxaes, 95)),
                    "max_maxae_k": float(np.max(maxaes)),
                    "pct_under_5k": float(np.mean([m <= 5.0 for m in maxaes]) * 100.0),
                    "pct_under_10k": float(np.mean([m <= 10.0 for m in maxaes]) * 100.0),
                    "mean_bnd_maxae_k": float(np.mean(bnd_maxaes)),
                    "mean_int_maxae_k": float(np.mean(int_maxaes)),
                    "median_flux_imbalance_w": float(np.median(flux_imb)),
                    "mean_rel_flux_imbalance_pct": float(np.mean(rel_flux_imb)),
                }
                results["candidates"][cid]["evaluations"][eval_key] = eval_summary
                logger.info(
                    f"[{cid}] Holdout {eval_key} -> RMSE: {eval_summary['mean_rmse_k']:.2f} K | "
                    f"Median MaxAE: {eval_summary['median_maxae_k']:.2f} K | "
                    f"MaxAE: {eval_summary['max_maxae_k']:.2f} K | "
                    f"<=10K: {eval_summary['pct_under_10k']:.1f}%"
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Holdout confirmation complete. Results persisted to {output_path}")
    return results


def main() -> int:
    """CLI for independent cohort evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate 08F models on independent cohort")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("artifacts/data/independent_cohort_08f.h5"),
        help="Path to independent cohort HDF5 file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/milestone_08f_independent_confirmation.json"),
        help="Path to output JSON results",
    )
    args = parser.parse_args()

    evaluate_cohort(args.cohort, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
