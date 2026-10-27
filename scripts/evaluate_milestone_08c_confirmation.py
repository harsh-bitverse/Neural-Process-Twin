#!/usr/bin/env python3
"""Touchstone Independent Confirmation Evaluation.

Milestone 08C Phase 7: Independent Confirmation on Escrow-Locked Dataset.
Evaluates the frozen winning candidate model and historical M0 baseline against
the untouched touchstone confirmation dataset (confirmation_dataset.h5).

Strict Verification Invariants:
- Verifies SHA-256 hash of confirmation_dataset.h5 matches escrow record exactly.
- Strictly segregates confirmation_ood into High-Contrast (>= 20.0) and GRF (< 20.0).
- Reports Kelvin RMSE (overall, interior, boundary), MaxAE, Rel L2, Flux Imbalance (signed, abs),
  and PDE Residual L2 across 32x32 and 64x64.
- Persists machine-readable confirmation report to artifacts/reports/milestone_08c_confirmation_results.json.
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

from neural_process_twin.data.boundary_encoding import encode_boundary_conditions
from neural_process_twin.data.continuous_encoding import encode_continuous_physical_inputs
from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.training.physics_loss import DiscretePDEResidual

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_confirmation_08c")

EXPECTED_CONFIRMATION_SHA256 = "ca615ab6c4cd27436f66551728bb5e9aca24d76149782b68335523678deadf20"


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not path.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_distribution_stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a 1D scalar list."""
    a = np.asarray(values, dtype=np.float64)
    if len(a) == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "p95": float(np.percentile(a, 95)),
    }


def load_model(
    model_id: str,
    ckpt_path: Path,
    device: torch.device,
) -> tuple[FourierNeuralOperator2D, FieldNormalizer, bool, float, float]:
    """Load model architecture, weights, and normalizer metadata."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = FourierNeuralOperator2D(
        in_channels=12,
        out_channels=1,
        modes_x=12,
        modes_y=12,
        width=32,
        depth=4,
        activation="gelu",
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    use_log_k = ckpt.get("use_log_k", False)
    log_k_mean = float(ckpt.get("log_k_mean", 0.0))
    log_k_std = float(ckpt.get("log_k_std", 1.0))

    if "normalizer" in ckpt:
        normalizer = FieldNormalizer.from_dict(ckpt["normalizer"])
    elif "normalizer_stats" in ckpt:
        normalizer = FieldNormalizer.from_dict(ckpt["normalizer_stats"])
    else:
        norm_json = Path("artifacts/data/normalization_stats.json")
        with open(norm_json, encoding="utf-8") as f:
            normalizer = FieldNormalizer.from_dict(json.load(f))

    return model, normalizer, use_log_k, log_k_mean, log_k_std


def evaluate_confirmation_sample(
    model: FourierNeuralOperator2D,
    model_id: str,
    use_log_k: bool,
    log_k_mean: float,
    log_k_std: float,
    normalizer: FieldNormalizer,
    k_field: np.ndarray,
    q_field: np.ndarray,
    t_ref: np.ndarray,
    bcs: BoundaryConditions,
    grid: Grid2D,
    pde_op: DiscretePDEResidual,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate single touchstone sample."""
    ny, nx = grid.ny, grid.nx

    if model_id == "M0_fno_baseline":
        k_norm = normalizer.transform_k(k_field)
        q_norm = normalizer.transform_q(q_field)
        x_norm = (grid.x_mesh - grid.x_min) / grid.lx
        y_norm = (grid.y_mesh - grid.y_min) / grid.ly
        bc_channels = encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer)

        x_in = np.zeros((12, ny, nx), dtype=np.float32)
        x_in[0] = k_norm.astype(np.float32)
        x_in[1] = q_norm.astype(np.float32)
        x_in[2] = x_norm.astype(np.float32)
        x_in[3] = y_norm.astype(np.float32)
        x_in[4:12] = bc_channels
    else:
        x_in = encode_continuous_physical_inputs(
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            grid=grid,
            normalizer=normalizer,
            use_log_k=use_log_k,
            log_k_mean=log_k_mean,
            log_k_std=log_k_std,
        )

    t0 = time.perf_counter()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_in).unsqueeze(0).to(device)
        y_norm_pred = model(x_tensor).squeeze().cpu().numpy()
    inference_ms = (time.perf_counter() - t0) * 1000.0

    pred_t = normalizer.inverse_transform_temperature(y_norm_pred)

    diff = pred_t - t_ref
    overall_rmse = float(np.sqrt(np.mean(diff**2)))
    interior_rmse = float(np.sqrt(np.mean(diff[1:-1, 1:-1] ** 2)))

    bnd_mask = np.ones((ny, nx), dtype=bool)
    bnd_mask[1:-1, 1:-1] = False
    boundary_rmse = float(np.sqrt(np.mean(diff[bnd_mask] ** 2)))

    max_ae = float(np.max(np.abs(diff)))
    rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(t_ref) * 100.0)

    diag = compute_heat_balance(grid, pred_t, k_field, q_field, bcs)
    q_gen = float(diag.q_generated_total)
    q_out = float(diag.q_boundary_out_total)
    imbalance_signed = q_out - q_gen
    imbalance_abs = abs(imbalance_signed)
    rel_imbalance = float(diag.relative_imbalance * 100.0)

    pred_t_torch = torch.from_numpy(pred_t).float().unsqueeze(0).unsqueeze(0).to(device)
    k_torch = torch.from_numpy(k_field).float().unsqueeze(0).unsqueeze(0).to(device)
    q_torch = torch.from_numpy(q_field).float().unsqueeze(0).unsqueeze(0).to(device)
    bc_8ch = encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer)
    bc_torch = torch.from_numpy(bc_8ch).float().unsqueeze(0).to(device)

    with torch.no_grad():
        r_tensor, _ = pde_op(
            temperature_k=pred_t_torch,
            k_field=k_torch,
            q_field=q_torch,
            bc_channels=bc_torch,
            t_mean=float(normalizer.t_stats.mean),
            t_std=float(normalizer.t_stats.std),
        )
        pde_res_l2 = float(torch.norm(r_tensor).item())

    return {
        "overall_rmse_k": overall_rmse,
        "interior_rmse_k": interior_rmse,
        "boundary_rmse_k": boundary_rmse,
        "max_ae_k": max_ae,
        "relative_l2_pct": rel_l2,
        "q_generated_w": q_gen,
        "q_boundary_out_w": q_out,
        "flux_imbalance_abs_w": imbalance_abs,
        "flux_imbalance_signed_w": imbalance_signed,
        "relative_imbalance_pct": rel_imbalance,
        "pde_residual_l2_w": pde_res_l2,
        "inference_ms": inference_ms,
    }


def evaluate_confirmation_partition(
    model: FourierNeuralOperator2D,
    model_id: str,
    use_log_k: bool,
    log_k_mean: float,
    log_k_std: float,
    normalizer: FieldNormalizer,
    h5_path: Path,
    res: int,
    split_key: str,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate candidate on confirmation partition."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
    pde_op = DiscretePDEResidual(grid=grid).to(device)

    with h5py.File(h5_path, "r") as f:
        grp = f[f"res_{res}/{split_key}"]
        k_all = grp["k"][:]
        q_all = grp["q"][:]
        t_all = grp["temperature"][:]
        bc_grp = grp["boundary_conditions"]
        l_type_all = bc_grp["left_type"][:]
        r_type_all = bc_grp["right_type"][:]
        b_type_all = bc_grp["bottom_type"][:]
        t_type_all = bc_grp["top_type"][:]
        l_val_all = grp["boundary_conditions/left_value"][:]
        r_val_all = grp["boundary_conditions/right_value"][:]
        b_val_all = grp["boundary_conditions/bottom_value"][:]
        t_val_all = grp["boundary_conditions/top_value"][:]

    n_samples = k_all.shape[0]

    def to_str(val: Any) -> str:
        if isinstance(val, bytes):
            return val.decode("utf-8").lower()
        return str(val).lower()

    all_metrics: list[dict[str, float]] = []
    subgroups: list[str] = []

    for i in range(n_samples):
        k_field = k_all[i]
        q_field = q_all[i]
        t_ref = t_all[i]

        bcs = BoundaryConditions(
            left=BoundaryCondition(
                BoundarySide.LEFT, BoundaryType(to_str(l_type_all[i])), l_val_all[i]
            ),
            right=BoundaryCondition(
                BoundarySide.RIGHT, BoundaryType(to_str(r_type_all[i])), r_val_all[i]
            ),
            bottom=BoundaryCondition(
                BoundarySide.BOTTOM, BoundaryType(to_str(b_type_all[i])), b_val_all[i]
            ),
            top=BoundaryCondition(
                BoundarySide.TOP, BoundaryType(to_str(t_type_all[i])), t_val_all[i]
            ),
        )

        contrast = float(np.max(k_field) / np.min(k_field))
        subgroup = ("high_contrast" if contrast >= 20.0 else "grf") if "ood" in split_key else "iid"
        subgroups.append(subgroup)

        m = evaluate_confirmation_sample(
            model=model,
            model_id=model_id,
            use_log_k=use_log_k,
            log_k_mean=log_k_mean,
            log_k_std=log_k_std,
            normalizer=normalizer,
            k_field=k_field,
            q_field=q_field,
            t_ref=t_ref,
            bcs=bcs,
            grid=grid,
            pde_op=pde_op,
            device=device,
        )
        all_metrics.append(m)

    metric_keys = list(all_metrics[0].keys())
    combined_summary: dict[str, Any] = {
        "sample_count": n_samples,
    }
    for k in metric_keys:
        vals = [m[k] for m in all_metrics]
        combined_summary[k] = compute_distribution_stats(vals)

    res_dict: dict[str, Any] = {
        "combined": combined_summary,
    }

    if "ood" in split_key:
        for sub in ["high_contrast", "grf"]:
            sub_indices = [idx for idx, s in enumerate(subgroups) if s == sub]
            sub_summary: dict[str, Any] = {
                "sample_count": len(sub_indices),
            }
            if len(sub_indices) > 0:
                for k in metric_keys:
                    vals = [all_metrics[idx][k] for idx in sub_indices]
                    sub_summary[k] = compute_distribution_stats(vals)
            res_dict[sub] = sub_summary

    return res_dict


def main() -> int:
    parser = argparse.ArgumentParser(description="Touchstone Independent Confirmation")
    parser.add_argument(
        "--candidate-id",
        type=str,
        required=True,
        help="Selected candidate ID (e.g. M2_cont_fno_mixed or M3_logk_fno_mixed)",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    torch.set_num_threads(8)
    device = torch.device(args.device)

    conf_h5 = Path("artifacts/data/confirmation_dataset.h5")
    if not conf_h5.is_file():
        logger.error(f"Confirmation dataset not found: {conf_h5}")
        return 1

    actual_hash = compute_file_sha256(conf_h5)
    logger.info(f"Touchstone confirmation file: {conf_h5} | SHA-256: {actual_hash}")
    if actual_hash != EXPECTED_CONFIRMATION_SHA256:
        logger.error(
            f"Integrity violation! Expected {EXPECTED_CONFIRMATION_SHA256}, got {actual_hash}"
        )
        return 1
    logger.info("Touchstone confirmation file integrity verified against escrow lock record.")

    models_to_evaluate = [
        {
            "model_id": "M0_fno_baseline",
            "ckpt_path": Path("artifacts/checkpoints/scaled_fno_supervised_seed42/best_model.pt"),
            "display_name": "M0: Historical Dirac FNO (32x32)",
        },
        {
            "model_id": args.candidate_id,
            "ckpt_path": Path(
                f"artifacts/checkpoints/milestone_08c/{args.candidate_id}/best_model.pt"
            ),
            "display_name": f"Selected Frozen Candidate ({args.candidate_id})",
        },
    ]

    resolutions = [32, 64]
    splits = [
        ("confirmation_iid", "confirmation_iid"),
        ("confirmation_ood", "confirmation_ood"),
    ]

    results: dict[str, Any] = {
        "metadata": {
            "dataset": str(conf_h5),
            "dataset_sha256": actual_hash,
            "escrow_verified": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selected_candidate": args.candidate_id,
            "resolutions": resolutions,
        },
        "candidates": {},
    }

    for cand in models_to_evaluate:
        model_id = cand["model_id"]
        ckpt_path = cand["ckpt_path"]
        logger.info(f"=== Evaluating Touchstone Confirmation for {model_id} ===")

        model, normalizer, use_log_k, log_k_mean, log_k_std = load_model(
            model_id=model_id,
            ckpt_path=ckpt_path,
            device=device,
        )

        cand_results: dict[str, Any] = {
            "display_name": cand["display_name"],
            "checkpoint": str(ckpt_path),
            "checkpoint_sha256": compute_file_sha256(ckpt_path),
            "use_log_k": use_log_k,
            "evaluations": {},
        }

        for res in resolutions:
            cand_results["evaluations"][f"res_{res}"] = {}
            for split_name, split_key in splits:
                logger.info(f"Confirmation evaluate {model_id} on res_{res} / {split_name}...")
                part_res = evaluate_confirmation_partition(
                    model=model,
                    model_id=model_id,
                    use_log_k=use_log_k,
                    log_k_mean=log_k_mean,
                    log_k_std=log_k_std,
                    normalizer=normalizer,
                    h5_path=conf_h5,
                    res=res,
                    split_key=split_key,
                    device=device,
                )
                cand_results["evaluations"][f"res_{res}"][split_name] = part_res

                overall_rmse = part_res["combined"]["overall_rmse_k"]["mean"]
                rel_l2 = part_res["combined"]["relative_l2_pct"]["mean"]
                imb = part_res["combined"]["flux_imbalance_abs_w"]["mean"]
                logger.info(
                    f"-> [CONFIRMATION: {model_id} | res_{res} | {split_name}] "
                    f"RMSE: {overall_rmse:.3f} K | Rel-L2: {rel_l2:.2f}% | Flux Imb: {imb:.2f} W"
                )

        results["candidates"][model_id] = cand_results

    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_json = reports_dir / "milestone_08c_confirmation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Touchstone confirmation complete. Saved to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
