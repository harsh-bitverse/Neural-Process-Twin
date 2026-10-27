#!/usr/bin/env python3
"""Reliability Baseline Audit for Milestone 08D Phase 1.

Audits the current M3 neural operator architecture, normalization, and inference pipeline.
Establishes the reliability baseline across mesh resolutions (32x32 to 256x256) and physical shifts:
- Per-sample RMSE and MaxAE distributions.
- Absolute and relative discrete PDE residuals.
- Signed and absolute heat imbalance.
- Error breakdown by conductivity contrast, correlation length proxy, boundary condition regime, and resolution.
- Rank correlation analysis between candidate physical error signals and true temperature errors.

Outputs machine-readable JSON to artifacts/reports/milestone_08d_baseline_reliability.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.stats as stats
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
logger = logging.getLogger("audit_milestone_08d_baseline")


def to_str(val: Any) -> str:
    """Decode byte string or object to lowercase string."""
    if hasattr(val, "tolist"):
        val = val.tolist()
    if isinstance(val, bytes):
        return val.decode("utf-8").lower()
    if isinstance(val, list):
        return to_str(val[0])
    return str(val).lower()


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


def evaluate_partition_detailed(
    model: FourierNeuralOperator2D,
    normalizer: FieldNormalizer,
    use_log_k: bool,
    log_k_mean: float,
    log_k_std: float,
    h5_path: Path,
    res: int,
    split: str,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate M3 on a partition and compute all error and candidate uncertainty signals."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
    pde_op = DiscretePDEResidual(grid=grid).to(device)

    with h5py.File(h5_path, "r") as f:
        grp = f[f"res_{res}/{split}"]
        k_all = grp["k"][:]
        q_all = grp["q"][:]
        t_all = grp["temperature"][:]
        bc_grp = grp["boundary_conditions"]
        l_type = bc_grp["left_type"][:]
        r_type = bc_grp["right_type"][:]
        b_type = bc_grp["bottom_type"][:]
        t_type = bc_grp["top_type"][:]
        l_val = grp["boundary_conditions/left_value"][:]
        r_val = grp["boundary_conditions/right_value"][:]
        b_val = grp["boundary_conditions/bottom_value"][:]
        t_val = grp["boundary_conditions/top_value"][:]

    n_samples = k_all.shape[0]
    sample_metrics: list[dict[str, Any]] = []

    for i in range(n_samples):
        k_field = k_all[i]
        q_field = q_all[i]
        t_ref = t_all[i]

        b_type_str = to_str(b_type[i])
        t_type_str = to_str(t_type[i])
        both_neumann = (b_type_str == "neumann") and (t_type_str == "neumann")

        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType(to_str(l_type[i])), l_val[i]),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType(to_str(r_type[i])), r_val[i]),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType(b_type_str), b_val[i]),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType(t_type_str), t_val[i]),
        )

        contrast = float(np.max(k_field) / np.min(k_field))
        if split == "test_ood":
            subgroup = "high_contrast" if contrast >= 20.0 else "short_corr_grf"
        else:
            subgroup = "iid_smooth_grf"

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

        with torch.no_grad():
            x_t = torch.from_numpy(x_in).unsqueeze(0).to(device)
            y_pred = model(x_t).squeeze().cpu().numpy()

        pred_t = normalizer.inverse_transform_temperature(y_pred)
        diff = pred_t - t_ref
        overall_rmse = float(np.sqrt(np.mean(diff**2)))
        max_ae = float(np.max(np.abs(diff)))
        rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(t_ref) * 100.0)

        # Physical Diagnostics
        diag = compute_heat_balance(grid, pred_t, k_field, q_field, bcs)
        q_gen = float(diag.q_generated_total)
        q_out = float(diag.q_boundary_out_total)
        imbalance_signed = q_out - q_gen
        imbalance_abs = abs(imbalance_signed)
        rel_imbalance = float(diag.relative_imbalance * 100.0)

        # Discrete PDE Residual
        pred_t_t = torch.from_numpy(pred_t).float().unsqueeze(0).unsqueeze(0).to(device)
        k_t = torch.from_numpy(k_field).float().unsqueeze(0).unsqueeze(0).to(device)
        q_t = torch.from_numpy(q_field).float().unsqueeze(0).unsqueeze(0).to(device)
        bc_8ch = encode_boundary_conditions(bcs, grid=grid, normalizer=normalizer)
        bc_t = torch.from_numpy(bc_8ch).float().unsqueeze(0).to(device)

        with torch.no_grad():
            r_tensor, _ = pde_op(
                temperature_k=pred_t_t,
                k_field=k_t,
                q_field=q_t,
                bc_channels=bc_t,
                t_mean=float(normalizer.t_stats.mean),
                t_std=float(normalizer.t_stats.std),
            )
            pde_res_l2 = float(torch.norm(r_tensor).item())

        # Relative PDE residual: normalized by approximate source scale
        source_norm = float(np.linalg.norm(q_field) * (grid.dx * grid.dy))
        pde_res_rel = pde_res_l2 / source_norm if source_norm > 1e-6 else pde_res_l2

        sample_metrics.append(
            {
                "index": i,
                "subgroup": subgroup,
                "both_neumann": both_neumann,
                "contrast": contrast,
                "source_max": float(np.max(q_field)),
                "source_mean": float(np.mean(q_field)),
                "rmse_k": overall_rmse,
                "max_ae_k": max_ae,
                "rel_l2_pct": rel_l2,
                "q_generated_w": q_gen,
                "q_boundary_out_w": q_out,
                "flux_imbalance_abs_w": imbalance_abs,
                "flux_imbalance_signed_w": imbalance_signed,
                "relative_imbalance_pct": rel_imbalance,
                "pde_res_l2_w": pde_res_l2,
                "pde_res_rel": pde_res_rel,
            }
        )

    summary = {
        "num_samples": n_samples,
        "rmse_k": compute_distribution_stats([s["rmse_k"] for s in sample_metrics]),
        "max_ae_k": compute_distribution_stats([s["max_ae_k"] for s in sample_metrics]),
        "rel_l2_pct": compute_distribution_stats([s["rel_l2_pct"] for s in sample_metrics]),
        "flux_imbalance_abs_w": compute_distribution_stats(
            [s["flux_imbalance_abs_w"] for s in sample_metrics]
        ),
        "relative_imbalance_pct": compute_distribution_stats(
            [s["relative_imbalance_pct"] for s in sample_metrics]
        ),
        "pde_res_l2_w": compute_distribution_stats([s["pde_res_l2_w"] for s in sample_metrics]),
        "pde_res_rel": compute_distribution_stats([s["pde_res_rel"] for s in sample_metrics]),
    }

    return summary, sample_metrics


def run_baseline_reliability_audit() -> dict[str, Any]:
    """Execute complete Phase 1 reliability baseline audit."""
    logger.info("Executing Milestone 08D Phase 1: Baseline Reliability Audit...")

    ckpt_path = Path("artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m3 = FourierNeuralOperator2D(
        in_channels=12,
        out_channels=1,
        modes_x=12,
        modes_y=12,
        width=32,
        depth=4,
        activation="gelu",
    )
    m3.load_state_dict(ckpt["model_state_dict"])
    m3.eval()
    norm = FieldNormalizer.from_dict(ckpt["normalizer"])
    use_log_k = ckpt.get("use_log_k", False)
    log_k_mean = float(ckpt.get("log_k_mean", 0.0))
    log_k_std = float(ckpt.get("log_k_std", 1.0))

    h5_path = Path("artifacts/data/multires_dataset.h5")
    resolutions = [32, 64, 128, 256]

    results: dict[str, Any] = {
        "metadata": {
            "description": "M3 Reliability Baseline Audit (Milestone 08D Phase 1)",
            "checkpoint": str(ckpt_path),
            "resolutions": resolutions,
            "timestamp": "2026-10-27T16:50:00Z",
        },
        "per_resolution_reliability": {},
        "correlations": {},
        "error_breakdowns": {},
    }

    all_pooled_samples: list[dict[str, Any]] = []

    for res in resolutions:
        logger.info(f"Auditing M3 reliability at resolution {res}x{res}...")
        iid_summary, iid_samples = evaluate_partition_detailed(
            model=m3,
            normalizer=norm,
            use_log_k=use_log_k,
            log_k_mean=log_k_mean,
            log_k_std=log_k_std,
            h5_path=h5_path,
            res=res,
            split="test_iid",
            device=torch.device("cpu"),
        )
        ood_summary, ood_samples = evaluate_partition_detailed(
            model=m3,
            normalizer=norm,
            use_log_k=use_log_k,
            log_k_mean=log_k_mean,
            log_k_std=log_k_std,
            h5_path=h5_path,
            res=res,
            split="test_ood",
            device=torch.device("cpu"),
        )

        results["per_resolution_reliability"][f"res_{res}"] = {
            "test_iid": iid_summary,
            "test_ood": ood_summary,
        }

        # Tag with resolution for pooled correlation analysis
        for s in iid_samples:
            s["resolution"] = res
            s["split"] = "iid"
            all_pooled_samples.append(s)
        for s in ood_samples:
            s["resolution"] = res
            s["split"] = "ood"
            all_pooled_samples.append(s)

    # 1. Rank Correlation Analysis: Candidate Uncertainty Signals vs True Errors
    logger.info("Computing correlation matrices between candidate signals and true errors...")

    signals = [
        ("pde_res_l2_w", "Absolute PDE Residual L2"),
        ("pde_res_rel", "Relative PDE Residual"),
        ("flux_imbalance_abs_w", "Absolute Flux Imbalance"),
        ("relative_imbalance_pct", "Relative Flux Imbalance %"),
        ("contrast", "Conductivity Contrast Ratio"),
        ("source_max", "Peak Volumetric Heat Source"),
    ]

    for scope_name, samples_subset in [
        ("pooled_all", all_pooled_samples),
        ("iid_only", [s for s in all_pooled_samples if s["split"] == "iid"]),
        ("ood_only", [s for s in all_pooled_samples if s["split"] == "ood"]),
        ("res_32_all", [s for s in all_pooled_samples if s["resolution"] == 32]),
        ("res_256_all", [s for s in all_pooled_samples if s["resolution"] == 256]),
    ]:
        corr_dict: dict[str, Any] = {}
        rmse_arr = np.array([s["rmse_k"] for s in samples_subset], dtype=np.float64)
        maxae_arr = np.array([s["max_ae_k"] for s in samples_subset], dtype=np.float64)

        for sig_key, sig_name in signals:
            sig_arr = np.array([s[sig_key] for s in samples_subset], dtype=np.float64)

            # Avoid degenerate constant arrays
            if np.std(sig_arr) < 1e-12 or np.std(rmse_arr) < 1e-12:
                r_rmse, _p_r_rmse = 0.0, 1.0
                rho_rmse, _p_rho_rmse = 0.0, 1.0
                r_maxae, _p_r_maxae = 0.0, 1.0
                rho_maxae, _p_rho_maxae = 0.0, 1.0
            else:
                r_rmse, _p_r_rmse = stats.pearsonr(sig_arr, rmse_arr)
                rho_rmse, _p_rho_rmse = stats.spearmanr(sig_arr, rmse_arr)
                r_maxae, _p_r_maxae = stats.pearsonr(sig_arr, maxae_arr)
                rho_maxae, _p_rho_maxae = stats.spearmanr(sig_arr, maxae_arr)

            corr_dict[sig_key] = {
                "display_name": sig_name,
                "pearson_r_rmse": float(r_rmse),
                "spearman_rho_rmse": float(rho_rmse),
                "pearson_r_maxae": float(r_maxae),
                "spearman_rho_maxae": float(rho_maxae),
            }
        results["correlations"][scope_name] = corr_dict

    # 2. Detailed Breakdown across physical regimes
    # a. Contrast ratio buckets
    c_low = [s["rmse_k"] for s in all_pooled_samples if s["contrast"] < 5.0]
    c_med = [s["rmse_k"] for s in all_pooled_samples if 5.0 <= s["contrast"] < 20.0]
    c_high = [s["rmse_k"] for s in all_pooled_samples if s["contrast"] >= 20.0]

    # b. Boundary regime
    b_neu = [s["rmse_k"] for s in all_pooled_samples if s["both_neumann"]]
    b_dir = [s["rmse_k"] for s in all_pooled_samples if not s["both_neumann"]]

    results["error_breakdowns"] = {
        "by_contrast": {
            "contrast_under_5": compute_distribution_stats(c_low),
            "contrast_5_to_20": compute_distribution_stats(c_med),
            "contrast_ge_20": compute_distribution_stats(c_high),
        },
        "by_boundary_regime": {
            "both_neumann_top_bottom": compute_distribution_stats(b_neu),
            "has_dirichlet_top_bottom": compute_distribution_stats(b_dir),
        },
    }

    out_path = Path("artifacts/reports/milestone_08d_baseline_reliability.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved baseline reliability audit results to {out_path}")

    return results


if __name__ == "__main__":
    run_baseline_reliability_audit()
