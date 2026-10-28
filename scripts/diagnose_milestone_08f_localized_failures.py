"""Phase 1 Diagnosis Script for Milestone 08F: Localized Failure Analysis.

Investigates the worst 256x256 IID cases for M3, specifically analyzing:
- Full temperature error maps
- Peak error locations and distances to boundaries
- Conductivity interfaces and gradients
- Boundary conditions and flux errors
- Discrete cell-centred PDE residuals
- U1 aleatoric and U3 epistemic uncertainty maps
- Classification into global offset, boundary layer, interface jump, and spectral ringing
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from neural_process_twin.data.continuous_encoding import encode_continuous_physical_inputs
from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.models.uncertainty_fno import HeteroscedasticFNO2D, MCDropoutFNO2D
from neural_process_twin.solvers.assembly import assemble_steady_heat_system
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("milestone_08f_phase1")


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

    def _str(val: Any) -> str:
        return val.decode("utf-8").lower() if isinstance(val, bytes) else str(val).lower()

    def _side_bc(side: BoundarySide, t_str: str, val_arr: np.ndarray) -> BoundaryCondition:
        bc_t = BoundaryType.DIRICHLET if _str(t_str) == "dirichlet" else BoundaryType.NEUMANN
        return BoundaryCondition(side, bc_t, val_arr)

    return BoundaryConditions(
        left=_side_bc(BoundarySide.LEFT, l_type, l_val),
        right=_side_bc(BoundarySide.RIGHT, r_type, r_val),
        bottom=_side_bc(BoundarySide.BOTTOM, b_type, b_val),
        top=_side_bc(BoundarySide.TOP, t_type, t_val),
    )


def run_phase1_diagnosis() -> None:
    """Execute Phase 1 localized failure analysis on 256x256 IID queries."""
    logger.info("Loading frozen checkpoints: M3, U1, U3...")
    m3_path = Path("artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt")
    u1_path = Path("artifacts/checkpoints/milestone_08d/U1_heteroscedastic/best_model.pt")
    u3_path = Path("artifacts/checkpoints/milestone_08d/U3_mc_dropout/best_model.pt")

    m3_ckpt = torch.load(m3_path, map_location="cpu", weights_only=False)
    normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])

    m3_model = FourierNeuralOperator2D(
        in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4
    )
    m3_model.load_state_dict(m3_ckpt.get("model_state_dict", m3_ckpt))
    m3_model.eval()

    u1_model = HeteroscedasticFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
    u1_ckpt = torch.load(u1_path, map_location="cpu", weights_only=False)
    u1_model.load_state_dict(u1_ckpt.get("model_state_dict", u1_ckpt))
    u1_model.eval()

    u3_model = MCDropoutFNO2D(modes_x=12, modes_y=12, width=32, depth=4, dropout_p=0.10)
    u3_ckpt = torch.load(u3_path, map_location="cpu", weights_only=False)
    u3_model.load_state_dict(u3_ckpt.get("model_state_dict", u3_ckpt))
    u3_model.eval()

    grid = Grid2D(0.0, 1.0, 0.0, 1.0, 256, 256, thickness=1.0)

    # Load 256x256 IID test dataset
    h5_path = Path("artifacts/data/multires_dataset.h5")
    logger.info(f"Opening {h5_path} res_256/test_iid...")
    with h5py.File(h5_path, "r") as f:
        grp = f["res_256/test_iid"]
        n_samples = grp["temperature"].shape[0]
        k_all = grp["k"][:]
        q_all = grp["q"][:]
        t_all = grp["temperature"][:]
        bcs_all = [parse_bcs_row(grp, i) for i in range(n_samples)]

    case_diagnostics: list[dict[str, Any]] = []

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
        x_tensor = torch.from_numpy(x_in).unsqueeze(0).float()

        # M3 inference
        with torch.no_grad():
            out_norm = m3_model(x_tensor)
            t_pred = normalizer.inverse_transform_temperature(out_norm.numpy()[0, 0])

            # U1 inference
            _u1_mean_n, u1_logvar = u1_model(x_tensor)
            u1_std_k = np.sqrt(np.exp(u1_logvar.numpy()[0, 0])) * float(normalizer.t_stats.std)

            # U3 inference (T=3)
            u3_model.train()  # activate dropout
            u3_preds = []
            for _ in range(3):
                u3_out = u3_model(x_tensor)
                u3_preds.append(normalizer.inverse_transform_temperature(u3_out.numpy()[0, 0]))
            u3_std_k = np.std(np.stack(u3_preds, axis=0), axis=0, ddof=1)
            u3_model.eval()

        err_map = np.abs(t_pred - t_true)
        maxae = float(np.max(err_map))
        rmse = float(np.sqrt(np.mean(err_map**2)))
        rel_l2 = float(np.linalg.norm(err_map) / (np.linalg.norm(t_true) + 1e-8)) * 100.0

        # Location of peak error
        idx_max = np.unravel_index(np.argmax(err_map), err_map.shape)
        y_idx, x_idx = int(idx_max[0]), int(idx_max[1])
        x_phys = float(grid.x_mesh[y_idx, x_idx])
        y_phys = float(grid.y_mesh[y_idx, x_idx])

        # Distance to boundaries: [west, east, south, north]
        d_west = x_phys
        d_east = 1.0 - x_phys
        d_south = y_phys
        d_north = 1.0 - y_phys
        d_min_bnd = min(d_west, d_east, d_south, d_north)

        closest_side = (
            "left"
            if d_min_bnd == d_west
            else "right"
            if d_min_bnd == d_east
            else "bottom"
            if d_min_bnd == d_south
            else "top"
        )

        # Boundary type at closest side
        side_bc = getattr(bcs, closest_side)
        closest_bnd_type = side_bc.bc_type.value

        # Local conductivity gradient
        grad_ky, grad_kx = np.gradient(k_field, grid.dy, grid.dx)
        grad_k_mag = np.sqrt(grad_kx**2 + grad_ky**2)
        local_grad_k = float(grad_k_mag[y_idx, x_idx])
        local_k = float(k_field[y_idx, x_idx])
        k_contrast = float(np.max(k_field) / max(np.min(k_field), 1e-12))

        # Local source
        local_q = float(q_field[y_idx, x_idx])

        # Exact FVM PDE residual map: r = A*T - b [Watts]
        system = assemble_steady_heat_system(grid, k_field, q_field, bcs)
        t_flat = t_pred.reshape(-1)
        res_vec = np.abs(system.A @ t_flat - system.b)
        res_full_w = res_vec.reshape((grid.ny, grid.nx))
        local_res_w = float(res_full_w[y_idx, x_idx])

        # Heat balance diagnostics
        hb = compute_heat_balance(grid, t_pred, k_field, q_field, bcs)

        # Spatial error decomposition
        bnd_mask = np.zeros_like(err_map, dtype=bool)
        bnd_mask[0, :] = True
        bnd_mask[-1, :] = True
        bnd_mask[:, 0] = True
        bnd_mask[:, -1] = True
        bnd_maxae = float(np.max(err_map[bnd_mask]))
        int_maxae = float(np.max(err_map[~bnd_mask]))

        # Interface error (top 5% of conductivity gradient)
        high_grad_mask = grad_k_mag > np.percentile(grad_k_mag, 95)
        interface_maxae = float(np.max(err_map[high_grad_mask])) if np.any(high_grad_mask) else 0.0

        # Global offset
        global_offset = float(abs(np.mean(t_pred - t_true)))

        # Failure classification
        if d_min_bnd <= 2.0 * grid.dx:
            classification = "BOUNDARY_LAYER_ERROR"
        elif interface_maxae > 0.8 * maxae and local_grad_k > 10.0:
            classification = "CONDUCTIVITY_INTERFACE_ERROR"
        elif global_offset > 0.5 * maxae:
            classification = "GLOBAL_OFFSET_ERROR"
        else:
            classification = "SPECTRAL_GIBBS_RINGING"

        case_info = {
            "sample_index": i,
            "maxae_k": maxae,
            "rmse_k": rmse,
            "rel_l2_pct": rel_l2,
            "peak_location": {
                "grid_idx": [y_idx, x_idx],
                "phys_coord": [x_phys, y_phys],
                "d_min_boundary": d_min_bnd,
                "closest_boundary": closest_side,
                "closest_boundary_type": closest_bnd_type,
            },
            "local_physics_at_peak": {
                "conductivity_w_mk": local_k,
                "grad_k_mag": local_grad_k,
                "overall_k_contrast": k_contrast,
                "heat_source_w_m3": local_q,
                "cell_pde_residual_w": local_res_w,
            },
            "uncertainty_at_peak": {
                "u1_std_k": float(u1_std_k[y_idx, x_idx]),
                "u3_std_k": float(u3_std_k[y_idx, x_idx]),
                "u3_max_std_k": float(np.max(u3_std_k)),
            },
            "error_decomposition": {
                "global_offset_k": global_offset,
                "boundary_maxae_k": bnd_maxae,
                "interior_maxae_k": int_maxae,
                "interface_maxae_k": interface_maxae,
                "flux_imbalance_pct": float(hb.relative_imbalance * 100.0),
            },
            "classification": classification,
        }
        case_diagnostics.append(case_info)

    # Sort by MaxAE descending
    case_diagnostics.sort(key=lambda c: c["maxae_k"], reverse=True)

    # Summary statistics
    maxaes = np.array([c["maxae_k"] for c in case_diagnostics])
    rmses = np.array([c["rmse_k"] for c in case_diagnostics])
    d_bnds = np.array([c["peak_location"]["d_min_boundary"] for c in case_diagnostics])

    classifications = {}
    for c in case_diagnostics:
        cls_name = c["classification"]
        classifications[cls_name] = classifications.get(cls_name, 0) + 1

    summary = {
        "n_samples": n_samples,
        "maxae_distribution": {
            "min": float(np.min(maxaes)),
            "p25": float(np.percentile(maxaes, 25)),
            "median": float(np.median(maxaes)),
            "p75": float(np.percentile(maxaes, 75)),
            "p90": float(np.percentile(maxaes, 90)),
            "p95": float(np.percentile(maxaes, 95)),
            "max": float(np.max(maxaes)),
        },
        "threshold_violations": {
            "count_over_5k": int(np.sum(maxaes > 5.0)),
            "pct_over_5k": float(np.mean(maxaes > 5.0)) * 100.0,
            "count_over_10k": int(np.sum(maxaes > 10.0)),
            "pct_over_10k": float(np.mean(maxaes > 10.0)) * 100.0,
            "count_over_15k": int(np.sum(maxaes > 15.0)),
            "pct_over_15k": float(np.mean(maxaes > 15.0)) * 100.0,
            "count_over_20k": int(np.sum(maxaes > 20.0)),
            "pct_over_20k": float(np.mean(maxaes > 20.0)) * 100.0,
            "count_over_30k": int(np.sum(maxaes > 30.0)),
            "pct_over_30k": float(np.mean(maxaes > 30.0)) * 100.0,
        },
        "mean_rmse_k": float(np.mean(rmses)),
        "mean_peak_distance_to_boundary": float(np.mean(d_bnds)),
        "pct_peaks_within_3_cells_of_boundary": float(np.mean(d_bnds <= 3.0 * grid.dx)) * 100.0,
        "dominant_failure_mechanism_counts": classifications,
        "top_5_worst_cases": case_diagnostics[:5],
    }

    out_path = Path("artifacts/reports/milestone_08f_phase1_diagnosis.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Phase 1 Diagnosis Complete. Worst MaxAE = {summary['maxae_distribution']['max']:.2f} K | "
        f"Median MaxAE = {summary['maxae_distribution']['median']:.2f} K | "
        f"Violations (>10K): {summary['threshold_violations']['pct_over_10k']:.1f}% | "
        f"Peaks within 3 boundary cells: {summary['pct_peaks_within_3_cells_of_boundary']:.1f}%"
    )
    logger.info(f"Report saved to {out_path}")


if __name__ == "__main__":
    run_phase1_diagnosis()
