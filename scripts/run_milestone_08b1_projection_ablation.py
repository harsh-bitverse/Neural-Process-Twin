"""Milestone 08B.1 Phase 4: Boundary Projection Ablation on Validation Split.

Compares three boundary treatments on BoundaryAwareFNO2D (Seed 42) across the validation split (N=150):
- Variant B0: No boundary projection (correction_mode="none")
- Variant B1: Extrapolation-based projection (correction_mode="hard" / extrapolation: T_0 = (2/3)*T_dir + (1/3)*T_1)
- Variant B2: Exact discrete-consistent boundary update (correction_mode="fvm_balance": (A*T - b)_bnd = 0)

Measures:
- Overall RMSE [K]
- Interior RMSE [K]
- Boundary RMSE [K]
- Relative L2 Error [%]
- Maximum Absolute Error [K]
- Boundary Heat Flow Imbalance [W]
- Relative Heat Imbalance [%]
- Discrete PDE Residual (A*T - b) [W]

Outputs:
- Results saved to artifacts/reports/milestone_08b1_projection_ablation.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_process_twin.data.boundary_encoding import decode_boundary_conditions
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.models.boundary_aware_fno import BoundaryAwareFNO2D
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.training.physics_loss import DiscretePDEResidual

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("milestone_08b1_ablation")


def evaluate_projection_variant(
    model: BoundaryAwareFNO2D,
    mode: str,
    val_ds: HeatFieldDataset,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate BoundaryAwareFNO2D under a specific correction mode on validation partition."""
    # Set model correction mode
    model.correction_mode = mode
    model.boundary_correction.correction_mode = mode
    model.eval()

    grid = val_ds.grid
    normalizer = val_ds.normalizer
    if normalizer is None or normalizer.t_stats is None:
        raise ValueError("Normalizer stats missing from dataset.")

    t_mean = float(normalizer.t_stats.mean)
    t_std = float(normalizer.t_stats.std)

    pde_op = DiscretePDEResidual(grid=grid)

    overall_rmses: list[float] = []
    interior_rmses: list[float] = []
    boundary_rmses: list[float] = []
    rel_l2s: list[float] = []
    max_aes: list[float] = []
    flux_imbalances: list[float] = []
    rel_imbalances: list[float] = []
    pde_residuals: list[float] = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            sample = val_ds[i]
            x_tensor = sample["x"].unsqueeze(0).to(device)
            t_ref_k = sample["temperature_raw"].squeeze().numpy()
            k_raw = sample["k_raw"].squeeze().numpy()
            q_raw = sample["q_raw"].squeeze().numpy()

            k_raw_tensor = sample["k_raw"].unsqueeze(0).to(device)
            q_raw_tensor = sample["q_raw"].unsqueeze(0).to(device)

            # Predict with active correction mode
            pred_norm = (
                model(
                    x_tensor,
                    k_raw=k_raw_tensor,
                    q_raw=q_raw_tensor,
                    t_mean=t_mean,
                    t_std=t_std,
                    apply_boundary_correction=(mode != "none"),
                )
                .squeeze(0)
                .squeeze(0)
                .cpu()
                .numpy()
            )
            pred_k = pred_norm * t_std + t_mean

            # Errors
            diff = pred_k - t_ref_k
            overall_rmses.append(float(np.sqrt(np.mean(diff**2))))
            interior_rmses.append(float(np.sqrt(np.mean(diff[1:-1, 1:-1] ** 2))))

            bnd_mask = np.ones_like(diff, dtype=bool)
            bnd_mask[1:-1, 1:-1] = False
            boundary_rmses.append(float(np.sqrt(np.mean(diff[bnd_mask] ** 2))))

            rel_l2s.append(float(np.linalg.norm(diff) / np.linalg.norm(t_ref_k) * 100.0))
            max_aes.append(float(np.max(np.abs(diff))))

            # Heat balance
            bc_tensor_np = sample["x"][4:12].numpy()
            bcs = decode_boundary_conditions(bc_tensor_np, grid=grid, normalizer=normalizer)
            diag = compute_heat_balance(grid, pred_k, k_raw, q_raw, bcs)
            flux_imbalances.append(float(diag.flux_imbalance))
            rel_imbalances.append(float(diag.relative_imbalance * 100.0))

            # Discrete PDE Residual
            temp_k_tensor = torch.from_numpy(pred_k).float().unsqueeze(0).unsqueeze(0).to(device)
            bc_chan_tensor = sample["x"][4:12].unsqueeze(0).to(device)
            r_tensor, _ = pde_op(
                temperature_k=temp_k_tensor,
                k_field=k_raw_tensor,
                q_field=q_raw_tensor,
                bc_channels=bc_chan_tensor,
                t_mean=t_mean,
                t_std=t_std,
            )
            pde_residuals.append(float(torch.norm(r_tensor).item()))

    def stats(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
        }

    return {
        "variant": mode,
        "sample_count": len(val_ds),
        "overall_rmse_k": stats(overall_rmses),
        "interior_rmse_k": stats(interior_rmses),
        "boundary_rmse_k": stats(boundary_rmses),
        "relative_l2_pct": stats(rel_l2s),
        "max_ae_k": stats(max_aes),
        "flux_imbalance_w": stats(flux_imbalances),
        "relative_imbalance_pct": stats(rel_imbalances),
        "discrete_pde_residual_w": stats(pde_residuals),
    }


def run_projection_ablation() -> dict[str, Any]:
    dataset_path = Path("artifacts/data/scaled_dataset.h5")
    checkpoint_path = Path(
        "artifacts/checkpoints/milestone_08b/boundary_aware_fno_seed42/best_model.pt"
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing BA-FNO checkpoint: {checkpoint_path}")

    dev = torch.device("cpu")
    val_ds = HeatFieldDataset(
        dataset_path, split="validation", include_coords=True, include_bcs=True
    )
    grid = val_ds.grid

    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model = BoundaryAwareFNO2D(
        grid=grid,
        in_channels=12,
        out_channels=1,
        modes_x=12,
        modes_y=12,
        width=32,
        depth=4,
        activation="gelu",
        use_boundary_bypass=True,
        correction_mode="hard",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)

    modes = {
        "B0_none": "none",
        "B1_extrapolation": "hard",
        "B2_fvm_balance": "fvm_balance",
    }

    results: dict[str, Any] = {}
    for var_name, mode in modes.items():
        logger.info(f"=== Evaluating Variant {var_name} (mode='{mode}') on Validation Set ===")
        res = evaluate_projection_variant(model, mode, val_ds, dev)
        results[var_name] = res
        logger.info(
            f"{var_name} -> "
            f"Overall RMSE: {res['overall_rmse_k']['mean']:.3f} K | "
            f"Interior RMSE: {res['interior_rmse_k']['mean']:.3f} K | "
            f"Boundary RMSE: {res['boundary_rmse_k']['mean']:.3f} K | "
            f"Flux Imbal: {res['flux_imbalance_w']['mean']:.1f} W | "
            f"PDE Res: {res['discrete_pde_residual_w']['mean']:.1f} W"
        )

    out_path = Path("artifacts/reports/milestone_08b1_projection_ablation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Boundary projection ablation saved to {out_path}")

    return results


if __name__ == "__main__":
    run_projection_ablation()
