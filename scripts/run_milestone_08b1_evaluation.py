"""Milestone 08B.1 Phase 5: Comprehensive Independent Evaluation.

Evaluates candidates on:
1. test_iid (N=200) from scaled_dataset.h5
2. test_ood (N=200) from scaled_dataset.h5
   - Subgroup High-Contrast Inclusions (N=153)
   - Subgroup Short-Correlation GRF (N=47)
3. calibration (N=200) from calibration_dataset.h5 (independent uncorrupted holdout)

Evaluates:
- A0: Frozen Supervised FNO baseline (scaled_fno_supervised_seed42)
- A1: 08B Face-Consistent FNO (w_bc = 0.05) (fno_face_consistent_seed42)
- A1_recalibrated: Pareto-optimal w_bc* from Phase 3
- A2: 08B Boundary-Aware FNO with extrapolation projection (hard)
- A2_fvm: 08B.1 Boundary-Aware FNO with exact discrete FVM balance (fvm_balance)

Outputs:
- artifacts/reports/milestone_08b1_results.json
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_process_twin.data.boundary_encoding import decode_boundary_conditions
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.models.boundary_aware_fno import BoundaryAwareFNO2D
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.training.physics_loss import DiscretePDEResidual

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("milestone_08b1_evaluation")


def evaluate_model_on_dataset(
    model: torch.nn.Module,
    ds: HeatFieldDataset,
    device: torch.device,
    is_ba_fno: bool = False,
    ba_mode: str = "none",
) -> dict[str, Any]:
    """Evaluate a single model on a dataset partition and return aggregated metrics."""
    grid = ds.grid
    normalizer = ds.normalizer
    if normalizer is None or normalizer.t_stats is None:
        raise ValueError("Normalizer stats missing.")

    t_mean = float(normalizer.t_stats.mean)
    t_std = float(normalizer.t_stats.std)

    if is_ba_fno and isinstance(model, BoundaryAwareFNO2D):
        model.correction_mode = ba_mode
        model.boundary_correction.correction_mode = ba_mode
    model.eval()

    pde_op = DiscretePDEResidual(grid=grid)

    latencies_ms: list[float] = []
    overall_rmses: list[float] = []
    interior_rmses: list[float] = []
    boundary_rmses: list[float] = []
    rel_l2s: list[float] = []
    max_aes: list[float] = []
    flux_imbalances_abs: list[float] = []
    flux_imbalances_signed: list[float] = []
    rel_imbalances: list[float] = []
    pde_residuals: list[float] = []
    q_gen_list: list[float] = []
    q_out_list: list[float] = []

    # Warmup
    with torch.no_grad():
        x_warmup = ds[0]["x"].unsqueeze(0).to(device)
        for _ in range(5):
            _ = model(x_warmup)

    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            x_tensor = sample["x"].unsqueeze(0).to(device)
            t_ref_k = sample["temperature_raw"].squeeze().numpy()
            k_raw = sample["k_raw"].squeeze().numpy()
            q_raw = sample["q_raw"].squeeze().numpy()

            k_raw_tensor = sample["k_raw"].unsqueeze(0).to(device)
            q_raw_tensor = sample["q_raw"].unsqueeze(0).to(device)

            t0 = time.perf_counter()
            if is_ba_fno:
                pred_norm = model(
                    x_tensor,
                    k_raw=k_raw_tensor,
                    q_raw=q_raw_tensor,
                    t_mean=t_mean,
                    t_std=t_std,
                    apply_boundary_correction=(ba_mode != "none"),
                )
            else:
                pred_norm = model(x_tensor)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(lat_ms)

            pred_k = pred_norm.squeeze(0).squeeze(0).cpu().numpy() * t_std + t_mean

            # Errors
            diff = pred_k - t_ref_k
            overall_rmses.append(float(np.sqrt(np.mean(diff**2))))
            interior_rmses.append(float(np.sqrt(np.mean(diff[1:-1, 1:-1] ** 2))))

            bnd_mask = np.ones_like(diff, dtype=bool)
            bnd_mask[1:-1, 1:-1] = False
            boundary_rmses.append(float(np.sqrt(np.mean(diff[bnd_mask] ** 2))))

            rel_l2s.append(float(np.linalg.norm(diff) / np.linalg.norm(t_ref_k) * 100.0))
            max_aes.append(float(np.max(np.abs(diff))))

            # Physical energy balance
            bc_tensor_np = sample["x"][4:12].numpy()
            bcs = decode_boundary_conditions(bc_tensor_np, grid=grid, normalizer=normalizer)
            diag = compute_heat_balance(grid, pred_k, k_raw, q_raw, bcs)

            q_out = float(diag.q_boundary_out_total)
            q_gen = float(diag.q_generated_total)
            q_out_list.append(q_out)
            q_gen_list.append(q_gen)

            imbalance_signed = q_out - q_gen
            imbalance_abs = abs(imbalance_signed)
            flux_imbalances_signed.append(imbalance_signed)
            flux_imbalances_abs.append(imbalance_abs)
            rel_imbalances.append(float(diag.relative_imbalance * 100.0))

            # Discrete PDE Residual (W)
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
        "sample_count": len(ds),
        "latency_ms": stats(latencies_ms),
        "overall_rmse_k": stats(overall_rmses),
        "interior_rmse_k": stats(interior_rmses),
        "boundary_rmse_k": stats(boundary_rmses),
        "relative_l2_pct": stats(rel_l2s),
        "max_ae_k": stats(max_aes),
        "flux_imbalance_abs_w": stats(flux_imbalances_abs),
        "flux_imbalance_signed_w": stats(flux_imbalances_signed),
        "relative_imbalance_pct": stats(rel_imbalances),
        "discrete_pde_residual_w": stats(pde_residuals),
        "q_gen_total_w": stats(q_gen_list),
        "q_boundary_out_total_w": stats(q_out_list),
    }


def evaluate_subgroup(
    model: torch.nn.Module,
    ds_full: HeatFieldDataset,
    indices: list[int],
    device: torch.device,
    is_ba_fno: bool = False,
    ba_mode: str = "none",
) -> dict[str, Any]:
    """Evaluate model on a subset of indices from a dataset."""
    sub_ds = torch.utils.data.Subset(ds_full, indices)

    # Wrap in lightweight adapter that exposes needed attributes
    class SubsetAdapter:
        def __init__(self, subset, parent):
            self.subset = subset
            self.grid = parent.grid
            self.normalizer = parent.normalizer

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            return self.subset[idx]

    adapter = SubsetAdapter(sub_ds, ds_full)
    return evaluate_model_on_dataset(model, adapter, device, is_ba_fno, ba_mode)


def run_full_evaluation(optimal_w_bc: float = 0.001) -> dict[str, Any]:
    """Run full evaluation across holdout test and calibration datasets."""
    logger.info("Starting Milestone 08B.1 Phase 5 Full Independent Evaluation...")
    dev = torch.device("cpu")

    # Load datasets
    scaled_ds_path = Path("artifacts/data/scaled_dataset.h5")
    calib_ds_path = Path("artifacts/data/calibration_dataset.h5")

    ds_test_iid = HeatFieldDataset(
        scaled_ds_path, split="test_iid", include_coords=True, include_bcs=True
    )
    ds_test_ood = HeatFieldDataset(
        scaled_ds_path, split="test_ood", include_coords=True, include_bcs=True
    )
    ds_calib = HeatFieldDataset(
        calib_ds_path, split="calibration", include_coords=True, include_bcs=True
    )

    grid = ds_test_iid.grid

    # Partition test_ood into subgroups
    hc_indices = list(range(153))
    grf_indices = list(range(153, 200))

    # Candidate definitions
    candidates: dict[str, dict[str, Any]] = {
        "A0_fno_supervised": {
            "type": "fno2d",
            "checkpoint": Path("artifacts/checkpoints/scaled_fno_supervised_seed42/best_model.pt"),
            "is_ba": False,
            "ba_mode": "none",
        },
        "A1_face_consistent_w0.05": {
            "type": "fno2d",
            "checkpoint": Path(
                "artifacts/checkpoints/milestone_08b/fno_face_consistent_seed42/best_model.pt"
            ),
            "is_ba": False,
            "ba_mode": "none",
        },
        "A1_recalibrated": {
            "type": "fno2d",
            "checkpoint": Path(
                f"artifacts/checkpoints/milestone_08b1/fno_wbc_{optimal_w_bc}_seed42/best_model.pt"
            ),
            "is_ba": False,
            "ba_mode": "none",
        },
        "A2_ba_fno_extrapolation": {
            "type": "ba_fno",
            "checkpoint": Path(
                "artifacts/checkpoints/milestone_08b/boundary_aware_fno_seed42/best_model.pt"
            ),
            "is_ba": True,
            "ba_mode": "hard",
        },
        "A2_ba_fno_fvm_balance": {
            "type": "ba_fno",
            "checkpoint": Path(
                "artifacts/checkpoints/milestone_08b/boundary_aware_fno_seed42/best_model.pt"
            ),
            "is_ba": True,
            "ba_mode": "fvm_balance",
        },
    }

    full_results: dict[str, Any] = {}

    for cand_id, cand_info in candidates.items():
        ckpt_path = cand_info["checkpoint"]
        if not ckpt_path.is_file():
            logger.warning(f"Checkpoint {ckpt_path} not found. Skipping candidate {cand_id}.")
            continue

        logger.info(f"--- Evaluating Candidate: {cand_id} ---")
        checkpoint = torch.load(ckpt_path, map_location=dev, weights_only=False)

        if cand_info["type"] == "fno2d":
            model = FourierNeuralOperator2D(
                in_channels=12,
                out_channels=1,
                modes_x=12,
                modes_y=12,
                width=32,
                depth=4,
                activation="gelu",
            )
        else:
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
                correction_mode=cand_info["ba_mode"],
            )

        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(dev)

        is_ba = cand_info["is_ba"]
        ba_mode = cand_info["ba_mode"]

        # 1. test_iid
        res_iid = evaluate_model_on_dataset(model, ds_test_iid, dev, is_ba, ba_mode)
        # 2. test_ood full
        res_ood = evaluate_model_on_dataset(model, ds_test_ood, dev, is_ba, ba_mode)
        # 3. test_ood high-contrast (N=153)
        res_hc = evaluate_subgroup(model, ds_test_ood, hc_indices, dev, is_ba, ba_mode)
        # 4. test_ood GRF (N=47)
        res_grf = evaluate_subgroup(model, ds_test_ood, grf_indices, dev, is_ba, ba_mode)
        # 5. calibration holdout (N=200)
        res_calib = evaluate_model_on_dataset(model, ds_calib, dev, is_ba, ba_mode)

        cand_results = {
            "candidate_id": cand_id,
            "architecture": cand_info["type"],
            "checkpoint": str(ckpt_path),
            "ba_mode": ba_mode,
            "test_iid": res_iid,
            "test_ood_full": res_ood,
            "test_ood_high_contrast": res_hc,
            "test_ood_grf": res_grf,
            "holdout_calibration": res_calib,
        }
        full_results[cand_id] = cand_results

        logger.info(
            f"{cand_id} -> "
            f"IID RMSE: {res_iid['overall_rmse_k']['mean']:.3f} K | "
            f"OOD RMSE: {res_ood['overall_rmse_k']['mean']:.3f} K | "
            f"GRF RMSE: {res_grf['overall_rmse_k']['mean']:.3f} K | "
            f"GRF Flux Imbal: {res_grf['flux_imbalance_abs_w']['mean']:.1f} W | "
            f"Calib RMSE: {res_calib['overall_rmse_k']['mean']:.3f} K"
        )

    out_file = Path("artifacts/reports/milestone_08b1_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"Full independent evaluation saved to {out_file}")

    return full_results


if __name__ == "__main__":
    run_full_evaluation()
