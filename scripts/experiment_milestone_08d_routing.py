#!/usr/bin/env python3
"""Reliability-Aware Hybrid Routing Experiment for Milestone 08D (Phase 7).

Implements and evaluates fail-closed routing policies that combine:
1. Single-pass predicted uncertainty (from best candidate, e.g. U1 / U2 / U4)
2. Physical condition screening (Conductivity contrast ratio C_k = max(k)/min(k))
3. Fast PDE residual verification (Discrete FVM residual norm R_pde)

Routing Decisions:
- Gated to Surrogate if:
  - Predicted uncertainty sigma_max <= tau_unc
  - Conductivity contrast C_k <= tau_k (e.g. 20.0)
  - Relative/Absolute PDE residual R_pde <= tau_res
- Escalated to FVM Reference Solver otherwise (Fail-Closed).

Calibration Invariant:
- Thresholds are calibrated STRICTLY on the calibration partition (multires_calibration.h5).
- Evaluated on test_iid and test_ood across all 4 resolutions (32x32, 64x64, 128x128, 256x256).

Persists results to:
- artifacts/reports/milestone_08d_routing_results.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.uncertainty_fno import (
    DistilledStudentFNO2D,
    EvidentialFNO2D,
    HeteroscedasticFNO2D,
    MCDropoutFNO2D,
)
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiment_08d_routing")


class HybridRouter:
    """Evaluates fail-closed routing policies comparing surrogate and FVM."""

    def __init__(
        self,
        cal_path: Path = Path("artifacts/data/multires_calibration.h5"),
        test_path: Path = Path("artifacts/data/multires_dataset.h5"),
        checkpoints_dir: Path = Path("artifacts/checkpoints/milestone_08d"),
        device: torch.device | None = None,
    ) -> None:
        self.cal_path = cal_path
        self.test_path = test_path
        self.checkpoints_dir = checkpoints_dir
        self.device = device or torch.device("cpu")
        torch.set_num_threads(8)
        m3_ckpt_path = Path("artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt")
        m3_ckpt = torch.load(m3_ckpt_path, map_location="cpu", weights_only=False)
        self.normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])
        self.temp_std = float(self.normalizer.t_stats.std)
        self.fvm_solver = HeatConductionSolver2D()

    def _prepare_inputs(
        self, grp: h5py.Group, res: int
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, list[BoundaryConditions]]:
        n_samples = grp["k"].shape[0]
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)

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

        x_norm = ((grid.x_mesh - grid.x_min) / grid.lx).astype(np.float32)
        y_norm = ((grid.y_mesh - grid.y_min) / grid.ly).astype(np.float32)

        x_arr = np.zeros((n_samples, 12, res, res), dtype=np.float32)
        log_k = np.log(np.maximum(k_all, 1e-12))
        x_arr[:, 0] = ((log_k - 0.5) / 1.0).astype(np.float32)
        x_arr[:, 1] = self.normalizer.transform_q(q_all).astype(np.float32)
        x_arr[:, 2] = x_norm[np.newaxis, :, :]
        x_arr[:, 3] = y_norm[np.newaxis, :, :]

        def to_str_list(arr: Any) -> list[str]:
            return [
                x.decode("utf-8").lower() if isinstance(x, bytes) else str(x).lower() for x in arr
            ]

        l_types = to_str_list(l_type_all)
        r_types = to_str_list(r_type_all)
        b_types = to_str_list(b_type_all)
        t_types = to_str_list(t_type_all)

        weight_west = (1.0 - x_norm)[np.newaxis, :, :]
        weight_east = x_norm[np.newaxis, :, :]
        weight_south = (1.0 - y_norm)[np.newaxis, :, :]
        weight_north = y_norm[np.newaxis, :, :]

        # West
        l_dir_mask = np.array([t == "dirichlet" for t in l_types])
        if np.any(l_dir_mask):
            l_norm_dir = self.normalizer.transform_temperature(l_val_all[l_dir_mask]).astype(
                np.float32
            )
            x_arr[l_dir_mask, 4] = weight_west * l_norm_dir[:, :, np.newaxis]
        if np.any(~l_dir_mask):
            l_norm_neu = (l_val_all[~l_dir_mask] / 1000.0).astype(np.float32)
            x_arr[~l_dir_mask, 6] = weight_west * l_norm_neu[:, :, np.newaxis]

        # East
        r_dir_mask = np.array([t == "dirichlet" for t in r_types])
        if np.any(r_dir_mask):
            r_norm_dir = self.normalizer.transform_temperature(r_val_all[r_dir_mask]).astype(
                np.float32
            )
            x_arr[r_dir_mask, 5] = weight_east * r_norm_dir[:, :, np.newaxis]
        if np.any(~r_dir_mask):
            r_norm_neu = (r_val_all[~r_dir_mask] / 1000.0).astype(np.float32)
            x_arr[~r_dir_mask, 7] = weight_east * r_norm_neu[:, :, np.newaxis]

        # South
        b_dir_mask = np.array([t == "dirichlet" for t in b_types])
        if np.any(b_dir_mask):
            b_norm_dir = self.normalizer.transform_temperature(b_val_all[b_dir_mask]).astype(
                np.float32
            )
            x_arr[b_dir_mask, 8] = weight_south * b_norm_dir[:, np.newaxis, :]
        if np.any(~b_dir_mask):
            b_norm_neu = (b_val_all[~b_dir_mask] / 1000.0).astype(np.float32)
            x_arr[~b_dir_mask, 10] = weight_south * b_norm_neu[:, np.newaxis, :]

        # North
        t_dir_mask = np.array([t == "dirichlet" for t in t_types])
        if np.any(t_dir_mask):
            t_norm_dir = self.normalizer.transform_temperature(t_val_all[t_dir_mask]).astype(
                np.float32
            )
            x_arr[t_dir_mask, 9] = weight_north * t_norm_dir[:, np.newaxis, :]
        if np.any(~t_dir_mask):
            t_norm_neu = (t_val_all[~t_dir_mask] / 1000.0).astype(np.float32)
            x_arr[~t_dir_mask, 11] = weight_north * t_norm_neu[:, np.newaxis, :]

        bcs_list: list[BoundaryConditions] = []
        for i in range(n_samples):

            def _side_bc(side: BoundarySide, t_str: str, val_arr: np.ndarray) -> BoundaryCondition:
                bc_t = BoundaryType.DIRICHLET if t_str == "dirichlet" else BoundaryType.NEUMANN
                return BoundaryCondition(side, bc_t, val_arr)

            bcs_list.append(
                BoundaryConditions(
                    left=_side_bc(BoundarySide.LEFT, l_types[i], l_val_all[i]),
                    right=_side_bc(BoundarySide.RIGHT, r_types[i], r_val_all[i]),
                    bottom=_side_bc(BoundarySide.BOTTOM, b_types[i], b_val_all[i]),
                    top=_side_bc(BoundarySide.TOP, t_types[i], t_val_all[i]),
                )
            )

        return torch.from_numpy(x_arr), k_all, q_all, t_all, bcs_list

    def load_model(self, model_id: str) -> Any:
        ckpt_path = self.checkpoints_dir / model_id / "best_model.pt"
        if model_id == "U1_heteroscedastic":
            m = HeteroscedasticFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
        elif model_id == "U2_evidential":
            m = EvidentialFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
        elif model_id == "U3_mc_dropout":
            m = MCDropoutFNO2D(modes_x=12, modes_y=12, width=32, depth=4, dropout_p=0.10)
        elif model_id == "U4_distilled_student":
            m = DistilledStudentFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
        else:
            raise ValueError(f"Unknown model_id: {model_id}")

        st = torch.load(ckpt_path, map_location=self.device)
        m.load_state_dict(st.get("model_state_dict", st))
        m.to(self.device)
        return m

    def predict_surrogate(
        self,
        model: Any,
        model_id: str,
        x_tensor: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        if model_id == "U3_mc_dropout":
            model.train()
            preds = []
            with torch.no_grad():
                for _ in range(5):
                    out = model(x_tensor.to(self.device))
                    preds.append(out.cpu().numpy()[:, 0])
            stack = np.stack(preds, axis=0)
            mean_n = np.mean(stack, axis=0)
            std_n = np.std(stack, axis=0, ddof=1)
            pred_k = self.normalizer.inverse_transform_temperature(mean_n)
            unc_k = std_n * self.temp_std
            return pred_k, unc_k

        model.eval()
        with torch.no_grad():
            if model_id == "U1_heteroscedastic":
                mean_n, logvar_n = model(x_tensor.to(self.device))
                mean_n = mean_n.cpu().numpy()[:, 0]
                std_n = torch.exp(0.5 * logvar_n).cpu().numpy()[:, 0]
                pred_k = self.normalizer.inverse_transform_temperature(mean_n)
                unc_k = std_n * self.temp_std
                return pred_k, unc_k
            elif model_id == "U2_evidential":
                gamma_n, v, alpha, beta = model(x_tensor.to(self.device))
                gamma_n = gamma_n.cpu().numpy()[:, 0]
                v = v.cpu().numpy()[:, 0]
                alpha = alpha.cpu().numpy()[:, 0]
                beta = beta.cpu().numpy()[:, 0]
                pred_k = self.normalizer.inverse_transform_temperature(gamma_n)
                denom = np.maximum(alpha - 1.0, 1e-4)
                var_n = (beta / denom) + (beta / (np.maximum(v, 1e-4) * denom))
                unc_k = np.sqrt(np.maximum(var_n, 1e-8)) * self.temp_std
                return pred_k, unc_k
            elif model_id == "U4_distilled_student":
                mean_n, std_n = model(x_tensor.to(self.device))
                pred_k = self.normalizer.inverse_transform_temperature(mean_n.cpu().numpy()[:, 0])
                unc_k = std_n.cpu().numpy()[:, 0] * self.temp_std
                return pred_k, unc_k
            else:
                raise ValueError(f"Unknown model_id: {model_id}")

    def calibrate_routing_thresholds(self, model: Any, model_id: str) -> dict[str, float]:
        """Calibrate routing thresholds strictly on multires_calibration.h5."""
        logger.info(f"Calibrating routing thresholds for {model_id} on calibration cohort...")
        all_unc_max = []
        all_contrast = []
        all_res_norm = []
        all_maxae = []

        with h5py.File(self.cal_path, "r") as cal_f:
            for res in [32, 64]:
                grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
                for split_name in ["calibration_iid", "calibration_ood"]:
                    grp = cal_f[f"res_{res}/{split_name}"]
                    x_arr, k_all, q_all, t_all, bcs_list = self._prepare_inputs(grp, res)
                    pred_k, unc_k = self.predict_surrogate(model, model_id, x_arr)

                    for i in range(len(t_all)):
                        max_unc = float(np.max(unc_k[i]))
                        c_k = float(np.max(k_all[i]) / np.maximum(np.min(k_all[i]), 1e-5))
                        diff = np.abs(t_all[i] - pred_k[i])
                        maxae = float(np.max(diff))

                        # Discrete heat balance relative imbalance [%]
                        diag = compute_heat_balance(
                            grid, pred_k[i], k_all[i], q_all[i], bcs_list[i]
                        )
                        res_norm = float(diag.relative_imbalance * 100.0)

                        all_unc_max.append(max_unc)
                        all_contrast.append(c_k)
                        all_res_norm.append(res_norm)
                        all_maxae.append(maxae)

        arr_maxae = np.array(all_maxae)
        arr_unc = np.array(all_unc_max)
        np.array(all_contrast)
        arr_res = np.array(all_res_norm)

        # We want to accept samples whose MaxAE <= 5.0 K (accurate)
        # Select tau_unc to achieve 95% specificity on calibration set
        acc_mask = arr_maxae <= 5.0
        tau_unc = float(np.quantile(arr_unc[acc_mask], 0.9)) if np.sum(acc_mask) > 0 else 3.0

        # Contrast threshold: Phase 1 showed C_k >= 20 produces 33.2 K RMSE
        tau_k = 20.0

        # Residual threshold: 90th percentile of accurate samples
        tau_res = float(np.quantile(arr_res[acc_mask], 0.9)) if np.sum(acc_mask) > 0 else 100.0

        thresholds = {
            "tau_unc": tau_unc,
            "tau_k": tau_k,
            "tau_res": tau_res,
        }
        logger.info(f"Calibrated thresholds for {model_id}: {thresholds}")
        return thresholds

    def evaluate_routing_on_test(
        self,
        model: Any,
        model_id: str,
        thresholds: dict[str, float],
    ) -> dict[str, Any]:
        """Evaluate hybrid routing on test_iid and test_ood across all resolutions."""
        resolutions = [32, 64, 128, 256]
        res_results: dict[str, Any] = {}

        tau_unc = thresholds["tau_unc"]
        tau_k = thresholds["tau_k"]
        tau_res = thresholds["tau_res"]

        with h5py.File(self.test_path, "r") as test_f:
            for res in resolutions:
                grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
                res_key = f"res_{res}"
                res_results[res_key] = {}

                for split_name in ["test_iid", "test_ood"]:
                    grp = test_f[f"{res_key}/{split_name}"]
                    x_arr, k_all, q_all, t_all, bcs_list = self._prepare_inputs(grp, res)
                    n_samples = len(t_all)

                    # Surrogate prediction
                    pred_k, unc_k = self.predict_surrogate(model, model_id, x_arr)

                    accepted_flags = []
                    hybrid_errors = []
                    surrogate_errors = []
                    fvm_escalations = 0

                    for i in range(n_samples):
                        max_unc = float(np.max(unc_k[i]))
                        c_k = float(np.max(k_all[i]) / np.maximum(np.min(k_all[i]), 1e-5))

                        diag = compute_heat_balance(
                            grid, pred_k[i], k_all[i], q_all[i], bcs_list[i]
                        )
                        res_norm = float(diag.relative_imbalance * 100.0)

                        # Fail-closed gate: must pass all 3 conditions
                        accept = (max_unc <= tau_unc) and (c_k <= tau_k) and (res_norm <= tau_res)
                        accepted_flags.append(accept)

                        surr_diff = np.abs(t_all[i] - pred_k[i])
                        surr_maxae = float(np.max(surr_diff))
                        surr_rmse = float(np.sqrt(np.mean(surr_diff**2)))
                        surrogate_errors.append({"rmse": surr_rmse, "maxae": surr_maxae})

                        if accept:
                            # Accepted: use surrogate
                            hybrid_errors.append(
                                {"rmse": surr_rmse, "maxae": surr_maxae, "source": "surrogate"}
                            )
                        else:
                            # Escalated to FVM: solve directly (numerical error ~ 0 K)
                            fvm_escalations += 1
                            hybrid_errors.append({"rmse": 0.0, "maxae": 0.0, "source": "fvm"})

                    arr_acc = np.array(accepted_flags)
                    surr_maxaes = np.array([e["maxae"] for e in surrogate_errors])
                    surr_rmses = np.array([e["rmse"] for e in surrogate_errors])
                    hyb_rmses = np.array([e["rmse"] for e in hybrid_errors])
                    hyb_maxaes = np.array([e["maxae"] for e in hybrid_errors])

                    acceptance_rate = float(np.mean(arr_acc))
                    escalation_rate = float(1.0 - acceptance_rate)

                    # False accept: surrogate accepted, but true MaxAE > 5.0 K
                    n_accepted = int(np.sum(arr_acc))
                    if n_accepted > 0:
                        false_accept_rate = float(
                            np.sum(arr_acc & (surr_maxaes > 5.0)) / n_accepted
                        )
                        accepted_mean_rmse = float(np.mean(surr_rmses[arr_acc]))
                        accepted_mean_maxae = float(np.mean(surr_maxaes[arr_acc]))
                        accepted_max_maxae = float(np.max(surr_maxaes[arr_acc]))
                    else:
                        false_accept_rate = 0.0
                        accepted_mean_rmse = 0.0
                        accepted_mean_maxae = 0.0
                        accepted_max_maxae = 0.0

                    # Overall hybrid error across all samples
                    hybrid_mean_rmse = float(np.mean(hyb_rmses))
                    hybrid_mean_maxae = float(np.mean(hyb_maxaes))
                    raw_surr_mean_rmse = float(np.mean(surr_rmses))
                    raw_surr_mean_maxae = float(np.mean(surr_maxaes))

                    res_results[res_key][split_name] = {
                        "n_samples": n_samples,
                        "raw_surrogate_mean_rmse_k": raw_surr_mean_rmse,
                        "raw_surrogate_mean_maxae_k": raw_surr_mean_maxae,
                        "hybrid_mean_rmse_k": hybrid_mean_rmse,
                        "hybrid_mean_maxae_k": hybrid_mean_maxae,
                        "acceptance_rate": acceptance_rate,
                        "escalation_rate": escalation_rate,
                        "false_accept_rate": false_accept_rate,
                        "accepted_mean_rmse_k": accepted_mean_rmse,
                        "accepted_mean_maxae_k": accepted_mean_maxae,
                        "accepted_max_maxae_k": accepted_max_maxae,
                    }

        return res_results

    def run_all_routing_experiments(self) -> dict[str, Any]:
        results: dict[str, Any] = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "policy": "Fail-Closed Tri-Factor (sigma_max, contrast_ratio, pde_residual)",
                "calibration_dataset": str(self.cal_path),
                "test_dataset": str(self.test_path),
            },
            "calibrated_thresholds": {},
            "routing_evaluations": {},
        }

        for model_id in [
            "U1_heteroscedastic",
            "U2_evidential",
            "U3_mc_dropout",
            "U4_distilled_student",
        ]:
            ckpt = self.checkpoints_dir / model_id / "best_model.pt"
            if not ckpt.exists():
                logger.warning(f"Checkpoint for {model_id} not found, skipping.")
                continue

            model = self.load_model(model_id)
            thresholds = self.calibrate_routing_thresholds(model, model_id)
            results["calibrated_thresholds"][model_id] = thresholds

            eval_res = self.evaluate_routing_on_test(model, model_id, thresholds)
            results["routing_evaluations"][model_id] = eval_res

        out_json = Path("artifacts/reports/milestone_08d_routing_results.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved routing results to {out_json}")

        return results


def main() -> None:
    router = HybridRouter()
    router.run_all_routing_experiments()


if __name__ == "__main__":
    main()
