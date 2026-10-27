#!/usr/bin/env python3
"""Phase 8 Independent Evaluation and Confirmation for Milestone 08D.

Evaluates frozen uncertainty models and routing policies on the pristine,
untouched independent confirmation cohort:
- artifacts/data/independent_cohort_08d.h5 (200 samples: 100 IID + 100 OOD across 32, 64, 128, 256)
- Seeds: 600000 - 601099 (disjoint from train, val, calibration, and historical confirmation)

Governance Rules:
- Evaluated ONLY after candidate training and calibration are complete.
- Computes empirical coverage, MPIW, NLL, error detection (AUROC, AUPRC, AURC),
  and hybrid routing performance (accepted RMSE, MaxAE, false-accept rate, escalation rate).

Persists results to:
- artifacts/reports/milestone_08d_independent_results.json
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
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
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

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_08d_independent")


def compute_auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    auroc = float(np.trapezoid(tpr, fpr))

    precision = np.r_[1.0, tps / (tps + fps)]
    recall = tpr
    auprc = float(np.trapezoid(precision, recall))
    return auroc, auprc


def compute_aurc(errors: np.ndarray, scores: np.ndarray) -> float:
    n = len(errors)
    if n == 0:
        return 0.0
    order = np.argsort(scores)
    sorted_err = errors[order]
    coverages = np.linspace(0.1, 1.0, 19)
    risks = []
    for c in coverages:
        k = max(1, round(c * n))
        risks.append(float(np.mean(sorted_err[:k])))
    return float(np.trapezoid(risks, coverages))


class IndependentCohortEvaluator:
    def __init__(
        self,
        cohort_path: Path = Path("artifacts/data/independent_cohort_08d.h5"),
        cohort_manifest: Path = Path("artifacts/data/independent_cohort_08d_manifest.json"),
        cal_results_path: Path = Path("artifacts/reports/milestone_08d_calibration_results.json"),
        routing_results_path: Path = Path("artifacts/reports/milestone_08d_routing_results.json"),
        checkpoints_dir: Path = Path("artifacts/checkpoints/milestone_08d"),
        m3_checkpoint: Path = Path(
            "artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt"
        ),
        device: torch.device | None = None,
    ) -> None:
        self.cohort_path = cohort_path
        self.cohort_manifest = cohort_manifest
        self.cal_results_path = cal_results_path
        self.routing_results_path = routing_results_path
        self.checkpoints_dir = checkpoints_dir
        self.m3_checkpoint = m3_checkpoint
        self.device = device or torch.device("cpu")

        torch.set_num_threads(8)
        m3_ckpt = torch.load(self.m3_checkpoint, map_location=self.device, weights_only=False)
        self.normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])
        self.temp_std = float(self.normalizer.t_stats.std)

        # Load calibrated quantiles and routing thresholds
        self.conformal_quantiles: dict[str, Any] = {}
        if self.cal_results_path.exists():
            with open(self.cal_results_path, encoding="utf-8") as f:
                cal_data = json.load(f)
                self.conformal_quantiles = cal_data.get("conformal_calibration", {})

        self.routing_thresholds: dict[str, Any] = {}
        if self.routing_results_path.exists():
            with open(self.routing_results_path, encoding="utf-8") as f:
                rout_data = json.load(f)
                self.routing_thresholds = rout_data.get("calibrated_thresholds", {})

        self.models = self._load_models()

    def _load_models(self) -> dict[str, Any]:
        models: dict[str, Any] = {}

        # U0
        if self.m3_checkpoint.exists():
            u0 = FourierNeuralOperator2D(
                in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4
            )
            st = torch.load(self.m3_checkpoint, map_location=self.device)
            u0.load_state_dict(st.get("model_state_dict", st))
            u0.eval().to(self.device)
            models["U0_baseline_m3"] = u0

        # U1
        u1_p = self.checkpoints_dir / "U1_heteroscedastic" / "best_model.pt"
        if u1_p.exists():
            u1 = HeteroscedasticFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            st = torch.load(u1_p, map_location=self.device)
            u1.load_state_dict(st.get("model_state_dict", st))
            u1.eval().to(self.device)
            models["U1_heteroscedastic"] = u1

        # U2
        u2_p = self.checkpoints_dir / "U2_evidential" / "best_model.pt"
        if u2_p.exists():
            u2 = EvidentialFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            st = torch.load(u2_p, map_location=self.device)
            u2.load_state_dict(st.get("model_state_dict", st))
            u2.eval().to(self.device)
            models["U2_evidential"] = u2

        # U3
        u3_p = self.checkpoints_dir / "U3_mc_dropout" / "best_model.pt"
        if u3_p.exists():
            u3 = MCDropoutFNO2D(modes_x=12, modes_y=12, width=32, depth=4, dropout_p=0.10)
            st = torch.load(u3_p, map_location=self.device)
            u3.load_state_dict(st.get("model_state_dict", st))
            u3.to(self.device)
            models["U3_mcdropout_T3"] = (u3, 3)
            models["U3_mcdropout_T5"] = (u3, 5)
            models["U3_mcdropout_T8"] = (u3, 8)

        # U4
        u4_p = self.checkpoints_dir / "U4_distilled_student" / "best_model.pt"
        if u4_p.exists():
            u4 = DistilledStudentFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            st = torch.load(u4_p, map_location=self.device)
            u4.load_state_dict(st.get("model_state_dict", st))
            u4.eval().to(self.device)
            models["U4_distilled_student"] = u4

        return models

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

    def predict_model(
        self,
        model_name: str,
        x_tensor: torch.Tensor,
        batch_size: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_total = x_tensor.shape[0]
        if n_total > batch_size:
            preds_list = []
            sigmas_list = []
            for start_idx in range(0, n_total, batch_size):
                end_idx = min(start_idx + batch_size, n_total)
                p_chunk, s_chunk = self.predict_model(
                    model_name, x_tensor[start_idx:end_idx], batch_size=n_total
                )
                preds_list.append(p_chunk)
                sigmas_list.append(s_chunk)
            return np.concatenate(preds_list, axis=0), np.concatenate(sigmas_list, axis=0)

        model_entry = self.models[model_name]

        if model_name == "U0_baseline_m3":
            with torch.no_grad():
                pred_norm = model_entry(x_tensor.to(self.device)).cpu().numpy()[:, 0]
            pred_k = self.normalizer.inverse_transform_temperature(pred_norm)
            sigma_k = np.ones_like(pred_k, dtype=np.float32) * 1.5
            return pred_k, sigma_k

        elif model_name == "U1_heteroscedastic":
            with torch.no_grad():
                mean_norm, logvar_norm = model_entry(x_tensor.to(self.device))
                mean_norm = mean_norm.cpu().numpy()[:, 0]
                std_norm = torch.exp(0.5 * logvar_norm).cpu().numpy()[:, 0]
            pred_k = self.normalizer.inverse_transform_temperature(mean_norm)
            sigma_k = std_norm * self.temp_std
            return pred_k, sigma_k

        elif model_name == "U2_evidential":
            with torch.no_grad():
                gamma_norm, v, alpha, beta = model_entry(x_tensor.to(self.device))
                gamma_norm = gamma_norm.cpu().numpy()[:, 0]
                v = v.cpu().numpy()[:, 0]
                alpha = alpha.cpu().numpy()[:, 0]
                beta = beta.cpu().numpy()[:, 0]
            pred_k = self.normalizer.inverse_transform_temperature(gamma_norm)
            denom = np.maximum(alpha - 1.0, 1e-4)
            var_norm = (beta / denom) + (beta / (np.maximum(v, 1e-4) * denom))
            std_norm = np.sqrt(np.maximum(var_norm, 1e-8))
            sigma_k = std_norm * self.temp_std
            return pred_k, sigma_k

        elif model_name.startswith("U3_mcdropout"):
            u3_model, t_samples = model_entry
            u3_model.train()
            preds_norm = []
            with torch.no_grad():
                for _ in range(t_samples):
                    p_norm = u3_model(x_tensor.to(self.device)).cpu().numpy()[:, 0]
                    preds_norm.append(p_norm)
            preds_stack = np.stack(preds_norm, axis=0)
            mean_norm = np.mean(preds_stack, axis=0)
            std_norm = np.std(preds_stack, axis=0, ddof=1)
            pred_k = self.normalizer.inverse_transform_temperature(mean_norm)
            sigma_k = std_norm * self.temp_std
            return pred_k, sigma_k

        elif model_name == "U4_distilled_student":
            with torch.no_grad():
                mean_norm, std_pred_norm = model_entry(x_tensor.to(self.device))
                mean_norm = mean_norm.cpu().numpy()[:, 0]
                std_pred_norm = std_pred_norm.cpu().numpy()[:, 0]
            pred_k = self.normalizer.inverse_transform_temperature(mean_norm)
            sigma_k = np.maximum(std_pred_norm * self.temp_std, 0.05)
            return pred_k, sigma_k

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def run_confirmation_audit(self) -> dict[str, Any]:
        """Audit frozen uncertainty models and routing on independent confirmation cohort."""
        logger.info("Executing Phase 8 Independent Evaluation on locked cohort...")

        with open(self.cohort_manifest, encoding="utf-8") as f:
            manifest_data = json.load(f)

        results: dict[str, Any] = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cohort_file": str(self.cohort_path),
                "cohort_sha256": manifest_data.get("sha256"),
                "cohort_governance": manifest_data.get("governance_status"),
                "models_audited": list(self.models.keys()),
            },
            "confirmation_results": {},
            "routing_confirmation": {},
        }

        resolutions = [32, 64, 128, 256]

        with h5py.File(self.cohort_path, "r") as f:
            for m_name in self.models:
                m_results: dict[str, Any] = {}
                q_dict = self.conformal_quantiles.get(m_name, {})
                q_pt_95 = q_dict.get("comb_ptwise_95", 1.96)
                q_sim_95 = q_dict.get("comb_simult_95", 3.0)

                for res in resolutions:
                    res_key = f"res_{res}"
                    m_results[res_key] = {}

                    for split_name in ["eval_iid", "eval_ood"]:
                        grp = f[f"{res_key}/{split_name}"]
                        x_arr, k_all, q_all, t_true, bcs_list = self._prepare_inputs(grp, res)
                        pred_k, unc_k = self.predict_model(m_name, x_arr)

                        n_samples = len(t_true)
                        residuals = np.abs(t_true - pred_k)
                        sample_rmse = np.sqrt(np.mean(residuals**2, axis=(1, 2)))
                        sample_maxae = np.max(residuals, axis=(1, 2))
                        sample_max_unc = np.max(unc_k, axis=(1, 2))
                        sample_mean_unc = np.mean(unc_k, axis=(1, 2))

                        mean_rmse = float(np.mean(sample_rmse))
                        mean_maxae = float(np.mean(sample_maxae))
                        max_maxae = float(np.max(sample_maxae))

                        # Coverage at pre-calibrated 95%
                        pt_covered_95 = residuals <= (q_pt_95 * unc_k)
                        sim_covered_95 = np.all(residuals <= (q_sim_95 * unc_k), axis=(1, 2))
                        pt_cov_rate = float(np.mean(pt_covered_95))
                        sim_cov_rate = float(np.mean(sim_covered_95))
                        mpiw = float(np.mean(2.0 * q_pt_95 * unc_k))

                        # Error detection
                        y_5k = (sample_maxae > 5.0).astype(int)
                        y_10k = (sample_maxae > 10.0).astype(int)
                        auroc_5k, auprc_5k = compute_auroc_auprc(y_5k, sample_max_unc)
                        auroc_10k, auprc_10k = compute_auroc_auprc(y_10k, sample_max_unc)
                        aurc_rmse = compute_aurc(sample_rmse, sample_mean_unc)
                        aurc_maxae = compute_aurc(sample_maxae, sample_max_unc)

                        from scipy.stats import spearmanr

                        rho_maxae, _ = spearmanr(sample_max_unc, sample_maxae)

                        m_results[res_key][split_name] = {
                            "n_samples": n_samples,
                            "mean_rmse_k": mean_rmse,
                            "mean_maxae_k": mean_maxae,
                            "max_maxae_k": max_maxae,
                            "mpiw_95_k": mpiw,
                            "empirical_coverage_ptwise_95": pt_cov_rate,
                            "empirical_coverage_simult_95": sim_cov_rate,
                            "spearman_rho_maxae": float(rho_maxae)
                            if not np.isnan(rho_maxae)
                            else 0.0,
                            "auroc_5k": auroc_5k,
                            "auprc_5k": auprc_5k,
                            "auroc_10k": auroc_10k,
                            "auprc_10k": auprc_10k,
                            "aurc_rmse": aurc_rmse,
                            "aurc_maxae": aurc_maxae,
                        }

                results["confirmation_results"][m_name] = m_results

            # Routing evaluation on independent cohort for top candidates
            routing_model_map = {
                "U1_heteroscedastic": "U1_heteroscedastic",
                "U2_evidential": "U2_evidential",
                "U3_mc_dropout": "U3_mcdropout_T5",
                "U4_distilled_student": "U4_distilled_student",
            }
            for model_id, eval_m_name in routing_model_map.items():
                if eval_m_name not in self.models or model_id not in self.routing_thresholds:
                    continue

                th = self.routing_thresholds[model_id]
                tau_unc = th["tau_unc"]
                tau_k = th["tau_k"]
                tau_res = th["tau_res"]

                r_conf: dict[str, Any] = {}
                for res in resolutions:
                    grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
                    res_key = f"res_{res}"
                    r_conf[res_key] = {}

                    for split_name in ["eval_iid", "eval_ood"]:
                        grp = f[f"{res_key}/{split_name}"]
                        x_arr, k_all, q_all, t_true, bcs_list = self._prepare_inputs(grp, res)
                        pred_k, unc_k = self.predict_model(eval_m_name, x_arr)

                        accepted_flags = []
                        surr_maxaes = []
                        surr_rmses = []
                        hyb_rmses = []

                        for i in range(len(t_true)):
                            max_unc = float(np.max(unc_k[i]))
                            c_k = float(np.max(k_all[i]) / np.maximum(np.min(k_all[i]), 1e-5))
                            diag = compute_heat_balance(
                                grid, pred_k[i], k_all[i], q_all[i], bcs_list[i]
                            )
                            res_norm = float(diag.relative_imbalance * 100.0)

                            accept = (
                                (max_unc <= tau_unc) and (c_k <= tau_k) and (res_norm <= tau_res)
                            )
                            accepted_flags.append(accept)

                            diff = np.abs(t_true[i] - pred_k[i])
                            m_ae = float(np.max(diff))
                            rmse = float(np.sqrt(np.mean(diff**2)))
                            surr_maxaes.append(m_ae)
                            surr_rmses.append(rmse)

                            if accept:
                                hyb_rmses.append(rmse)
                            else:
                                hyb_rmses.append(0.0)

                        arr_acc = np.array(accepted_flags)
                        arr_maxae = np.array(surr_maxaes)
                        arr_rmse = np.array(surr_rmses)
                        n_acc = int(np.sum(arr_acc))
                        acc_rate = float(np.mean(arr_acc))

                        if n_acc > 0:
                            false_accept_rate = float(np.sum(arr_acc & (arr_maxae > 5.0)) / n_acc)
                            accepted_mean_rmse = float(np.mean(arr_rmse[arr_acc]))
                            accepted_mean_maxae = float(np.mean(arr_maxae[arr_acc]))
                        else:
                            false_accept_rate = 0.0
                            accepted_mean_rmse = 0.0
                            accepted_mean_maxae = 0.0

                        r_conf[res_key][split_name] = {
                            "acceptance_rate": acc_rate,
                            "escalation_rate": float(1.0 - acc_rate),
                            "false_accept_rate": false_accept_rate,
                            "accepted_mean_rmse_k": accepted_mean_rmse,
                            "accepted_mean_maxae_k": accepted_mean_maxae,
                            "hybrid_mean_rmse_k": float(np.mean(hyb_rmses)),
                        }

                results["routing_confirmation"][model_id] = r_conf

        out_json = Path("artifacts/reports/milestone_08d_independent_results.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as out_f:
            json.dump(results, out_f, indent=2)
        logger.info(f"Saved independent confirmation results to {out_json}")

        return results


def main() -> None:
    evaluator = IndependentCohortEvaluator()
    evaluator.run_confirmation_audit()


if __name__ == "__main__":
    main()
