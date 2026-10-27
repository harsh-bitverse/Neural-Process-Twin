#!/usr/bin/env python3
"""Statistical Calibration and Error-Detection Evaluation for Milestone 08D.

Milestone 08D Phases 4 & 5:
- Phase 4: Statistical calibration and interval validity (Conformal Prediction, NLL, MPIW, ECE).
- Phase 5: Error detection and risk-coverage analysis (AUROC, AUPRC, AURC, False-accept/reject rates).

Evaluates candidates:
- U0: Baseline M3 (deterministic with residual heuristic / physical residual proxy)
- U1: Heteroscedastic FNO (Single-pass heteroscedastic scale head)
- U2: Evidential FNO (Deep Evidential Regression)
- U3: MC Dropout FNO (T=3, T=5, T=8 stochastic forward passes)
- U4: Distilled Student FNO (Single-pass epistemic student matching ensemble residual scale)
- Ens5: Historical 5-member ensemble (evaluated on 32x32)

Persists results to:
- artifacts/reports/milestone_08d_calibration_results.json
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
from neural_process_twin.solvers.grid import Grid2D

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_08d_calibration")


def compute_auroc_auprc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute AUROC and AUPRC without external dependencies."""
    n_pos = int(np.sum(y_true))
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    # Sort descending by score
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    # AUROC via trapezoid / Mann-Whitney U
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend (0,0)
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    auroc = float(np.trapezoid(tpr, fpr))

    # AUPRC
    precision = tps / (tps + fps)
    recall = tpr[1:]
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    auprc = float(np.trapezoid(precision, recall))

    return auroc, auprc


def compute_aurc(errors: np.ndarray, scores: np.ndarray) -> float:
    """Compute Area Under Risk-Coverage Curve (AURC).

    Lower AURC is better: samples with lowest uncertainty scores are retained first.
    """
    n = len(errors)
    if n == 0:
        return 0.0

    order = np.argsort(scores)  # Ascending uncertainty
    sorted_err = errors[order]

    coverages = np.linspace(0.1, 1.0, 19)
    risks = []
    for c in coverages:
        k = max(1, round(c * n))
        risks.append(float(np.mean(sorted_err[:k])))

    # Integrate risk over coverage
    aurc = float(np.trapezoid(risks, coverages))
    return aurc


def compute_split_conformal_quantiles(
    residuals: np.ndarray | list[np.ndarray],
    uncertainties: np.ndarray | list[np.ndarray],
    alpha: float = 0.05,
    eps: float = 1e-4,
) -> tuple[float, float]:
    """Compute split conformal quantile for pointwise and simultaneous bands across resolutions."""
    if isinstance(residuals, np.ndarray):
        res_list = [residuals]
        unc_list = [uncertainties]
    else:
        res_list = residuals
        unc_list = uncertainties

    ptwise_scores = []
    simult_scores = []

    for r, u in zip(res_list, unc_list, strict=False):
        safe_u = np.maximum(u, eps)
        scaled = r / safe_u
        ptwise_scores.append(scaled.flatten())
        simult_scores.append(np.max(scaled, axis=(1, 2)))

    flat_scaled = np.concatenate(ptwise_scores)
    sample_max_scaled = np.concatenate(simult_scores)
    n_cal = len(sample_max_scaled)

    # Pointwise nonconformity
    p_target = min(1.0, np.ceil((len(flat_scaled) + 1) * (1.0 - alpha)) / len(flat_scaled))
    q_ptwise = float(np.quantile(flat_scaled, p_target, method="higher"))

    # Simultaneous (sample max)
    p_sim_target = min(1.0, np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)
    q_simult = float(np.quantile(sample_max_scaled, p_sim_target, method="higher"))

    return q_ptwise, q_simult


def compute_ece(residuals: np.ndarray, uncertainties: np.ndarray, eps: float = 1e-4) -> float:
    """Compute Expected Calibration Error across standard normal quantiles."""
    safe_unc = np.maximum(uncertainties, eps)
    norm_res = (residuals / safe_unc).flatten()

    # Theoretical normal quantiles for coverage levels 0.1 to 0.95
    conf_levels = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    from scipy.stats import norm

    ece_acc = 0.0
    for p in conf_levels:
        z = norm.ppf((1.0 + p) / 2.0)
        empirical_p = np.mean(norm_res <= z)
        ece_acc += abs(p - empirical_p)

    return float(ece_acc / len(conf_levels))


class CalibrationEvaluator:
    """Evaluates uncertainty models, fits conformal quantiles, and audits test sets."""

    def __init__(
        self,
        cal_path: Path = Path("artifacts/data/multires_calibration.h5"),
        test_path: Path = Path("artifacts/data/multires_dataset.h5"),
        checkpoints_dir: Path = Path("artifacts/checkpoints/milestone_08d"),
        m3_checkpoint: Path = Path(
            "artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt"
        ),
        device: torch.device | None = None,
    ) -> None:
        self.cal_path = cal_path
        self.test_path = test_path
        self.checkpoints_dir = checkpoints_dir
        self.m3_checkpoint = m3_checkpoint
        self.device = device or torch.device("cpu")

        torch.set_num_threads(8)
        m3_ckpt = torch.load(self.m3_checkpoint, map_location="cpu", weights_only=False)
        self.normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])
        self.temp_std = float(self.normalizer.t_stats.std)

        # Load models
        self.models = self._load_all_models()

    def _load_all_models(self) -> dict[str, Any]:
        models: dict[str, Any] = {}

        # U0: Baseline M3
        if self.m3_checkpoint.exists():
            u0 = FourierNeuralOperator2D(
                in_channels=12,
                out_channels=1,
                modes_x=12,
                modes_y=12,
                width=32,
                depth=4,
            )
            state = torch.load(self.m3_checkpoint, map_location=self.device)
            u0.load_state_dict(state.get("model_state_dict", state))
            u0.eval().to(self.device)
            models["U0_baseline_m3"] = u0
            logger.info("Loaded U0 baseline M3.")

        # U1: Heteroscedastic FNO
        u1_path = self.checkpoints_dir / "U1_heteroscedastic" / "best_model.pt"
        if u1_path.exists():
            u1 = HeteroscedasticFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            state = torch.load(u1_path, map_location=self.device)
            u1.load_state_dict(state.get("model_state_dict", state))
            u1.eval().to(self.device)
            models["U1_heteroscedastic"] = u1
            logger.info("Loaded U1 Heteroscedastic FNO.")

        # U2: Evidential FNO
        u2_path = self.checkpoints_dir / "U2_evidential" / "best_model.pt"
        if u2_path.exists():
            u2 = EvidentialFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            state = torch.load(u2_path, map_location=self.device)
            u2.load_state_dict(state.get("model_state_dict", state))
            u2.eval().to(self.device)
            models["U2_evidential"] = u2
            logger.info("Loaded U2 Evidential FNO.")

        # U3: MC Dropout FNO
        u3_path = self.checkpoints_dir / "U3_mc_dropout" / "best_model.pt"
        if u3_path.exists():
            u3 = MCDropoutFNO2D(modes_x=12, modes_y=12, width=32, depth=4, dropout_p=0.10)
            state = torch.load(u3_path, map_location=self.device)
            u3.load_state_dict(state.get("model_state_dict", state))
            u3.to(self.device)
            models["U3_mcdropout_T3"] = (u3, 3)
            models["U3_mcdropout_T5"] = (u3, 5)
            models["U3_mcdropout_T8"] = (u3, 8)
            logger.info("Loaded U3 MC Dropout FNO (T=3, 5, 8).")

        # U4: Distilled Student FNO
        u4_path = self.checkpoints_dir / "U4_distilled_student" / "best_model.pt"
        if u4_path.exists():
            u4 = DistilledStudentFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            state = torch.load(u4_path, map_location=self.device)
            u4.load_state_dict(state.get("model_state_dict", state))
            u4.eval().to(self.device)
            models["U4_distilled_student"] = u4
            logger.info("Loaded U4 Distilled Student FNO.")

        # Ensemble5 (32x32 only)
        ens_models = []
        for seed in [42, 43, 44, 45, 46]:
            ckpt = Path(f"artifacts/checkpoints/scaled_fno_supervised_seed{seed}.pt")
            if ckpt.exists():
                m = FourierNeuralOperator2D(
                    in_channels=12,
                    out_channels=1,
                    modes_x=12,
                    modes_y=12,
                    width=32,
                    depth=4,
                )
                st = torch.load(ckpt, map_location=self.device)
                m.load_state_dict(st.get("model_state_dict", st))
                m.eval().to(self.device)
                ens_models.append(m)
        if len(ens_models) == 5:
            models["Ensemble5"] = ens_models
            logger.info("Loaded 5-member ensemble.")

        return models

    def _prepare_inputs(
        self, grp: h5py.Group, res: int
    ) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Extract and format 12-channel inputs and ground-truth fields."""
        n_samples = grp["k"].shape[0]
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)

        k_all = grp["k"][:]
        q_all = grp["q"][:]
        t_all = grp["temperature"][:]
        if "subgroup" in grp:
            subgroups = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in grp["subgroup"][:]
            ]
        else:
            subgroups = ["unspecified"] * n_samples

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

        return torch.from_numpy(x_arr), k_all, q_all, t_all, subgroups

    def predict_model(
        self,
        model_name: str,
        x_tensor: torch.Tensor,
        res: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate predicted temperature (Kelvin) and uncertainty standard deviation (Kelvin)."""
        model_entry = self.models[model_name]
        n_samples = x_tensor.shape[0]

        if model_name == "U0_baseline_m3":
            with torch.no_grad():
                pred_norm = model_entry(x_tensor.to(self.device)).cpu().numpy()[:, 0]
            pred_k = self.normalizer.inverse_transform_temperature(pred_norm)
            # U0 uncertainty proxy: constant residual scale (1.5 K default)
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
            # Total variance = aleatoric + epistemic = beta/(alpha-1) + beta/(v*(alpha-1))
            denom = np.maximum(alpha - 1.0, 1e-4)
            var_norm = (beta / denom) + (beta / (np.maximum(v, 1e-4) * denom))
            std_norm = np.sqrt(np.maximum(var_norm, 1e-8))
            sigma_k = std_norm * self.temp_std
            return pred_k, sigma_k

        elif model_name.startswith("U3_mcdropout"):
            u3_model, t_samples = model_entry
            u3_model.train()  # Keep dropout active!
            preds_norm = []
            with torch.no_grad():
                for _ in range(t_samples):
                    p_norm = u3_model(x_tensor.to(self.device)).cpu().numpy()[:, 0]
                    preds_norm.append(p_norm)
            preds_stack = np.stack(preds_norm, axis=0)  # (T, N, H, W)
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

        elif model_name == "Ensemble5":
            if res != 32:
                # 5-member ensemble was 32x32 only
                return np.zeros((n_samples, res, res), dtype=np.float32), np.zeros(
                    (n_samples, res, res), dtype=np.float32
                )
            preds_norm = []
            with torch.no_grad():
                for m in model_entry:
                    p = m(x_tensor.to(self.device)).cpu().numpy()[:, 0]
                    preds_norm.append(p)
            preds_stack = np.stack(preds_norm, axis=0)  # (5, N, 32, 32)
            mean_norm = np.mean(preds_stack, axis=0)
            std_norm = np.std(preds_stack, axis=0, ddof=1)
            pred_k = self.normalizer.inverse_transform_temperature(mean_norm)
            sigma_k = std_norm * self.temp_std
            return pred_k, sigma_k

        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def run_full_calibration_audit(self) -> dict[str, Any]:
        """Fit conformal quantiles and evaluate comprehensive UQ metrics across resolutions."""
        results: dict[str, Any] = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "calibration_dataset": str(self.cal_path),
                "test_dataset": str(self.test_path),
                "models_evaluated": list(self.models.keys()),
            },
            "conformal_calibration": {},
            "evaluation_results": {},
            "summary_rankings": {},
        }

        # ---------------------------------------------------------------------
        # 1. Fit Conformal Calibration Quantiles
        # ---------------------------------------------------------------------
        logger.info("--- Step 1: Fitting Conformal Calibration Quantiles ---")
        conformal_quantiles: dict[str, dict[str, float]] = {}

        with h5py.File(self.cal_path, "r") as cal_f:
            # We fit quantiles across calibration resolutions [32, 64]
            # using both iid-only (100) and combined iid+ood (200)
            for m_name in self.models:
                cal_res_list = [32] if m_name == "Ensemble5" else [32, 64]

                all_cal_res_iid = []
                all_cal_unc_iid = []
                all_cal_res_comb = []
                all_cal_unc_comb = []

                for res in cal_res_list:
                    grp_iid = cal_f[f"res_{res}/calibration_iid"]
                    x_iid, _, _, t_iid, _ = self._prepare_inputs(grp_iid, res)
                    pred_iid, unc_iid = self.predict_model(m_name, x_iid, res)
                    res_iid = np.abs(t_iid - pred_iid)

                    all_cal_res_iid.append(res_iid)
                    all_cal_unc_iid.append(unc_iid)

                    grp_ood = cal_f[f"res_{res}/calibration_ood"]
                    x_ood, _, _, t_ood, _ = self._prepare_inputs(grp_ood, res)
                    pred_ood, unc_ood = self.predict_model(m_name, x_ood, res)
                    res_ood = np.abs(t_ood - pred_ood)

                    all_cal_res_comb.extend([res_iid, res_ood])
                    all_cal_unc_comb.extend([unc_iid, unc_ood])

                q_pt_90_iid, q_sim_90_iid = compute_split_conformal_quantiles(
                    all_cal_res_iid, all_cal_unc_iid, alpha=0.10
                )
                q_pt_95_iid, q_sim_95_iid = compute_split_conformal_quantiles(
                    all_cal_res_iid, all_cal_unc_iid, alpha=0.05
                )

                q_pt_90_comb, q_sim_90_comb = compute_split_conformal_quantiles(
                    all_cal_res_comb, all_cal_unc_comb, alpha=0.10
                )
                q_pt_95_comb, q_sim_95_comb = compute_split_conformal_quantiles(
                    all_cal_res_comb, all_cal_unc_comb, alpha=0.05
                )

                conformal_quantiles[m_name] = {
                    "iid_ptwise_90": q_pt_90_iid,
                    "iid_ptwise_95": q_pt_95_iid,
                    "iid_simult_90": q_sim_90_iid,
                    "iid_simult_95": q_sim_95_iid,
                    "comb_ptwise_90": q_pt_90_comb,
                    "comb_ptwise_95": q_pt_95_comb,
                    "comb_simult_90": q_sim_90_comb,
                    "comb_simult_95": q_sim_95_comb,
                }
                logger.info(
                    f"Model {m_name}: IID 95% pt={q_pt_95_iid:.2f}, sim={q_sim_95_iid:.2f} | "
                    f"Comb 95% pt={q_pt_95_comb:.2f}, sim={q_sim_95_comb:.2f}"
                )

        results["conformal_calibration"] = conformal_quantiles

        # ---------------------------------------------------------------------
        # 2. Evaluate on Test Sets across All Resolutions (32, 64, 128, 256)
        # ---------------------------------------------------------------------
        logger.info("--- Step 2: Evaluating on Test Sets (IID and OOD) ---")
        resolutions = [32, 64, 128, 256]

        with h5py.File(self.test_path, "r") as test_f:
            for m_name in self.models:
                q_dict = conformal_quantiles[m_name]
                model_res_dict: dict[str, Any] = {}

                for res in resolutions:
                    if m_name == "Ensemble5" and res != 32:
                        continue

                    res_key = f"res_{res}"
                    model_res_dict[res_key] = {}

                    for split_name in ["test_iid", "test_ood"]:
                        grp = test_f[f"{res_key}/{split_name}"]
                        x_arr, _k_all, _q_all, t_true, _subgroups = self._prepare_inputs(grp, res)
                        pred_k, unc_k = self.predict_model(m_name, x_arr, res)

                        n_samples = len(t_true)
                        residuals = np.abs(t_true - pred_k)  # (N, H, W)
                        sample_rmse = np.sqrt(np.mean(residuals**2, axis=(1, 2)))
                        sample_maxae = np.max(residuals, axis=(1, 2))
                        sample_mean_unc = np.mean(unc_k, axis=(1, 2))
                        sample_max_unc = np.max(unc_k, axis=(1, 2))

                        # Overall accuracy
                        mean_rmse = float(np.mean(sample_rmse))
                        median_rmse = float(np.median(sample_rmse))
                        mean_maxae = float(np.mean(sample_maxae))
                        max_maxae = float(np.max(sample_maxae))

                        # Empirical Coverages using Combined 95% quantiles
                        q_pt_95 = q_dict["comb_ptwise_95"]
                        q_sim_95 = q_dict["comb_simult_95"]
                        q_dict["comb_ptwise_90"]
                        q_dict["comb_simult_90"]

                        # Also measure with IID-only quantiles to inspect OOD coverage breakdown
                        q_pt_95_iid = q_dict["iid_ptwise_95"]
                        q_sim_95_iid = q_dict["iid_simult_95"]

                        half_w_pt_95 = q_pt_95 * unc_k
                        half_w_sim_95 = q_sim_95 * unc_k
                        half_w_pt_95_iid = q_pt_95_iid * unc_k

                        pt_covered_95 = residuals <= half_w_pt_95
                        sim_covered_95 = np.all(residuals <= half_w_sim_95, axis=(1, 2))
                        pt_covered_95_iid = residuals <= half_w_pt_95_iid

                        pt_cov_rate_95 = float(np.mean(pt_covered_95))
                        sim_cov_rate_95 = float(np.mean(sim_covered_95))
                        pt_cov_rate_95_iid = float(np.mean(pt_covered_95_iid))
                        mpiw_95 = float(np.mean(2.0 * half_w_pt_95))

                        # Gaussian NLL
                        safe_unc = np.maximum(unc_k, 1e-4)
                        nll_field = 0.5 * np.log(2 * np.pi * (safe_unc**2)) + (residuals**2) / (
                            2 * (safe_unc**2)
                        )
                        mean_nll = float(np.mean(nll_field))

                        # ECE
                        ece = compute_ece(residuals, unc_k)

                        # Correlations
                        from scipy.stats import pearsonr, spearmanr

                        r_rmse, _ = pearsonr(sample_mean_unc, sample_rmse)
                        rho_rmse, _ = spearmanr(sample_mean_unc, sample_rmse)
                        r_maxae, _ = pearsonr(sample_max_unc, sample_maxae)
                        rho_maxae, _ = spearmanr(sample_max_unc, sample_maxae)

                        # Error detection AUROC / AUPRC
                        y_5k = (sample_maxae > 5.0).astype(int)
                        y_10k = (sample_maxae > 10.0).astype(int)

                        auroc_5k, auprc_5k = compute_auroc_auprc(y_5k, sample_max_unc)
                        auroc_10k, auprc_10k = compute_auroc_auprc(y_10k, sample_max_unc)

                        # AURC
                        aurc_rmse = compute_aurc(sample_rmse, sample_mean_unc)
                        aurc_maxae = compute_aurc(sample_maxae, sample_max_unc)

                        # Decision threshold metrics for tau = 5.0 K
                        # False accept: sample has MaxAE > 5.0 K but sample_max_unc <= threshold
                        # Set threshold to 90th percentile of accurate sample uncertainties
                        acc_mask = sample_maxae <= 5.0
                        if np.sum(acc_mask) > 5 and np.sum(~acc_mask) > 5:
                            tau_unc = float(np.quantile(sample_max_unc[acc_mask], 0.90))
                            accepted = sample_max_unc <= tau_unc
                            acceptance_rate = float(np.mean(accepted))
                            # False accept rate: proportion of accepted samples that were inaccurate
                            n_accepted = np.sum(accepted)
                            if n_accepted > 0:
                                false_accept_rate = float(np.sum(accepted & ~acc_mask) / n_accepted)
                            else:
                                false_accept_rate = 0.0
                            # False reject rate: proportion of accurate samples that were rejected
                            false_reject_rate = float(
                                np.sum(~accepted & acc_mask) / np.sum(acc_mask)
                            )
                        else:
                            tau_unc = float("nan")
                            acceptance_rate = 1.0
                            false_accept_rate = 0.0
                            false_reject_rate = 0.0

                        model_res_dict[res_key][split_name] = {
                            "n_samples": n_samples,
                            "mean_rmse_k": mean_rmse,
                            "median_rmse_k": median_rmse,
                            "mean_maxae_k": mean_maxae,
                            "max_maxae_k": max_maxae,
                            "mpiw_95_k": mpiw_95,
                            "empirical_coverage_ptwise_95": pt_cov_rate_95,
                            "empirical_coverage_simult_95": sim_cov_rate_95,
                            "empirical_coverage_ptwise_95_with_iid_cal": pt_cov_rate_95_iid,
                            "nll": mean_nll,
                            "ece": ece,
                            "pearson_r_rmse": float(r_rmse) if not np.isnan(r_rmse) else 0.0,
                            "spearman_rho_rmse": float(rho_rmse) if not np.isnan(rho_rmse) else 0.0,
                            "pearson_r_maxae": float(r_maxae) if not np.isnan(r_maxae) else 0.0,
                            "spearman_rho_maxae": float(rho_maxae)
                            if not np.isnan(rho_maxae)
                            else 0.0,
                            "auroc_maxae_5k": auroc_5k,
                            "auprc_maxae_5k": auprc_5k,
                            "auroc_maxae_10k": auroc_10k,
                            "auprc_maxae_10k": auprc_10k,
                            "aurc_rmse": aurc_rmse,
                            "aurc_maxae": aurc_maxae,
                            "acceptance_rate_tau5k": acceptance_rate,
                            "false_accept_rate_tau5k": false_accept_rate,
                            "false_reject_rate_tau5k": false_reject_rate,
                        }

                results["evaluation_results"][m_name] = model_res_dict

        # ---------------------------------------------------------------------
        # 3. Compute Summary Rankings
        # ---------------------------------------------------------------------
        summary = {}
        for m_name in self.models:
            m_res = results["evaluation_results"][m_name]
            all_rho = []
            all_cov = []
            all_cov_iid = []
            all_mpiw = []
            all_auroc = []
            all_nll = []

            for _r_k, r_val in m_res.items():
                for _s_k, s_val in r_val.items():
                    all_rho.append(s_val["spearman_rho_maxae"])
                    all_cov.append(s_val["empirical_coverage_ptwise_95"])
                    all_cov_iid.append(s_val["empirical_coverage_ptwise_95_with_iid_cal"])
                    all_mpiw.append(s_val["mpiw_95_k"])
                    if not np.isnan(s_val["auroc_maxae_5k"]):
                        all_auroc.append(s_val["auroc_maxae_5k"])
                    all_nll.append(s_val["nll"])

            summary[m_name] = {
                "mean_spearman_rho_maxae": float(np.mean(all_rho)),
                "mean_ptwise_coverage_comb95": float(np.mean(all_cov)),
                "mean_ptwise_coverage_iid95": float(np.mean(all_cov_iid)),
                "mean_mpiw_95_k": float(np.mean(all_mpiw)),
                "mean_auroc_5k": float(np.mean(all_auroc)) if all_auroc else 0.0,
                "mean_nll": float(np.mean(all_nll)),
            }

        results["summary_rankings"] = summary

        # Persist report JSON
        out_json = Path("artifacts/reports/milestone_08d_calibration_results.json")
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved calibration results to {out_json}")

        return results


def main() -> None:
    evaluator = CalibrationEvaluator()
    evaluator.run_full_calibration_audit()


if __name__ == "__main__":
    main()
