"""Milestone 08G Phase 4: Predict Localized Errors and Evaluate Uncertainty Signals.

Evaluates 6 uncertainty / residual signals on their ability to detect high-error cases (MaxAE > 10 K and MaxAE > 5 K):
1. U1 (Heteroscedastic scale max)
2. U3_T3 (MC Dropout T=3 std max)
3. U3_T5 (MC Dropout T=5 std max)
4. U_res (Learned residual predictor max error)
5. U_phys_bnd (Discrete boundary flux balance relative error)
6. U_phys_res (Linear system relative residual norm)

Computes:
- Pearson and Spearman rank correlation with actual MaxAE
- AUROC and AUPRC for MaxAE > 10 K and MaxAE > 5 K
- Risk-coverage curves across coverage percentiles (10% to 100%)
- False-accept rate over all queries and conditional failure rate among accepted

Outputs: artifacts/reports/milestone_08g_error_detection.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn

from neural_process_twin.data.continuous_encoding import encode_continuous_physical_inputs
from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.refined_fno import RefinedFNO2D
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
logger = logging.getLogger("benchmark_08g_error_detection")


def compute_file_sha256(path: Path) -> str:
    if not path.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_bcs_row(grp: h5py.Group, idx: int) -> BoundaryConditions:
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


# --- Model Definitions ---
class A3HeteroscedasticHead(nn.Module):
    def __init__(self, in_channels: int = 13, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, padding_mode="replicate"),
        )

    def forward(self, x_in: torch.Tensor, t_pred_norm: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([x_in, t_pred_norm], dim=1)
        logvar = self.net(feat)
        return torch.clamp(logvar, min=-12.0, max=8.0)


class A3ResidualPredictor(nn.Module):
    def __init__(self, in_channels: int = 13, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.Softplus(),
        )

    def forward(self, x_in: torch.Tensor, t_pred_norm: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([x_in, t_pred_norm], dim=1)
        return self.net(feat)


def sample_mc_dropout(
    a3_model: RefinedFNO2D,
    x: torch.Tensor,
    num_passes: int = 3,
    dropout_rate: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MC dropout on refinement layers and compute mean and std."""
    preds = []
    y_fno = a3_model.fno(x)
    feat_combined = torch.cat([x, y_fno], dim=1)

    for _ in range(num_passes):
        h = feat_combined
        for layer in a3_model.local_refinement.net:
            h = layer(h)
            if isinstance(layer, nn.GELU):
                h = nn.functional.dropout2d(h, p=dropout_rate, training=True)
        delta_y = h
        preds.append(y_fno + delta_y)

    stack = torch.stack(preds, dim=0)
    mean = torch.mean(stack, dim=0)
    std = torch.std(stack, dim=0, unbiased=True)
    return mean, std


def compute_auroc_auprc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return 0.5, float(np.mean(labels))

    order = np.argsort(-scores)
    sorted_labels = labels[order]

    n_pos = np.sum(sorted_labels == 1)
    n_neg = np.sum(sorted_labels == 0)

    tps = np.cumsum(sorted_labels == 1)
    fps = np.cumsum(sorted_labels == 0)

    tprs = tps / n_pos
    fprs = fps / n_neg
    precisions = tps / (tps + fps)

    fpr_diff = np.diff(np.concatenate(([0.0], fprs)))
    auroc = float(np.sum(fpr_diff * tprs))

    rec_diff = np.diff(np.concatenate(([0.0], tprs)))
    auprc = float(np.sum(rec_diff * precisions))

    return auroc, auprc


def compute_correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 0.0
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    denom = np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2))
    pearson = float(np.sum(vx * vy) / denom) if denom > 1e-12 else 0.0

    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    vrx = rx - np.mean(rx)
    vry = ry - np.mean(ry)
    denom_r = np.sqrt(np.sum(vrx**2)) * np.sqrt(np.sum(vry**2))
    spearman = float(np.sum(vrx * vry) / denom_r) if denom_r > 1e-12 else 0.0

    return pearson, spearman


def compute_risk_coverage(
    scores: np.ndarray, actual_errors: np.ndarray, unsafe_thresh: float = 10.0
) -> list[dict[str, float]]:
    order = np.argsort(scores)
    n = len(scores)
    curve = []
    coverage_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for frac in coverage_fractions:
        k = max(1, int(np.round(frac * n)))
        accepted_idx = order[:k]
        accepted_errors = actual_errors[accepted_idx]
        unsafe_count = int(np.sum(accepted_errors > unsafe_thresh))
        curve.append(
            {
                "coverage": frac,
                "n_accepted": k,
                "mean_maxae": float(np.mean(accepted_errors)),
                "p90_maxae": float(np.percentile(accepted_errors, 90)),
                "unsafe_count": unsafe_count,
                "conditional_failure_rate": float(unsafe_count / k),
            }
        )
    return curve


def evaluate_split_signals(
    a3_model: RefinedFNO2D,
    het_head: A3HeteroscedasticHead,
    res_pred: A3ResidualPredictor,
    normalizer: FieldNormalizer,
    log_k_mean: float,
    log_k_std: float,
    h5_path: Path,
    resolution: int,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    """Extract actual errors and 6 uncertainty scores for a given split."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, resolution, resolution, thickness=1.0)

    with h5py.File(h5_path, "r") as f:
        grp = f[f"res_{resolution}/{split}"]
        n_samples = grp["temperature"].shape[0]
        k_all = grp["k"][:]
        q_all = grp["q"][:]
        t_all = grp["temperature"][:]
        bcs_all = [parse_bcs_row(grp, i) for i in range(n_samples)]

    actual_maxaes = []
    actual_rmses = []

    s_het = []
    s_mcd3 = []
    s_mcd5 = []
    s_res = []
    s_bnd = []
    s_lin = []

    for i in range(n_samples):
        k_f = k_all[i]
        q_f = q_all[i]
        t_true = t_all[i]
        bcs = bcs_all[i]

        x_in = encode_continuous_physical_inputs(
            k_field=k_f,
            q_field=q_f,
            bcs=bcs,
            grid=grid,
            normalizer=normalizer,
            use_log_k=True,
            log_k_mean=log_k_mean,
            log_k_std=log_k_std,
        )
        x_tensor = torch.from_numpy(x_in).unsqueeze(0).to(device)

        with torch.no_grad():
            out_norm = a3_model(x_tensor)
            t_pred = normalizer.inverse_transform_temperature(out_norm.cpu().numpy()[0, 0])

            # 1. U1 Heteroscedastic scale
            logvar = het_head(x_tensor, out_norm)
            var_norm = torch.exp(logvar).cpu().numpy()[0, 0]
            # scale standard deviation back to Kelvin using t_std
            t_scale = float(normalizer.t_std) if hasattr(normalizer, "t_std") else 1.0
            sigma_k = np.sqrt(var_norm) * t_scale
            u1_val = float(np.max(sigma_k))

            # 2. U3 MC Dropout T=3
            _, std_t3 = sample_mc_dropout(a3_model, x_tensor, num_passes=3)
            std_t3_k = std_t3.cpu().numpy()[0, 0] * t_scale
            u3_t3_val = float(np.max(std_t3_k))

            # 3. U3 MC Dropout T=5
            _, std_t5 = sample_mc_dropout(a3_model, x_tensor, num_passes=5)
            std_t5_k = std_t5.cpu().numpy()[0, 0] * t_scale
            u3_t5_val = float(np.max(std_t5_k))

            # 4. U_res Learned Residual Predictor
            e_hat = res_pred(x_tensor, out_norm).cpu().numpy()[0, 0]
            u_res_val = float(np.max(e_hat))

            # 5. U_phys_bnd Discrete boundary balance error
            diag = compute_heat_balance(grid, t_pred, k_f, q_f, bcs)
            u_bnd_val = float(np.abs(diag.relative_imbalance))

            # 6. U_phys_lin Linear residual norm
            sys = assemble_steady_heat_system(grid, k_f, q_f, bcs)
            r = sys.A.dot(t_pred.ravel()) - sys.b
            b_norm = np.linalg.norm(sys.b)
            u_lin_val = float(np.linalg.norm(r) / (b_norm if b_norm > 1e-12 else 1.0))

        err_map = np.abs(t_pred - t_true)
        actual_maxaes.append(float(np.max(err_map)))
        actual_rmses.append(float(np.sqrt(np.mean(err_map**2))))

        s_het.append(u1_val)
        s_mcd3.append(u3_t3_val)
        s_mcd5.append(u3_t5_val)
        s_res.append(u_res_val)
        s_bnd.append(u_bnd_val)
        s_lin.append(u_lin_val)

    actual_maxaes_arr = np.array(actual_maxaes)
    labels_10k = (actual_maxaes_arr > 10.0).astype(int)
    labels_5k = (actual_maxaes_arr > 5.0).astype(int)

    signal_dict = {
        "U1_heteroscedastic": np.array(s_het),
        "U3_mcd_t3": np.array(s_mcd3),
        "U3_mcd_t5": np.array(s_mcd5),
        "U_res_predictor": np.array(s_res),
        "U_phys_bnd_flux": np.array(s_bnd),
        "U_phys_lin_res": np.array(s_lin),
    }

    eval_results = {
        "n_samples": n_samples,
        "n_unsafe_10k": int(np.sum(labels_10k)),
        "pct_unsafe_10k": float(np.mean(labels_10k) * 100.0),
        "n_unsafe_5k": int(np.sum(labels_5k)),
        "pct_unsafe_5k": float(np.mean(labels_5k) * 100.0),
        "signals": {},
    }

    for sig_name, sig_scores in signal_dict.items():
        pearson, spearman = compute_correlations(sig_scores, actual_maxaes_arr)
        auroc_10k, auprc_10k = compute_auroc_auprc(sig_scores, labels_10k)
        auroc_5k, auprc_5k = compute_auroc_auprc(sig_scores, labels_5k)
        risk_cov = compute_risk_coverage(sig_scores, actual_maxaes_arr, unsafe_thresh=10.0)

        eval_results["signals"][sig_name] = {
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "auroc_10k": auroc_10k,
            "auprc_10k": auprc_10k,
            "auroc_5k": auroc_5k,
            "auprc_5k": auprc_5k,
            "risk_coverage_curve": risk_cov,
        }

    return eval_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 08G Error Detection Benchmarking")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/multires_dataset.h5"),
        help="Path to multires dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/milestone_08g_error_detection.json"),
        help="Path to output error detection report",
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    logger.info(f"Using compute device: {device}")

    # Load frozen A3 model
    a3_path = Path("artifacts/checkpoints/milestone_08f/A3_hybrid_refined_fno/best_model.pt")
    a3_ckpt = torch.load(a3_path, map_location=device, weights_only=False)
    normalizer = FieldNormalizer.from_dict(a3_ckpt["normalizer"])
    log_k_mean = float(a3_ckpt["log_k_mean"])
    log_k_std = float(a3_ckpt["log_k_std"])

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
    a3_model.load_state_dict(a3_ckpt.get("model_state_dict", a3_ckpt))
    a3_model.eval()

    # Load U1 Heteroscedastic Head
    het_path = Path("artifacts/checkpoints/milestone_08g/A3_heteroscedastic/best_model.pt")
    het_ckpt = torch.load(het_path, map_location=device, weights_only=False)
    het_head = A3HeteroscedasticHead(in_channels=13, hidden_dim=32).to(device)
    het_head.load_state_dict(het_ckpt["model_state_dict"])
    het_head.eval()

    # Load U_res Residual Predictor
    res_path = Path("artifacts/checkpoints/milestone_08g/A3_residual_predictor/best_model.pt")
    res_ckpt = torch.load(res_path, map_location=device, weights_only=False)
    res_pred = A3ResidualPredictor(in_channels=13, hidden_dim=32).to(device)
    res_pred.load_state_dict(res_ckpt["model_state_dict"])
    res_pred.eval()

    resolutions = [32, 64, 128, 256]
    splits = ["test_iid", "test_ood"]

    results: dict[str, Any] = {
        "milestone": "08G",
        "phase": 4,
        "title": "Localized Error Detection and Risk Calibration across Resolutions",
        "provenance": {
            "a3_checkpoint_sha256": compute_file_sha256(a3_path),
            "het_checkpoint_sha256": compute_file_sha256(het_path),
            "res_checkpoint_sha256": compute_file_sha256(res_path),
        },
        "benchmarks": {},
    }

    for res in resolutions:
        results["benchmarks"][f"{res}x{res}"] = {}
        for split in splits:
            logger.info(f"Evaluating uncertainty signals on {res}x{res} {split}...")
            split_res = evaluate_split_signals(
                a3_model=a3_model,
                het_head=het_head,
                res_pred=res_pred,
                normalizer=normalizer,
                log_k_mean=log_k_mean,
                log_k_std=log_k_std,
                h5_path=args.dataset,
                resolution=res,
                split=split,
                device=device,
            )
            results["benchmarks"][f"{res}x{res}"][split] = split_res
            for sname, sdata in split_res["signals"].items():
                logger.info(
                    f"{res:3d}x{res:3d} {split:8s} | {sname:20s} | "
                    f"Spearman: {sdata['spearman_rho']:+.3f} | "
                    f"AUROC (10K): {sdata['auroc_10k']:.3f} | "
                    f"AUPRC (10K): {sdata['auprc_10k']:.3f}"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Error detection benchmark saved to {args.output}")


if __name__ == "__main__":
    main()
