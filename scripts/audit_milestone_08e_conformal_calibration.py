"""Conformal Calibration Audit for Milestone 08E (Phase 6).

Evaluates:
- Pointwise vs Simultaneous field coverage.
- Conditional coverage across:
  - Spatial resolution (32x32, 64x64, 128x128, 256x256)
  - Physical regime (IID vs OOD)
  - Conductivity contrast bins: C_k <= 5, 5 < C_k < 20, C_k >= 20
  - Heat source magnitude bins: q_max < 500, 500 <= q_max < 1000, q_max >= 1000
  - Boundary condition configurations (pure Dirichlet vs mixed Neumann)
- Evaluates both IID-only and Combined calibration to audit exchangeability breakdown.
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

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.uncertainty_fno import MCDropoutFNO2D
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("conformal_audit_08e")


def parse_bcs_row(grp: h5py.Group, idx: int) -> BoundaryConditions:
    """Parse BoundaryConditions from group row."""
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


def prepare_input_tensor(
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
    normalizer: FieldNormalizer,
) -> torch.Tensor:
    """Prepare 12-channel continuous input tensor."""
    res_y, res_x = k_field.shape
    x_norm = ((grid.x_mesh - grid.x_min) / grid.lx).astype(np.float32)
    y_norm = ((grid.y_mesh - grid.y_min) / grid.ly).astype(np.float32)

    x_arr = np.zeros((1, 12, res_y, res_x), dtype=np.float32)
    log_k = np.log(np.maximum(k_field, 1e-12))
    x_arr[0, 0] = ((log_k - 0.5) / 1.0).astype(np.float32)
    x_arr[0, 1] = normalizer.transform_q(q_field).astype(np.float32)
    x_arr[0, 2] = x_norm
    x_arr[0, 3] = y_norm

    weight_west = 1.0 - x_norm
    weight_east = x_norm
    weight_south = 1.0 - y_norm
    weight_north = y_norm

    if bcs.left.bc_type == BoundaryType.DIRICHLET:
        l_norm = normalizer.transform_temperature(bcs.left.value).astype(np.float32)
        x_arr[0, 4] = weight_west * l_norm[:, np.newaxis]
    else:
        l_norm = (bcs.left.value / 1000.0).astype(np.float32)
        x_arr[0, 6] = weight_west * l_norm[:, np.newaxis]

    if bcs.right.bc_type == BoundaryType.DIRICHLET:
        r_norm = normalizer.transform_temperature(bcs.right.value).astype(np.float32)
        x_arr[0, 5] = weight_east * r_norm[:, np.newaxis]
    else:
        r_norm = (bcs.right.value / 1000.0).astype(np.float32)
        x_arr[0, 7] = weight_east * r_norm[:, np.newaxis]

    if bcs.bottom.bc_type == BoundaryType.DIRICHLET:
        b_norm = normalizer.transform_temperature(bcs.bottom.value).astype(np.float32)
        x_arr[0, 8] = weight_south * b_norm[np.newaxis, :]
    else:
        b_norm = (bcs.bottom.value / 1000.0).astype(np.float32)
        x_arr[0, 10] = weight_south * b_norm[np.newaxis, :]

    if bcs.top.bc_type == BoundaryType.DIRICHLET:
        t_norm = normalizer.transform_temperature(bcs.top.value).astype(np.float32)
        x_arr[0, 9] = weight_north * t_norm[np.newaxis, :]
    else:
        t_norm = (bcs.top.value / 1000.0).astype(np.float32)
        x_arr[0, 11] = weight_north * t_norm[np.newaxis, :]

    return torch.from_numpy(x_arr)


def run_conformal_audit() -> dict[str, Any]:
    """Execute conditional conformal calibration audit."""
    cal_path = Path("artifacts/data/multires_calibration.h5")
    test_path = Path("artifacts/data/multires_dataset.h5")
    u3_ckpt_path = Path("artifacts/checkpoints/milestone_08d/U3_mc_dropout/best_model.pt")

    logger.info("Loading U3 MC Dropout checkpoint...")
    u3_ckpt = torch.load(u3_ckpt_path, map_location="cpu", weights_only=False)
    u3_model = MCDropoutFNO2D(modes_x=12, modes_y=12, width=32, depth=4, dropout_p=0.10)
    u3_model.load_state_dict(u3_ckpt.get("model_state_dict", u3_ckpt))
    normalizer = FieldNormalizer.from_dict(u3_ckpt["normalizer"])

    # 1. Fit Conformal Quantiles on multires_calibration.h5
    # (Separately on IID-only and Combined)
    logger.info("Computing conformal nonconformity scores on calibration cohort...")
    iid_scores_ptwise = []
    iid_scores_simult = []
    comb_scores_ptwise = []
    comb_scores_simult = []

    with h5py.File(cal_path, "r") as cal_f:
        for res in [32, 64]:
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
            for split in ["calibration_iid", "calibration_ood"]:
                grp = cal_f[f"res_{res}/{split}"]
                n = grp["temperature"].shape[0]
                k_all = grp["k"][:]
                q_all = grp["q"][:]
                t_all = grp["temperature"][:]

                for i in range(n):
                    bcs = parse_bcs_row(grp, i)
                    x_t = prepare_input_tensor(grid, k_all[i], q_all[i], bcs, normalizer)

                    u3_model.train()
                    preds = []
                    with torch.no_grad():
                        for _ in range(5):
                            p = u3_model(x_t).numpy()[0, 0]
                            preds.append(normalizer.inverse_transform_temperature(p))
                    stack = np.stack(preds, axis=0)
                    mean_t = np.mean(stack, axis=0)
                    std_t = np.std(stack, axis=0, ddof=1)
                    denom = np.maximum(std_t, 1e-3)

                    score_field = np.abs(t_all[i] - mean_t) / denom
                    max_score = float(np.max(score_field))

                    comb_scores_ptwise.extend(score_field.ravel().tolist())
                    comb_scores_simult.append(max_score)

                    if split == "calibration_iid":
                        iid_scores_ptwise.extend(score_field.ravel().tolist())
                        iid_scores_simult.append(max_score)

    alpha = 0.05
    n_iid_sim = len(iid_scores_simult)
    q_level_iid = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / max(n_iid_sim, 1)))
    q_hat_iid_ptwise = float(np.quantile(iid_scores_ptwise, 0.95))
    q_hat_iid_simult = float(np.quantile(iid_scores_simult, q_level_iid))

    n_comb_sim = len(comb_scores_simult)
    q_level_comb = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / max(n_comb_sim, 1)))
    q_hat_comb_ptwise = float(np.quantile(comb_scores_ptwise, 0.95))
    q_hat_comb_simult = float(np.quantile(comb_scores_simult, q_level_comb))

    quantiles = {
        "iid_only": {
            "q_hat_ptwise_95": q_hat_iid_ptwise,
            "q_hat_simult_95": q_hat_iid_simult,
        },
        "combined": {
            "q_hat_ptwise_95": q_hat_comb_ptwise,
            "q_hat_simult_95": q_hat_comb_simult,
        },
    }
    logger.info(f"Calibrated quantiles: {quantiles}")

    # 2. Evaluate Conditional Coverage on multires_dataset.h5
    audit_results: dict[str, Any] = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "U3_mc_dropout (T=5)",
            "quantiles": quantiles,
        },
        "conditional_coverage": {},
    }

    resolutions = [32, 64, 128, 256]
    with h5py.File(test_path, "r") as test_f:
        for res in resolutions:
            res_key = f"res_{res}"
            grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)
            logger.info(f"Auditing conformal coverage at {res}x{res}...")

            res_records: list[dict[str, Any]] = []

            for split in ["test_iid", "test_ood"]:
                grp = test_f[f"{res_key}/{split}"]
                n = min(40, grp["temperature"].shape[0])
                k_all = grp["k"][:n]
                q_all = grp["q"][:n]
                t_all = grp["temperature"][:n]

                for i in range(n):
                    bcs = parse_bcs_row(grp, i)
                    x_t = prepare_input_tensor(grid, k_all[i], q_all[i], bcs, normalizer)

                    c_k = float(np.max(k_all[i]) / np.maximum(np.min(k_all[i]), 1e-12))
                    q_max = float(np.max(np.abs(q_all[i])))
                    is_pure_dir = (
                        bcs.left.bc_type == BoundaryType.DIRICHLET
                        and bcs.right.bc_type == BoundaryType.DIRICHLET
                        and bcs.bottom.bc_type == BoundaryType.DIRICHLET
                        and bcs.top.bc_type == BoundaryType.DIRICHLET
                    )

                    u3_model.train()
                    preds = []
                    with torch.no_grad():
                        for _ in range(5):
                            p = u3_model(x_t).numpy()[0, 0]
                            preds.append(normalizer.inverse_transform_temperature(p))
                    stack = np.stack(preds, axis=0)
                    mean_t = np.mean(stack, axis=0)
                    std_t = np.std(stack, axis=0, ddof=1)

                    err = np.abs(t_all[i] - mean_t)
                    maxae = float(np.max(err))

                    # Coverage under IID-only calibration
                    iid_ptwise_cov = float(np.mean(err <= q_hat_iid_ptwise * std_t))
                    iid_simult_cov = float(maxae <= q_hat_iid_simult * np.max(std_t))

                    # Coverage under Combined calibration
                    comb_ptwise_cov = float(np.mean(err <= q_hat_comb_ptwise * std_t))
                    comb_simult_cov = float(maxae <= q_hat_comb_simult * np.max(std_t))
                    mpiw_comb = float(np.mean(2.0 * q_hat_comb_ptwise * std_t))

                    res_records.append(
                        {
                            "split": split,
                            "contrast": c_k,
                            "q_max": q_max,
                            "is_pure_dir": is_pure_dir,
                            "iid_ptwise_cov": iid_ptwise_cov,
                            "iid_simult_cov": iid_simult_cov,
                            "comb_ptwise_cov": comb_ptwise_cov,
                            "comb_simult_cov": comb_simult_cov,
                            "mpiw_comb": mpiw_comb,
                        }
                    )

            # Conditional aggregations
            def _filter_stats(subset: list[dict[str, Any]]) -> dict[str, Any]:
                if not subset:
                    return {"n": 0, "comb_ptwise": None, "comb_simult": None, "iid_ptwise": None}
                return {
                    "n_samples": len(subset),
                    "comb_ptwise_cov_pct": float(np.mean([x["comb_ptwise_cov"] for x in subset]))
                    * 100.0,
                    "comb_simult_cov_pct": float(np.mean([x["comb_simult_cov"] for x in subset]))
                    * 100.0,
                    "iid_ptwise_cov_pct": float(np.mean([x["iid_ptwise_cov"] for x in subset]))
                    * 100.0,
                    "iid_simult_cov_pct": float(np.mean([x["iid_simult_cov"] for x in subset]))
                    * 100.0,
                    "mean_mpiw_k": float(np.mean([x["mpiw_comb"] for x in subset])),
                }

            audit_results["conditional_coverage"][res_key] = {
                "overall": _filter_stats(res_records),
                "by_regime": {
                    "iid": _filter_stats([x for x in res_records if x["split"] == "test_iid"]),
                    "ood": _filter_stats([x for x in res_records if x["split"] == "test_ood"]),
                },
                "by_contrast": {
                    "contrast_le_5": _filter_stats(
                        [x for x in res_records if x["contrast"] <= 5.0]
                    ),
                    "contrast_5_to_20": _filter_stats(
                        [x for x in res_records if 5.0 < x["contrast"] < 20.0]
                    ),
                    "contrast_ge_20": _filter_stats(
                        [x for x in res_records if x["contrast"] >= 20.0]
                    ),
                },
                "by_heat_source": {
                    "q_under_500": _filter_stats([x for x in res_records if x["q_max"] < 500.0]),
                    "q_500_to_1000": _filter_stats(
                        [x for x in res_records if 500.0 <= x["q_max"] < 1000.0]
                    ),
                    "q_ge_1000": _filter_stats([x for x in res_records if x["q_max"] >= 1000.0]),
                },
                "by_boundary_type": {
                    "pure_dirichlet": _filter_stats([x for x in res_records if x["is_pure_dir"]]),
                    "mixed_neumann": _filter_stats(
                        [x for x in res_records if not x["is_pure_dir"]]
                    ),
                },
            }

    out_file = Path("artifacts/reports/milestone_08e_conformal_audit.json")
    with open(out_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    logger.info(f"Conformal calibration audit saved to {out_file}")
    return audit_results


if __name__ == "__main__":
    audit = run_conformal_audit()
    print("\n=== CONFORMAL AUDIT SUMMARY (res_256) ===")
    r256 = audit["conditional_coverage"]["res_256"]
    print(
        f"Overall Comb Coverage: Ptwise {r256['overall']['comb_ptwise_cov_pct']:.1f}%, Simult {r256['overall']['comb_simult_cov_pct']:.1f}%"
    )
    print(
        f"IID: Comb Ptwise {r256['by_regime']['iid']['comb_ptwise_cov_pct']:.1f}% | IID Ptwise {r256['by_regime']['iid']['iid_ptwise_cov_pct']:.1f}%"
    )
    print(
        f"OOD: Comb Ptwise {r256['by_regime']['ood']['comb_ptwise_cov_pct']:.1f}% | IID Ptwise {r256['by_regime']['ood']['iid_ptwise_cov_pct']:.1f}%"
    )
    print(
        f"C_k >= 20: Comb Ptwise {r256['by_contrast']['contrast_ge_20']['comb_ptwise_cov_pct']:.1f}%, Mean MPIW: {r256['by_contrast']['contrast_ge_20']['mean_mpiw_k']:.1f} K"
    )
