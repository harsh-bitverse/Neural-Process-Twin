"""Milestone 10: Pristine Independent Evaluation of Dual-Mode Platform.

Evaluates Verified Numerical FVM and Experimental Neural (A3 & A4) modes on disjoint,
pristine physical problem realizations across all supported resolutions:
- Resolutions: 32x32, 64x64, 128x128, 256x256
- Subgroups:
  - IID: Moderate contrast inclusions (k_ratio in [2, 5]), uniform/moderate source (q in [10, 100] W/m^3)
  - OOD Contrast: High conductivity contrast (k_ratio in [10, 25])
  - OOD Source: High volumetric heat source (q in [300, 800] W/m^3)
  - OOD Boundary: Extreme temperature steps (200 K to 600 K) and sharp boundary gradients
- Metrics:
  - FVM: Relative algebraic residual, energy balance relative imbalance, solve time
  - Neural (A3 Uncorrected and A4 Corrected):
    - RMSE [K]
    - Relative L2 Error [%]
    - MaxAE [K] (median, 95th percentile, max)
    - MaxAE <= 10 K compliance rate [%]
    - End-to-end inference and correction latency [ms]

Saves results to: artifacts/reports/milestone_10_independent_evaluation.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from neural_process_twin.api.converters import (
    build_boundary_conditions,
    build_conductivity_array,
    build_domain_grid,
    build_source_array,
)
from neural_process_twin.api.models import (
    BoundaryConditionItem,
    BoundaryConditionsModel,
    FieldPatternModel,
    GridModel,
    HeatSourcePatternModel,
)
from neural_process_twin.api.routes import get_experimental_surrogate, get_solver


def create_iid_cases(seed: int = 1001, count: int = 20) -> list[dict[str, Any]]:
    """Generate independent IID problem specifications."""
    rng = np.random.default_rng(seed)
    cases = []
    for idx in range(count):
        k_bg = float(rng.uniform(0.8, 1.5))
        k_feat = float(rng.uniform(2.5, 6.0))
        cx = float(rng.uniform(0.3, 0.7))
        cy = float(rng.uniform(0.3, 0.7))
        r = float(rng.uniform(0.12, 0.28))
        q_val = float(rng.uniform(20.0, 80.0))
        t_left = float(rng.uniform(280.0, 320.0))
        t_right = float(rng.uniform(330.0, 380.0))

        case = {
            "case_id": f"iid_{idx:02d}",
            "subgroup": "IID",
            "k_pattern": FieldPatternModel(
                pattern_type="inclusion",
                inclusion_k_background=k_bg,
                inclusion_k_feature=k_feat,
                inclusion_center_x=cx,
                inclusion_center_y=cy,
                inclusion_radius=r,
            ),
            "q_pattern": HeatSourcePatternModel(
                pattern_type="uniform",
                uniform_value=q_val,
            ),
            "bcs": BoundaryConditionsModel(
                left=BoundaryConditionItem(side="left", type="dirichlet", value=t_left),
                right=BoundaryConditionItem(side="right", type="dirichlet", value=t_right),
                bottom=BoundaryConditionItem(side="bottom", type="neumann", value=0.0),
                top=BoundaryConditionItem(side="top", type="neumann", value=0.0),
            ),
        }
        cases.append(case)
    return cases


def create_ood_cases(seed: int = 2002, count_per_type: int = 10) -> list[dict[str, Any]]:
    """Generate independent physical OOD problem specifications."""
    rng = np.random.default_rng(seed)
    cases = []

    # 1. High-Contrast OOD
    for idx in range(count_per_type):
        k_bg = 1.0
        k_feat = float(rng.uniform(12.0, 25.0))  # Contrast up to 25:1
        cases.append(
            {
                "case_id": f"ood_contrast_{idx:02d}",
                "subgroup": "OOD_Contrast",
                "k_pattern": FieldPatternModel(
                    pattern_type="inclusion",
                    inclusion_k_background=k_bg,
                    inclusion_k_feature=k_feat,
                    inclusion_center_x=0.5,
                    inclusion_center_y=0.5,
                    inclusion_radius=0.22,
                ),
                "q_pattern": HeatSourcePatternModel(pattern_type="uniform", uniform_value=40.0),
                "bcs": BoundaryConditionsModel(
                    left=BoundaryConditionItem(side="left", type="dirichlet", value=300.0),
                    right=BoundaryConditionItem(side="right", type="dirichlet", value=350.0),
                    bottom=BoundaryConditionItem(side="bottom", type="neumann", value=0.0),
                    top=BoundaryConditionItem(side="top", type="neumann", value=0.0),
                ),
            }
        )

    # 2. High Heat Source OOD
    for idx in range(count_per_type):
        q_val = float(rng.uniform(300.0, 800.0))  # Severe thermal peaking
        cases.append(
            {
                "case_id": f"ood_source_{idx:02d}",
                "subgroup": "OOD_Source",
                "k_pattern": FieldPatternModel(
                    pattern_type="inclusion",
                    inclusion_k_background=1.0,
                    inclusion_k_feature=4.0,
                    inclusion_center_x=0.5,
                    inclusion_center_y=0.5,
                    inclusion_radius=0.2,
                ),
                "q_pattern": HeatSourcePatternModel(pattern_type="uniform", uniform_value=q_val),
                "bcs": BoundaryConditionsModel(
                    left=BoundaryConditionItem(side="left", type="dirichlet", value=300.0),
                    right=BoundaryConditionItem(side="right", type="dirichlet", value=300.0),
                    bottom=BoundaryConditionItem(side="bottom", type="dirichlet", value=300.0),
                    top=BoundaryConditionItem(side="top", type="dirichlet", value=300.0),
                ),
            }
        )

    # 3. Extreme Temperature Step OOD
    for idx in range(count_per_type):
        t_l = float(rng.uniform(150.0, 220.0))
        t_r = float(rng.uniform(550.0, 650.0))  # Delta T up to 500 K
        cases.append(
            {
                "case_id": f"ood_boundary_{idx:02d}",
                "subgroup": "OOD_Boundary",
                "k_pattern": FieldPatternModel(
                    pattern_type="inclusion",
                    inclusion_k_background=1.0,
                    inclusion_k_feature=3.0,
                    inclusion_center_x=0.5,
                    inclusion_center_y=0.5,
                    inclusion_radius=0.2,
                ),
                "q_pattern": HeatSourcePatternModel(pattern_type="uniform", uniform_value=10.0),
                "bcs": BoundaryConditionsModel(
                    left=BoundaryConditionItem(side="left", type="dirichlet", value=t_l),
                    right=BoundaryConditionItem(side="right", type="dirichlet", value=t_r),
                    bottom=BoundaryConditionItem(side="bottom", type="neumann", value=0.0),
                    top=BoundaryConditionItem(side="top", type="neumann", value=0.0),
                ),
            }
        )

    return cases


def evaluate_cohort() -> dict[str, Any]:
    solver = get_solver()
    surrogate = get_experimental_surrogate()

    resolutions = [32, 64, 128, 256]
    iid_cases = create_iid_cases(seed=1001, count=20)
    ood_cases = create_ood_cases(seed=2002, count_per_type=10)
    all_cases = iid_cases + ood_cases

    results: dict[str, Any] = {
        "metadata": {
            "benchmark": "Milestone 10 Independent Scientific Evaluation",
            "total_cases_per_resolution": len(all_cases),
            "iid_case_count": len(iid_cases),
            "ood_case_count": len(ood_cases),
            "resolutions": resolutions,
        },
        "per_resolution_summary": {},
        "per_subgroup_summary": {},
    }

    for n in resolutions:
        print("\n=======================================================")
        print(f"Evaluating Independent Cohort at Resolution {n}x{n} (N={len(all_cases)})")
        print("=======================================================")

        grid_model = GridModel(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0, nx=n, ny=n)
        grid = build_domain_grid(grid_model)

        cohort_records = []

        for case in all_cases:
            k_field = build_conductivity_array(grid, case["k_pattern"])
            q_field = build_source_array(grid, case["q_pattern"])
            bcs = build_boundary_conditions(case["bcs"])

            # 1. Reference FVM solve
            t0 = time.perf_counter()
            fvm_res = solver.solve(grid, k_field, q_field, bcs)
            t_fvm_ms = (time.perf_counter() - t0) * 1000.0
            t_true = fvm_res.temperature

            # 2. Neural A3 Uncorrected
            t0 = time.perf_counter()
            a3_res = surrogate.predict(grid, k_field, q_field, bcs, apply_physics_correction=False)
            t_a3_ms = (time.perf_counter() - t0) * 1000.0
            t_a3 = a3_res.temperature

            # 3. Neural A4 Corrected (Selective Boundary Jacobi)
            t0 = time.perf_counter()
            a4_res = surrogate.predict(grid, k_field, q_field, bcs, apply_physics_correction=True)
            t_a4_ms = (time.perf_counter() - t0) * 1000.0
            t_a4 = a4_res.temperature

            # Error metrics
            err_a3 = np.abs(t_a3 - t_true)
            rmse_a3 = float(np.sqrt(np.mean(err_a3**2)))
            rel_l2_a3 = float(np.linalg.norm(t_a3 - t_true) / np.linalg.norm(t_true) * 100.0)
            max_ae_a3 = float(np.max(err_a3))

            err_a4 = np.abs(t_a4 - t_true)
            rmse_a4 = float(np.sqrt(np.mean(err_a4**2)))
            rel_l2_a4 = float(np.linalg.norm(t_a4 - t_true) / np.linalg.norm(t_true) * 100.0)
            max_ae_a4 = float(np.max(err_a4))

            record = {
                "case_id": case["case_id"],
                "subgroup": case["subgroup"],
                "fvm": {
                    "latency_ms": t_fvm_ms,
                    "relative_residual": fvm_res.diagnostics.relative_residual,
                    "energy_imbalance_rel": fvm_res.diagnostics.energy_balance.relative_imbalance,
                },
                "a3_uncorrected": {
                    "rmse_k": rmse_a3,
                    "rel_l2_pct": rel_l2_a3,
                    "max_ae_k": max_ae_a3,
                    "satisfies_10k_tolerance": max_ae_a3 <= 10.0,
                    "latency_ms": t_a3_ms,
                },
                "a4_corrected": {
                    "rmse_k": rmse_a4,
                    "rel_l2_pct": rel_l2_a4,
                    "max_ae_k": max_ae_a4,
                    "satisfies_10k_tolerance": max_ae_a4 <= 10.0,
                    "latency_ms": t_a4_ms,
                },
            }
            cohort_records.append(record)

        # Aggregate per resolution
        def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
            max_aes_a3 = [r["a3_uncorrected"]["max_ae_k"] for r in records]
            rmses_a3 = [r["a3_uncorrected"]["rmse_k"] for r in records]
            rel_l2s_a3 = [r["a3_uncorrected"]["rel_l2_pct"] for r in records]
            comp_a3 = [r["a3_uncorrected"]["satisfies_10k_tolerance"] for r in records]

            max_aes_a4 = [r["a4_corrected"]["max_ae_k"] for r in records]
            rmses_a4 = [r["a4_corrected"]["rmse_k"] for r in records]
            rel_l2s_a4 = [r["a4_corrected"]["rel_l2_pct"] for r in records]
            comp_a4 = [r["a4_corrected"]["satisfies_10k_tolerance"] for r in records]

            fvm_rel_res = [r["fvm"]["relative_residual"] for r in records]
            fvm_rel_imb = [r["fvm"]["energy_imbalance_rel"] for r in records]

            return {
                "count": len(records),
                "fvm": {
                    "max_relative_residual": float(np.max(fvm_rel_res)),
                    "mean_relative_imbalance": float(np.mean(fvm_rel_imb)),
                },
                "a3_uncorrected": {
                    "rmse_mean_k": float(np.mean(rmses_a3)),
                    "rel_l2_mean_pct": float(np.mean(rel_l2s_a3)),
                    "max_ae_median_k": float(np.median(max_aes_a3)),
                    "max_ae_p95_k": float(np.percentile(max_aes_a3, 95)),
                    "max_ae_max_k": float(np.max(max_aes_a3)),
                    "tolerance_compliance_pct": float(np.mean(comp_a3) * 100.0),
                    "tolerance_failure_rate_pct": float((1.0 - np.mean(comp_a3)) * 100.0),
                },
                "a4_corrected": {
                    "rmse_mean_k": float(np.mean(rmses_a4)),
                    "rel_l2_mean_pct": float(np.mean(rel_l2s_a4)),
                    "max_ae_median_k": float(np.median(max_aes_a4)),
                    "max_ae_p95_k": float(np.percentile(max_aes_a4, 95)),
                    "max_ae_max_k": float(np.max(max_aes_a4)),
                    "tolerance_compliance_pct": float(np.mean(comp_a4) * 100.0),
                    "tolerance_failure_rate_pct": float((1.0 - np.mean(comp_a4)) * 100.0),
                },
            }

        res_summary = aggregate_metrics(cohort_records)
        results["per_resolution_summary"][f"{n}x{n}"] = res_summary

        # Aggregate by subgroup at this resolution
        for sg in ["IID", "OOD_Contrast", "OOD_Source", "OOD_Boundary"]:
            sg_records = [r for r in cohort_records if r["subgroup"] == sg]
            key = f"{n}x{n}_{sg}"
            results["per_subgroup_summary"][key] = aggregate_metrics(sg_records)

        print(f"Summary for {n}x{n}:")
        print(f"  FVM Max Rel Res: {res_summary['fvm']['max_relative_residual']:.2e}")
        print(
            f"  A3 (Uncorr) RMSE: {res_summary['a3_uncorrected']['rmse_mean_k']:.2f} K | MaxAE P95: {res_summary['a3_uncorrected']['max_ae_p95_k']:.2f} K | Compliance: {res_summary['a3_uncorrected']['tolerance_compliance_pct']:.1f}%"
        )
        print(
            f"  A4 (Corr)   RMSE: {res_summary['a4_corrected']['rmse_mean_k']:.2f} K | MaxAE P95: {res_summary['a4_corrected']['max_ae_p95_k']:.2f} K | Compliance: {res_summary['a4_corrected']['tolerance_compliance_pct']:.1f}%"
        )

    out_file = Path("artifacts/reports/milestone_10_independent_evaluation.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved independent evaluation results to: {out_file}")
    return results


if __name__ == "__main__":
    evaluate_cohort()
