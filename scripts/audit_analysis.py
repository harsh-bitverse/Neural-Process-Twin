"""Detailed statistical and numerical audit calculations for Milestone 04C."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.stats as stats


def run_audit() -> None:
    agg_path = Path("artifacts/reports/milestone_04b_aggregated_results.json")
    with open(agg_path, encoding="utf-8") as f:
        agg = json.load(f)

    print("=" * 80)
    print("1. SEED-LEVEL STATISTICS: POPULATION (ddof=0) VS SAMPLE (ddof=1)")
    print("=" * 80)

    for model, data in agg.items():
        print(f"\nModel: {model}")
        for split in ["test_iid", "test_ood"]:
            rel_l2 = data["evaluation_by_split"][split]["relative_l2_mean"]["values"]
            rmse = data["evaluation_by_split"][split]["rmse_kelvin_mean"]["values"]
            res = data["evaluation_by_split"][split]["pde_residual_w_mean"]["values"]
            flux = data["evaluation_by_split"][split]["flux_imbalance_w_mean"]["values"]

            print(f"  Split: {split}")
            print(
                f"    Rel-L2 [%]: mean={np.mean(rel_l2) * 100:.5f}, pop_std={np.std(rel_l2, ddof=0) * 100:.5f}, sample_std={np.std(rel_l2, ddof=1) * 100:.5f}, vals={[round(x * 100, 5) for x in rel_l2]}"
            )
            print(
                f"    RMSE [K]:   mean={np.mean(rmse):.4f}, pop_std={np.std(rmse, ddof=0):.4f}, sample_std={np.std(rmse, ddof=1):.4f}, vals={[round(x, 4) for x in rmse]}"
            )
            print(
                f"    Res [W]:    mean={np.mean(res):.2f}, pop_std={np.std(res, ddof=0):.2f}, sample_std={np.std(res, ddof=1):.2f}, vals={[round(x, 2) for x in res]}"
            )
            print(
                f"    Flux [W]:   mean={np.mean(flux):.2f}, pop_std={np.std(flux, ddof=0):.2f}, sample_std={np.std(flux, ddof=1):.2f}, vals={[round(x, 2) for x in flux]}"
            )

    print("\n" + "=" * 80)
    print("2. HYPOTHESIS TESTING ON RESIDUAL REDUCTION: FNO & CNN (test_iid)")
    print("=" * 80)

    # FNO test_iid
    fno_sup_res = np.array(
        agg["scaled_fno_supervised"]["evaluation_by_split"]["test_iid"]["pde_residual_w_mean"][
            "values"
        ]
    )
    fno_pbc_res = np.array(
        agg["scaled_fno_physics_bc"]["evaluation_by_split"]["test_iid"]["pde_residual_w_mean"][
            "values"
        ]
    )

    # CNN test_iid
    cnn_sup_res = np.array(
        agg["scaled_cnn_supervised"]["evaluation_by_split"]["test_iid"]["pde_residual_w_mean"][
            "values"
        ]
    )
    cnn_pbc_res = np.array(
        agg["scaled_cnn_physics_bc"]["evaluation_by_split"]["test_iid"]["pde_residual_w_mean"][
            "values"
        ]
    )

    for name, sup, pbc in [
        ("FNO-2D", fno_sup_res, fno_pbc_res),
        ("CNN-2D", cnn_sup_res, cnn_pbc_res),
    ]:
        print(f"\n--- {name} Residual Reduction Audit ---")
        diff = sup - pbc
        pct_reduction = (sup - pbc) / sup * 100.0
        mean_sup = np.mean(sup)
        mean_pbc = np.mean(pbc)
        mean_diff = np.mean(diff)
        pct_mean_reduction = (mean_sup - mean_pbc) / mean_sup * 100.0

        s_sup_pop = np.std(sup, ddof=0)
        s_pbc_pop = np.std(pbc, ddof=0)
        s_sup_sample = np.std(sup, ddof=1)
        s_pbc_sample = np.std(pbc, ddof=1)

        # Milestone 04b calculation:
        sigma_diff_04b = np.sqrt(s_sup_pop**2 + s_pbc_pop**2)
        t_04b = mean_diff / sigma_diff_04b

        # Unpaired Welch t-test (sample std):
        se_welch = np.sqrt(s_sup_sample**2 / 3 + s_pbc_sample**2 / 3)
        t_welch = mean_diff / se_welch
        # Welch-Satterthwaite df:
        df_num = (s_sup_sample**2 / 3 + s_pbc_sample**2 / 3) ** 2
        df_den = (s_sup_sample**2 / 3) ** 2 / 2 + (s_pbc_sample**2 / 3) ** 2 / 2
        df_welch = df_num / df_den
        p_welch = 2 * (1 - stats.t.cdf(np.abs(t_welch), df=df_welch))

        # Paired t-test (seeds 42, 43, 44 matched):
        s_diff_sample = np.std(diff, ddof=1)
        se_paired = s_diff_sample / np.sqrt(3)
        t_paired = mean_diff / se_paired
        df_paired = 2
        p_paired = 2 * (1 - stats.t.cdf(np.abs(t_paired), df=df_paired))

        print(
            f"  Supervised: {mean_sup:.2f} W (pop std={s_sup_pop:.2f}, sample std={s_sup_sample:.2f})"
        )
        print(
            f"  Physics+BC: {mean_pbc:.2f} W (pop std={s_pbc_pop:.2f}, sample std={s_pbc_sample:.2f})"
        )
        print(f"  Delta: {mean_diff:.2f} W ({pct_mean_reduction:.2f}% reduction)")
        print(f"  Per-seed values: sup={sup}, pbc={pbc}, diff={diff}")
        print(f"  Per-seed % reduction: {pct_reduction}% (mean={np.mean(pct_reduction):.2f}%)")
        print(
            f"  Milestone 04B claim: t = {t_04b:.2f} (reported as > {int(t_04b)} sigma, p < 10^-6)"
        )
        print(f"  Welch t-test: t = {t_welch:.2f}, df = {df_welch:.2f}, p-value = {p_welch:.6f}")
        print(f"  Paired t-test: t = {t_paired:.2f}, df = {df_paired}, p-value = {p_paired:.6f}")

    print("\n" + "=" * 80)
    print("3. WITHIN-PARTITION (SAMPLE-LEVEL) VARIABILITY")
    print("=" * 80)
    # Check individual run report for sample-level std
    rep_path = Path("artifacts/reports/scaled_fno_supervised_seed42.json")
    with open(rep_path, encoding="utf-8") as f:
        rep = json.load(f)
    for split in ["test_iid", "test_ood"]:
        agg_met = rep["evaluation"][split]
        print(f"Model: scaled_fno_supervised_seed42, Split: {split}")
        print(
            f"  Relative L2: mean={agg_met['relative_l2_mean'] * 100:.4f}%, std={agg_met['relative_l2_std'] * 100:.4f}%"
        )
        print(
            f"  RMSE:        mean={agg_met['rmse_kelvin_mean']:.4f} K, std={agg_met['rmse_kelvin_std']:.4f} K"
        )
        print(
            f"  MAE:         mean={agg_met['mae_kelvin_mean']:.4f} K, std={agg_met['mae_kelvin_std']:.4f} K"
        )
        print(
            f"  Max Error:   mean={agg_met['max_error_kelvin_mean']:.4f} K, std={agg_met['max_error_kelvin_std']:.4f} K"
        )
        print(
            f"  PDE Res:     mean={agg_met['pde_residual_w_mean']:.2f} W, std={agg_met['pde_residual_w_std']:.2f} W"
        )
        print(
            f"  Flux Imb:    mean={agg_met['flux_imbalance_w_mean']:.2f} W, std={agg_met['flux_imbalance_w_std']:.2f} W"
        )


if __name__ == "__main__":
    run_audit()
