"""Display summary results for Milestone 05B."""

import json

with open("artifacts/reports/milestone_05b_calibration_results.json") as f:
    data = json.load(f)

print("=" * 80)
print("1. ENSEMBLE PROVENANCE")
print("=" * 80)
for p in data["ensemble_provenance"]:
    print(
        f"  {p['path']}: sha256={p['sha256'][:16]}..., size={p['size_bytes']} bytes, epoch={p['epoch']}"
    )

print("\n" + "=" * 80)
print("2. CONFORMAL CALIBRATION: INDEPENDENT CALIBRATION VS VALIDATION REUSE")
print("=" * 80)
cal_srcs = data["conformal_calibration"]["calibration_source"]
for src_name, configs in cal_srcs.items():
    print(f"\nSource: {src_name}")
    print(
        f"{'Config':<18} | {'q_hat':<7} | {'Test-IID Cov':<12} | {'IID Width':<10} | {'Test-OOD Cov':<12} | {'OOD Width':<10}"
    )
    print("-" * 80)
    for cfg_key, val in configs.items():
        q = val["quantile_multiplier_q_hat"]
        iid_cov = val["splits"]["test_iid"]["empirical_coverage"]
        iid_w = val["splits"]["test_iid"]["mean_interval_width_k"]
        ood_cov = val["splits"]["test_ood"]["empirical_coverage"]
        ood_w = val["splits"]["test_ood"]["mean_interval_width_k"]
        print(
            f"{cfg_key:<18} | {q:<7.4f} | {iid_cov * 100:>10.2f}% | {iid_w:>8.3f} K | {ood_cov * 100:>10.2f}% | {ood_w:>8.1f} K"
        )

print("\n" + "=" * 80)
print("3. OOD DETECTION: FROZEN THRESHOLDS VS TEST-SET OPTIMAL DIAGNOSTIC")
print("=" * 80)
print(
    f"{'Detector':<28} | {'AUROC':<6} | {'Thresh(Cal)':<11} | {'IID FPR':<8} | {'OOD TPR':<8} | {'Test Acc':<8} | {'Test F1':<7}"
)
print("-" * 88)
for det_name, res in data["ood_detection"].items():
    cal_op = res["frozen_operating_point_calibration"]
    auroc = res["auroc"]
    print(
        f"{det_name:<28} | {auroc:<6.4f} | "
        f"{cal_op['frozen_threshold']:<11.4f} | "
        f"{cal_op['empirical_test_iid_fpr'] * 100:>6.2f}% | "
        f"{cal_op['empirical_test_ood_tpr'] * 100:>6.2f}% | "
        f"{cal_op['test_accuracy'] * 100:>6.2f}% | "
        f"{cal_op['test_f1']:<7.4f}"
    )

print("\nValidation-Derived Frozen Threshold Comparison:")
print(
    f"{'Detector':<28} | {'Thresh(Val)':<11} | {'IID FPR':<8} | {'OOD TPR':<8} | {'Test Acc':<8} | {'Test F1':<7}"
)
print("-" * 88)
for det_name, res in data["ood_detection"].items():
    val_op = res["frozen_operating_point_validation"]
    print(
        f"{det_name:<28} | "
        f"{val_op['frozen_threshold']:<11.4f} | "
        f"{val_op['empirical_test_iid_fpr'] * 100:>6.2f}% | "
        f"{val_op['empirical_test_ood_tpr'] * 100:>6.2f}% | "
        f"{val_op['test_accuracy'] * 100:>6.2f}% | "
        f"{val_op['test_f1']:<7.4f}"
    )

print("\n" + "=" * 80)
print("4. MECHANISTIC OOD BREAKDOWN")
print("=" * 80)
for cat_name, cat in data["robustness_by_mechanism"].items():
    n = cat["num_samples"]
    err = cat["prediction_error"]
    unc = cat["uncertainty"]
    cov = cat["conformal_coverage"]
    det = cat["ood_detection"]
    print(f"\n--- Category: {cat_name} (N={n}, {cat['sample_fraction'] * 100:.1f}%) ---")
    print(
        f"  Rel L2 Error:       {err['relative_l2_mean_pct']:.2f}% ± {err['relative_l2_std_pct']:.2f}% (RMSE: {err['rmse_k_mean']:.2f} ± {err['rmse_k_std']:.2f} K)"
    )
    print(
        f"  Max Error / MAE:    Max {err['max_error_k_mean']:.2f} K | MAE {err['mae_k_mean']:.2f} K"
    )
    print(
        f"  Physical Violation: PDE Residual {err['pde_residual_w_mean']:.1f} ± {err['pde_residual_w_std']:.1f} W | Flux Imbalance {err['flux_imbalance_w_mean']:.1f} ± {err['flux_imbalance_w_std']:.1f} W"
    )
    print(
        f"  Uncertainty:        Mean s_T: {unc['mean_s_t_kelvin']:.3f} K | Peak s_T: {unc['max_s_t_kelvin']:.3f} K"
    )
    print(
        f"  Pointwise 95% Cov:  {cov['pointwise_95_empirical_cov'] * 100:.2f}% (Mean Interval Width: {cov['pointwise_95_mean_width_k']:.1f} K)"
    )
    print(
        f"  Simult. 95% Cov:    {cov['simultaneous_95_empirical_cov'] * 100:.2f}% (Mean Interval Width: {cov['simultaneous_95_mean_width_k']:.1f} K)"
    )
    print(
        f"  Disagreement TPR:   {det['ensemble_disagreement_tpr_at_frozen_threshold'] * 100:.2f}% at frozen threshold"
    )
