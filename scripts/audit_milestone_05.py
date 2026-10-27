"""Comprehensive scientific and numerical audit script for Milestone 05."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neural_process_twin.training.checkpoint import load_checkpoint


def audit_ensemble() -> None:
    print("=" * 80)
    print("1. AUDITING ENSEMBLE MEMBER PROVENANCE & TRAINING RUNS")
    print("=" * 80)

    member_seeds = [42, 43, 44, 45, 46]
    member_data = []

    for seed in member_seeds:
        json_path = Path(f"artifacts/reports/scaled_fno_supervised_seed{seed}.json")
        ckpt_path = Path(f"artifacts/checkpoints/scaled_fno_supervised_seed{seed}/best_model.pt")

        if not json_path.is_file():
            print(f"MISSING JSON: {json_path}")
            continue
        if not ckpt_path.is_file():
            print(f"MISSING CKPT: {ckpt_path}")
            continue

        with open(json_path, encoding="utf-8") as f:
            rep = json.load(f)

        ckpt = load_checkpoint(ckpt_path, device="cpu")
        model_cfg = ckpt.get("model_config", {})
        norm_stats = ckpt.get("normalizer_stats", {})

        t_stats = norm_stats.get("t_stats") or norm_stats.get("temperature", {})
        t_mean = getattr(
            t_stats, "mean", t_stats.get("mean") if isinstance(t_stats, dict) else None
        )
        t_std = getattr(t_stats, "std", t_stats.get("std") if isinstance(t_stats, dict) else None)

        member_info = {
            "seed": seed,
            "best_epoch": rep["training_summary"]["best_epoch"],
            "epochs_completed": rep["training_summary"]["epochs_completed"],
            "total_steps": rep["training_summary"]["total_optimizer_steps"],
            "training_time_s": rep["training_summary"]["training_time_seconds"],
            "best_val_loss": rep["training_summary"]["best_val_loss"],
            "evaluated_checkpoint": rep["training_summary"]["checkpoint_evaluated"],
            "ckpt_val_metric": ckpt.get("val_metric"),
            "model_config": model_cfg,
            "t_mean": t_mean,
            "t_std": t_std,
            "eval": rep["evaluation"],
        }
        member_data.append(member_info)

        print(f"\nMember Seed {seed}:")
        print(f"  Checkpoint: {ckpt_path} (exists={ckpt_path.is_file()})")
        print(
            f"  Best Epoch: {member_info['best_epoch']}/{member_info['epochs_completed']} (Steps: {member_info['total_steps']})"
        )
        print(
            f"  Best Val Loss: {member_info['best_val_loss']:.6f} | Checkpoint val_metric: {member_info['ckpt_val_metric']:.6f}"
        )
        print(
            f"  Arch: {model_cfg.get('architecture')}, width={model_cfg.get('width')}, depth={model_cfg.get('depth')}, modes=({model_cfg.get('modes_x')}, {model_cfg.get('modes_y')})"
        )
        print(f"  Norm stats: T_mean={t_mean}, T_std={t_std}")

    print("\n" + "=" * 80)
    print("2. INDIVIDUAL MEMBER PERFORMANCE VS. ENSEMBLE MEAN")
    print("=" * 80)

    # Load master ensemble results
    ens_path = Path("artifacts/reports/milestone_05_ensemble_results.json")
    with open(ens_path, encoding="utf-8") as f:
        ens_rep = json.load(f)

    splits = ["validation", "test_iid", "test_ood"]
    metrics = [
        ("relative_l2_mean", "Relative L2 [%]", 100.0),
        ("rmse_kelvin_mean", "RMSE [K]", 1.0),
        ("mae_kelvin_mean", "MAE [K]", 1.0),
        ("max_error_kelvin_mean", "Max Error [K]", 1.0),
        ("pde_residual_w_mean", "PDE Residual [W]", 1.0),
        ("flux_imbalance_w_mean", "Flux Imbalance [W]", 1.0),
    ]

    for s in splits:
        print(f"\n--- Split: {s} ---")
        ens_split = ens_rep["splits"][s]["aggregate_error"]

        for m_key, m_label, scale in metrics:
            member_vals = [m["eval"][s][m_key] * scale for m in member_data]
            ens_val = ens_split[m_key] * scale

            mean_single = float(np.mean(member_vals))
            sample_std_single = float(np.std(member_vals, ddof=1))
            pop_std_single = float(np.std(member_vals, ddof=0))
            best_single = float(np.min(member_vals))
            worst_single = float(np.max(member_vals))

            # Unrounded percentage improvement
            pct_gain_vs_mean = (mean_single - ens_val) / mean_single * 100.0
            pct_gain_vs_best = (best_single - ens_val) / best_single * 100.0

            print(f"  {m_label:<20}:")
            print(f"    Members ({len(member_vals)} seeds): {[round(x, 4) for x in member_vals]}")
            print(
                f"    Member Mean: {mean_single:.4f} ± {sample_std_single:.4f} (sample) [{pop_std_single:.4f} pop] [worst: {worst_single:.4f}]"
            )
            print(f"    Ensemble Mean: {ens_val:.4f}")
            print(
                f"    Improvement vs. Member Mean: {pct_gain_vs_mean:+.2f}% | vs. Best Member ({best_single:.4f}): {pct_gain_vs_best:+.2f}%"
            )

    print("\n" + "=" * 80)
    print("3. AUDITING ERROR-UNCERTAINTY CORRELATIONS & STATISTICAL SIGNIFICANCE")
    print("=" * 80)
    for s in splits:
        uq = ens_rep["splits"][s]["uncertainty_quality"]
        print(f"\nSplit: {s}")
        print(
            f"  Mean Uncertainty: {uq['mean_uncertainty_k']:.4f} K | Max Uncertainty: {uq['max_uncertainty_k']:.4f} K"
        )
        print(f"  Pearson r (RMSE):      {uq['pearson_r_rmse']:.4f}")
        print(f"  Spearman rho (RMSE):   {uq['spearman_rho_rmse']:.4f}")
        print(f"  Pearson r (Rel-L2):    {uq['pearson_r_rel_l2']:.4f}")
        print(f"  Spearman rho (Rel-L2): {uq['spearman_rho_rel_l2']:.4f}")
        print(f"  Worst 10% AUROC:       {uq['worst_10pct_detection_auroc']:.4f}")
        print(f"  Worst 20% AUROC:       {uq['worst_20pct_detection_auroc']:.4f}")

    print("\n" + "=" * 80)
    print("4. AUDITING CONFORMAL CALIBRATION & COVERAGE")
    print("=" * 80)
    conf = ens_rep["conformal_calibration"]
    for conf_key, conf_data in conf.items():
        print(f"\nConfiguration: {conf_key}")
        print(
            f"  Confidence Level: {conf_data['confidence_level']} | Type: {conf_data['coverage_type']} | q_hat: {conf_data['quantile_multiplier_q_hat']:.4f}"
        )
        for s, s_data in conf_data["splits"].items():
            print(
                f"    {s:<12}: Empirical Coverage = {s_data['empirical_coverage'] * 100:.2f}% | Mean Width = {s_data['mean_interval_width_k']:.2f} K | Median Width = {s_data['median_interval_width_k']:.2f} K"
            )

    print("\n" + "=" * 80)
    print("5. AUDITING OOD DETECTION METRICS & THRESHOLD LEAKAGE")
    print("=" * 80)
    ood = ens_rep["ood_detection"]
    for det_name, det_res in ood.items():
        print(f"\nDetector: {det_name}")
        print(f"  AUROC: {det_res['auroc']:.4f} | AUPRC: {det_res['auprc']:.4f}")
        print(
            f"  Threshold: {det_res['threshold']:.4f} | TPR@5%FPR: {det_res['tpr_at_5pct_fpr'] * 100:.2f}% | F1: {det_res['optimal_f1']:.4f}"
        )


if __name__ == "__main__":
    audit_ensemble()
