import json
from pathlib import Path

results_file = Path("artifacts/reports/milestone_08b_results.json")
with open(results_file, encoding="utf-8") as f:
    data = json.load(f)

configs = data["configurations"]

print("=" * 90)
print(
    f"{'Configuration':<26} | {'IID RMSE':<10} | {'IID Bnd':<9} | {'IID Imb [W]':<12} | {'GRF Imb [W]':<13} | {'Params':<10}"
)
print("-" * 90)

for cid in ["A0", "A1", "A2"]:
    c = configs[cid]
    name = f"{cid}: {c['model_name']}"
    s = c["seeds"]["42"]
    iid = s["test_iid"]
    ood = s["test_ood"]
    grf_imb = ood["subgroups"]["short_correlation_grf"]["flux_imbalance_abs_w"]["mean"]
    print(
        f"{name:<26} | "
        f"{iid['overall_rmse_k']['mean']:<10.3f} | "
        f"{iid['boundary_rmse_k']['mean']:<9.3f} | "
        f"{iid['flux_imbalance_abs_w']['mean']:<12.1f} | "
        f"{grf_imb:<13.1f} | "
        f"{s['parameter_count']:<10,}"
    )

print("=" * 90)

print("\n--- DETAILED COMPARISON TABLE ---")
metrics = [
    ("IID Overall RMSE [K]", lambda s: s["test_iid"]["overall_rmse_k"]["mean"]),
    ("IID Interior RMSE [K]", lambda s: s["test_iid"]["interior_rmse_k"]["mean"]),
    ("IID Boundary RMSE [K]", lambda s: s["test_iid"]["boundary_rmse_k"]["mean"]),
    ("IID Relative L2 [%]", lambda s: s["test_iid"]["relative_l2_pct"]["mean"]),
    ("IID MaxAE [K]", lambda s: s["test_iid"]["max_ae_k"]["mean"]),
    ("IID Mean Imbalance [W]", lambda s: s["test_iid"]["flux_imbalance_abs_w"]["mean"]),
    ("IID Max Imbalance [W]", lambda s: s["test_iid"]["flux_imbalance_abs_w"]["max"]),
    ("IID PDE Residual L2 [W]", lambda s: s["test_iid"]["pde_residual_l2_w"]["mean"]),
    ("OOD Overall RMSE [K]", lambda s: s["test_ood"]["overall_rmse_k"]["mean"]),
    (
        "OOD High-Contrast RMSE [K]",
        lambda s: s["test_ood"]["subgroups"]["high_contrast_inclusions"]["overall_rmse_k"]["mean"],
    ),
    (
        "OOD High-Contrast Imbalance [W]",
        lambda s: s["test_ood"]["subgroups"]["high_contrast_inclusions"]["flux_imbalance_abs_w"][
            "mean"
        ],
    ),
    (
        "OOD GRF RMSE [K]",
        lambda s: s["test_ood"]["subgroups"]["short_correlation_grf"]["overall_rmse_k"]["mean"],
    ),
    (
        "OOD GRF Imbalance [W]",
        lambda s: s["test_ood"]["subgroups"]["short_correlation_grf"]["flux_imbalance_abs_w"][
            "mean"
        ],
    ),
    ("OOD Max Imbalance [W]", lambda s: s["test_ood"]["flux_imbalance_abs_w"]["max"]),
    ("Inference Latency [ms]", lambda s: s["test_iid"]["latency_ms"]["mean"]),
]

for label, func in metrics:
    v0 = func(configs["A0"]["seeds"]["42"])
    v1 = func(configs["A1"]["seeds"]["42"])
    v2 = func(configs["A2"]["seeds"]["42"])
    print(f"{label:<32}: A0 = {v0:12.3f} | A1 = {v1:12.3f} | A2 = {v2:12.3f}")

print("\n--- A0 ENSEMBLE SUMMARY (SEEDS 42, 43, 44) ---")
a0_ens = configs["A0"]["ensemble_summary"]
for split in ["test_iid", "test_ood"]:
    print(f"\n[{split.upper()}]")
    for k, v in a0_ens[split].items():
        print(f"  {k:<30}: {v['mean']:12.3f} +/- {v['std']:10.3f}")
