"""Summarize and tabulate ablation experiment metrics for Milestone 04."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    reports = [
        "ablation_cnn_supervised",
        "ablation_cnn_physics",
        "ablation_fno_supervised",
        "ablation_fno_physics",
        "ablation_fno_physics_bc",
    ]

    header = f"{'Model / Ablation':<26} | {'Split':<8} | {'Rel L2':<10} | {'RMSE (K)':<10} | {'MaxErr(K)':<10} | {'PDE Res(W)':<12} | {'Flux Imb(W)':<12}"
    print(header)
    print("-" * len(header))

    for rep in reports:
        p = Path(f"artifacts/reports/{rep}.json")
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for split in ["val", "test_iid", "test_ood"]:
            if split not in data["splits"]:
                continue
            s = data["splits"][split]
            agg = s["aggregate_metrics"]
            res = s.get("residual_metrics", {})
            rel_l2 = agg["relative_l2"]["mean"]
            rmse = agg["rmse_k"]["mean"]
            max_err = agg["max_error_k"]["mean"]
            pde_res = res.get("pde_residual_l2_w", {}).get("mean", float("nan"))
            flux_imb = res.get("flux_imbalance_w", {}).get("mean", float("nan"))
            print(
                f"{rep:<26} | {split:<8} | {rel_l2:<10.4e} | {rmse:<10.2f} | {max_err:<10.2f} | {pde_res:<12.2f} | {flux_imb:<12.2f}"
            )


if __name__ == "__main__":
    main()
