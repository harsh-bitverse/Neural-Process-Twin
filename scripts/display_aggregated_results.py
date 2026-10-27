"""Display aggregated Milestone 04B results across models and splits."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path("artifacts/reports/milestone_04b_aggregated_results.json")
    with open(path) as f:
        data = json.load(f)

    splits = ["validation", "test_iid", "test_ood"]

    for model_name, info in data.items():
        print("=" * 80)
        print(
            f"Model: {model_name} (Architecture: {info['model_architecture']}, Params: {info['trainable_parameters']:,})"
        )
        weights = info["loss_weights"]
        print(
            f"  Loss weights: w_data={weights['w_data']}, w_pde={weights['w_pde']}, w_bc={weights['w_bc']} (residual: {weights['residual_type']})"
        )
        ts = info["training_summary"]
        print(
            f"  Training: Best Epoch={ts['mean_best_epoch']:.1f} +- {ts['std_best_epoch']:.1f}, Steps={ts['mean_optimizer_steps']:.0f}, Time={ts['mean_training_time_s']:.1f}s"
        )

        for split in splits:
            s = info["evaluation_by_split"][split]
            rel_l2 = s["relative_l2_mean"]
            rmse = s["rmse_kelvin_mean"]
            mae = s["mae_kelvin_mean"]
            max_err = s["max_error_kelvin_mean"]
            pde = s["pde_residual_w_mean"]
            flux = s["flux_imbalance_w_mean"]

            print(
                f"  [{split:<10}] "
                f"Rel-L2: {rel_l2['mean'] * 100:.3f} +- {rel_l2['std'] * 100:.3f}% | "
                f"RMSE: {rmse['mean']:.2f} +- {rmse['std']:.2f} K | "
                f"MAE: {mae['mean']:.2f} +- {mae['std']:.2f} K | "
                f"MaxErr: {max_err['mean']:.2f} +- {max_err['std']:.2f} K | "
                f"PDE: {pde['mean']:.1f} +- {pde['std']:.1f} W | "
                f"Flux: {flux['mean']:.1f} +- {flux['std']:.1f} W"
            )


if __name__ == "__main__":
    main()
