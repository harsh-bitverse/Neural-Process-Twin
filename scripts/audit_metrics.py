"""Audit and recalculate all metrics directly from saved checkpoints and datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from neural_process_twin.data.boundary_encoding import decode_boundary_conditions
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.evaluation.metrics import (
    compute_aggregate_statistics,
    compute_all_metrics,
)
from neural_process_twin.evaluation.residual import evaluate_physical_residuals
from neural_process_twin.training.checkpoint import load_checkpoint


def audit_json_reports() -> None:
    print("=" * 80)
    print("1. INSPECTING RAW JSON REPORTS")
    print("=" * 80)
    models = [
        "ablation_cnn_supervised",
        "ablation_cnn_physics",
        "ablation_fno_supervised",
        "ablation_fno_physics",
        "ablation_fno_physics_bc",
    ]
    for model_name in models:
        json_path = Path(f"artifacts/reports/{model_name}.json")
        if not json_path.is_file():
            print(f"MISSING: {json_path}")
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"\nModel: {model_name}")
        print(f"Checkpoint: {data['checkpoint']}")
        for split in ["val", "test_iid", "test_ood"]:
            s = data["splits"][split]
            agg = s["aggregate_metrics"]
            res = s.get("residual_metrics", {})
            pde_l2 = res.get("pde_residual_l2_w", {}).get("mean", float("nan"))
            flux_imb = res.get("flux_imbalance_w", {}).get("mean", float("nan"))
            print(
                f"  [{split:<8}] Rel-L2={agg['relative_l2']['mean']:.6f}, "
                f"RMSE={agg['rmse_k']['mean']:.4f} K, "
                f"MAE={agg['mae_k']['mean']:.4f} K, "
                f"MaxErr={agg['max_error_k']['mean']:.4f} K, "
                f"PDE_Res={pde_l2:.2f} W, "
                f"Flux_Imb={flux_imb:.2f} W"
            )


def recompute_from_checkpoints() -> None:
    print("\n" + "=" * 80)
    print("2. RECOMPUTING METRICS DIRECTLY FROM MODEL CHECKPOINTS AND DATASETS")
    print("=" * 80)
    dataset_path = Path("artifacts/data/smoke_dataset.h5")
    models = [
        "ablation_cnn_supervised",
        "ablation_cnn_physics",
        "ablation_fno_supervised",
        "ablation_fno_physics",
        "ablation_fno_physics_bc",
    ]

    splits = ["validation", "test_iid", "test_ood"]

    for model_name in models:
        ckpt_path = Path(f"artifacts/checkpoints/{model_name}/best_model.pt")
        print(f"\n>>> Model Checkpoint: {ckpt_path}")
        ckpt_data = load_checkpoint(ckpt_path, device="cpu")
        model = ckpt_data["model"]
        model.eval()

        for split_key in splits:
            # Map validation to val for display
            split_disp = "val" if split_key == "validation" else split_key
            ds = HeatFieldDataset(
                file_path=dataset_path,
                split=split_key,
                include_bcs=True,
                include_coords=True,
            )
            loader = DataLoader(ds, batch_size=1, shuffle=False)

            sample_metrics_list = []
            residual_metrics_list = []

            with torch.no_grad():
                for _idx, batch in enumerate(loader):
                    x = batch["x"]
                    y_pred = model(x)
                    y_pred_np = y_pred.squeeze(0).squeeze(0).numpy()

                    # Reconstruct Kelvin
                    if ds.normalizer is not None and ds.normalizer.is_fitted:
                        t_pred_k = ds.normalizer.inverse_transform_temperature(y_pred_np)
                        t_ref_k = batch["temperature_raw"].squeeze(0).squeeze(0).numpy()
                    else:
                        t_pred_k = y_pred_np
                        t_ref_k = batch["y"].squeeze(0).squeeze(0).numpy()

                    m = compute_all_metrics(t_pred_k, t_ref_k)
                    sample_metrics_list.append(m)

                    # Compute physical residuals
                    k_raw = batch["k_raw"].squeeze(0).squeeze(0).numpy()
                    q_raw = batch["q_raw"].squeeze(0).squeeze(0).numpy()
                    x_np = x.squeeze(0).numpy()
                    bcs = decode_boundary_conditions(x_np, grid=ds.grid, normalizer=ds.normalizer)
                    r_diag = evaluate_physical_residuals(
                        grid=ds.grid,
                        k_field=k_raw,
                        q_field=q_raw,
                        bcs=bcs,
                        temperature_pred=t_pred_k,
                    )
                    residual_metrics_list.append(r_diag)

            agg = compute_aggregate_statistics(sample_metrics_list)
            res_agg = compute_aggregate_statistics(residual_metrics_list)

            print(
                f"  [{split_disp:<8}] Recomputed: "
                f"Rel-L2={agg['relative_l2']['mean']:.6f}, "
                f"RMSE={agg['rmse_k']['mean']:.4f} K, "
                f"MAE={agg['mae_k']['mean']:.4f} K, "
                f"MaxErr={agg['max_error_k']['mean']:.4f} K, "
                f"PDE_Res={res_agg['pde_residual_l2_w']['mean']:.2f} W, "
                f"Flux_Imb={res_agg['flux_imbalance_w']['mean']:.2f} W"
            )


def audit_checkpoints_provenance() -> None:
    print("\n" + "=" * 80)
    print("3. CHECKPOINT PROVENANCE, WEIGHT DIFFERENCES, AND OPTIMIZER STEPS")
    print("=" * 80)
    models = [
        "ablation_cnn_supervised",
        "ablation_cnn_physics",
        "ablation_fno_supervised",
        "ablation_fno_physics",
        "ablation_fno_physics_bc",
    ]
    ckpts = {}
    for name in models:
        p = Path(f"artifacts/checkpoints/{name}/best_model.pt")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        ckpts[name] = ckpt
        opt_state = ckpt.get("optimizer_state_dict", {})
        steps = [s.get("step", 0) for s in opt_state.get("state", {}).values()]
        max_step = max(steps) if steps else 0
        print(
            f"Model: {name:<26} | Best Epoch: {ckpt.get('epoch'):<2} | Best Val Loss: {ckpt.get('val_metric'):.6f} | Optimizer Step: {max_step}"
        )

    # Compare FNO supervised vs FNO physics weights
    diffs = []
    w_sup = ckpts["ablation_fno_supervised"]["model_state_dict"]
    w_phys = ckpts["ablation_fno_physics"]["model_state_dict"]
    for k in w_sup:
        d = torch.max(torch.abs(w_sup[k] - w_phys[k])).item()
        diffs.append(d)
    print(f"\nMax parameter delta between fno_supervised and fno_physics: {max(diffs):.8f}")
    mean_diff = float(np.mean([torch.mean(torch.abs(w_sup[k] - w_phys[k])).item() for k in w_sup]))
    print(f"Mean parameter delta between fno_supervised and fno_physics: {mean_diff:.8f}")

    # Compare FNO supervised vs FNO physics_bc weights
    w_bc = ckpts["ablation_fno_physics_bc"]["model_state_dict"]
    diffs_bc = [torch.max(torch.abs(w_sup[k] - w_bc[k])).item() for k in w_sup]
    print(f"Max parameter delta between fno_supervised and fno_physics_bc: {max(diffs_bc):.8f}")


def audit_loss_and_gradients() -> None:
    print("\n" + "=" * 80)
    print("4. LOSS COMPONENTS AND GRADIENT DYNAMICS (BATCH 0)")
    print("=" * 80)
    from neural_process_twin.models.fno2d import FourierNeuralOperator2D
    from neural_process_twin.training.loss import RelativeL2Loss
    from neural_process_twin.training.physics_loss import (
        BoundaryConditionLoss,
        DimensionlessPDELoss,
    )

    ds = HeatFieldDataset(
        "artifacts/data/smoke_dataset.h5", split="train", include_bcs=True, include_coords=True
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))

    torch.manual_seed(42)
    model = FourierNeuralOperator2D(
        in_channels=12, out_channels=1, modes_x=8, modes_y=8, width=32, depth=4
    )
    y_pred = model(batch["x"])

    # 1. Data loss
    loss_data_fn = RelativeL2Loss()
    loss_data = loss_data_fn(y_pred, batch["y"])

    # 2. PDE loss
    t_std = ds.normalizer.t_stats.std
    t_mean = ds.normalizer.t_stats.mean
    t_pred_k = y_pred * t_std + t_mean

    pde_fn = DimensionlessPDELoss(grid=ds.grid, residual_type="full", q_floor=1.0)
    loss_pde = pde_fn(
        temperature_k=t_pred_k,
        k_field=batch["k_raw"],
        q_field=batch["q_raw"],
        bc_channels=batch["x"][:, 4:12],
        t_mean=t_mean,
        t_std=t_std,
    )

    # 3. BC loss
    bc_fn = BoundaryConditionLoss(grid=ds.grid, delta_t_char=t_std, flux_scale=1000.0)
    loss_bc = bc_fn(
        temperature_k=t_pred_k,
        k_field=batch["k_raw"],
        bc_channels=batch["x"][:, 4:12],
        t_mean=t_mean,
        t_std=t_std,
    )

    print("Loss values at epoch 0 (untrained):")
    print(f"  loss_data (Relative L2):             {loss_data.item():.6f}")
    print(f"  loss_pde  (Dimensionless):           {loss_pde.item():.6f}")
    print(f"  loss_bc   (Dimensionless):           {loss_bc.item():.6f}")
    print(
        f"  Weighted PDE term (0.01 * loss_pde): {0.01 * loss_pde.item():.6f} ({(0.01 * loss_pde.item() / loss_data.item()) * 100:.3f}% of data loss)"
    )
    print(
        f"  Weighted BC term  (0.05 * loss_bc):  {0.05 * loss_bc.item():.6f} ({(0.05 * loss_bc.item() / loss_data.item()) * 100:.3f}% of data loss)"
    )

    # Gradients
    # Data gradient
    model.zero_grad()
    loss_data.backward(retain_graph=True)
    g_data = torch.linalg.norm(
        torch.stack([torch.linalg.norm(p.grad) for p in model.parameters()])
    ).item()

    # PDE gradient
    model.zero_grad()
    (0.01 * loss_pde).backward(retain_graph=True)
    g_pde_weighted = torch.linalg.norm(
        torch.stack([torch.linalg.norm(p.grad) for p in model.parameters()])
    ).item()

    # BC gradient
    model.zero_grad()
    (0.05 * loss_bc).backward()
    g_bc_weighted = torch.linalg.norm(
        torch.stack([torch.linalg.norm(p.grad) for p in model.parameters()])
    ).item()

    print("\nGradient norms ||grad_theta|| at epoch 0:")
    print(f"  Data loss gradient:         {g_data:.6f}")
    print(
        f"  Weighted PDE loss gradient: {g_pde_weighted:.6f} ({g_pde_weighted / g_data * 100:.3f}% of data grad)"
    )
    print(
        f"  Weighted BC loss gradient:  {g_bc_weighted:.6f} ({g_bc_weighted / g_data * 100:.3f}% of data grad)"
    )


if __name__ == "__main__":
    audit_json_reports()
    recompute_from_checkpoints()
    audit_checkpoints_provenance()
    audit_loss_and_gradients()
