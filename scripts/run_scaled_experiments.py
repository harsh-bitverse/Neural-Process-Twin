"""Orchestrate scaled training experiments, gradient telemetry, and rigorous evaluation."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from neural_process_twin.data.boundary_encoding import (
    CANONICAL_CHANNEL_NAMES,
    decode_boundary_conditions,
)
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.evaluation.metrics import (
    compute_aggregate_statistics,
    compute_all_metrics,
)
from neural_process_twin.evaluation.residual import evaluate_physical_residuals
from neural_process_twin.models.factory import build_model
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.training.checkpoint import load_checkpoint
from neural_process_twin.training.loss import build_loss
from neural_process_twin.training.trainer import Trainer
from neural_process_twin.utils.config import load_training_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_gradient_vector(loss: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Compute gradients with respect to model parameters and flatten into a real 1D vector."""
    model.zero_grad()
    loss.backward(retain_graph=True)
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            if g.is_complex():
                grads.append(g.real.flatten().double())
                grads.append(g.imag.flatten().double())
            else:
                grads.append(g.flatten().double())
        else:
            if p.is_complex():
                grads.append(torch.zeros_like(p.real).flatten().double())
                grads.append(torch.zeros_like(p.imag).flatten().double())
            else:
                grads.append(torch.zeros_like(p).flatten().double())
    model.zero_grad()
    return torch.cat(grads)


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D real gradient vectors."""
    n1 = torch.linalg.norm(v1)
    n2 = torch.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_val = torch.dot(v1, v2) / (n1 * n2)
    return float(torch.clamp(cos_val, -1.0, 1.0).item())


def capture_gradient_diagnostics(
    model: torch.nn.Module,
    loss_fn: Any,
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Capture loss components, gradient norms, and cosine similarities on a representative batch."""
    was_training = model.training
    model.eval()

    x = batch["x"].to(device)
    y = batch["y"].to(device)

    y_pred = model(x)

    # 1. Data loss
    if hasattr(loss_fn, "data_loss_fn"):
        l_data = loss_fn.data_loss_fn(y_pred, y)
    else:
        l_data = loss_fn(y_pred, y)
        if isinstance(l_data, tuple):
            l_data = l_data[0]

    g_data = compute_gradient_vector(l_data, model)
    norm_data = float(torch.linalg.norm(g_data))

    norm_pde_weighted = 0.0
    norm_bc_weighted = 0.0
    cos_data_pde = 0.0
    cos_data_bc = 0.0
    cos_pde_bc = 0.0
    l_pde_val = 0.0
    l_bc_val = 0.0

    if getattr(loss_fn, "is_physics_aware", False):
        t_mean, t_std = loss_fn._get_t_scales(device)
        k_raw, q_raw, bc_channels = loss_fn._extract_physics_inputs(batch, device)
        t_pred_k = y_pred * t_std + t_mean

        if loss_fn.pde_weight > 0.0:
            l_pde = loss_fn.pde_loss_fn(
                temperature_k=t_pred_k,
                k_field=k_raw,
                q_field=q_raw,
                bc_channels=bc_channels,
                t_mean=t_mean,
                t_std=t_std,
            )
            l_pde_val = float(l_pde.detach().item())
            g_pde = compute_gradient_vector(l_pde, model)
            norm_pde_weighted = float(loss_fn.pde_weight * torch.linalg.norm(g_pde))
            cos_data_pde = cosine_similarity(g_data, g_pde)

        if loss_fn.bc_weight > 0.0:
            l_bc = loss_fn.bc_loss_fn(
                temperature_k=t_pred_k,
                k_field=k_raw,
                bc_channels=bc_channels,
                t_mean=t_mean,
                t_std=t_std,
            )
            l_bc_val = float(l_bc.detach().item())
            g_bc = compute_gradient_vector(l_bc, model)
            norm_bc_weighted = float(loss_fn.bc_weight * torch.linalg.norm(g_bc))
            cos_data_bc = cosine_similarity(g_data, g_bc)

        if loss_fn.pde_weight > 0.0 and loss_fn.bc_weight > 0.0:
            cos_pde_bc = cosine_similarity(g_pde, g_bc)

    if was_training:
        model.train()

    return {
        "loss_data": float(l_data.detach().item()),
        "loss_pde": l_pde_val,
        "loss_bc": l_bc_val,
        "grad_data_norm": norm_data,
        "grad_pde_weighted_norm": norm_pde_weighted,
        "grad_bc_weighted_norm": norm_bc_weighted,
        "pde_to_data_ratio": norm_pde_weighted / (norm_data + 1e-12),
        "bc_to_data_ratio": norm_bc_weighted / (norm_data + 1e-12),
        "cos_data_pde": cos_data_pde,
        "cos_data_bc": cos_data_bc,
        "cos_pde_bc": cos_pde_bc,
    }


def evaluate_split(
    model: torch.nn.Module,
    dataset: HeatFieldDataset,
    grid: Grid2D,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate a trained model across an entire dataset split."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    sample_metrics_list = []
    residual_metrics_list = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_pred = model(x).cpu()
            y_pred_np = y_pred.squeeze(0).squeeze(0).numpy()

            if dataset.normalizer is not None and dataset.normalizer.is_fitted:
                t_pred_k = dataset.normalizer.inverse_transform_temperature(y_pred_np)
                t_ref_k = batch["temperature_raw"].squeeze(0).squeeze(0).numpy()
            else:
                t_pred_k = y_pred_np
                t_ref_k = batch["y"].squeeze(0).squeeze(0).numpy()

            # Error metrics in Kelvin
            m = compute_all_metrics(t_pred_k, t_ref_k)
            sample_metrics_list.append(m)

            # Physical PDE residual and flux imbalance in Watts
            x_np = x.squeeze(0).cpu().numpy()
            bcs = decode_boundary_conditions(x_np, grid=grid, normalizer=dataset.normalizer)
            k_field = batch["k_raw"].squeeze(0).squeeze(0).numpy()
            q_field = batch["q_raw"].squeeze(0).squeeze(0).numpy()

            res_diag = evaluate_physical_residuals(
                grid=grid,
                k_field=k_field,
                q_field=q_field,
                bcs=bcs,
                temperature_pred=t_pred_k,
            )
            residual_metrics_list.append(res_diag)

    # Compute aggregate mean and std across samples in split
    agg_err = compute_aggregate_statistics(sample_metrics_list)

    pde_res_vals = [r["pde_residual_l2_w"] for r in residual_metrics_list]
    flux_imb_vals = [r["flux_imbalance_w"] for r in residual_metrics_list]

    return {
        "relative_l2_mean": agg_err["relative_l2"]["mean"],
        "relative_l2_std": agg_err["relative_l2"]["std"],
        "rmse_kelvin_mean": agg_err["rmse_k"]["mean"],
        "rmse_kelvin_std": agg_err["rmse_k"]["std"],
        "mae_kelvin_mean": agg_err["mae_k"]["mean"],
        "mae_kelvin_std": agg_err["mae_k"]["std"],
        "max_error_kelvin_mean": agg_err["max_error_k"]["mean"],
        "max_error_kelvin_std": agg_err["max_error_k"]["std"],
        "pde_residual_w_mean": float(np.mean(pde_res_vals)),
        "pde_residual_w_std": float(np.std(pde_res_vals)),
        "flux_imbalance_w_mean": float(np.mean(flux_imb_vals)),
        "flux_imbalance_w_std": float(np.std(flux_imb_vals)),
    }


def run_experiment(
    config_path: Path,
    seed: int,
    dataset_path: Path = Path("artifacts/data/scaled_dataset.h5"),
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train and evaluate a single experiment run for a specific seed."""
    cfg = load_training_config(config_path)
    exp_stem = config_path.stem
    run_name = f"{exp_stem}_seed{seed}"
    logger.info(f"=== Starting Run: {run_name} ===")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")  # CPU execution

    # Datasets
    train_dataset = HeatFieldDataset(
        dataset_path, split="train", include_coords=True, include_bcs=True
    )
    val_dataset = HeatFieldDataset(dataset_path, split="val", include_coords=True, include_bcs=True)
    test_iid_dataset = HeatFieldDataset(
        dataset_path, split="test_iid", include_coords=True, include_bcs=True
    )
    test_ood_dataset = HeatFieldDataset(
        dataset_path, split="test_ood", include_coords=True, include_bcs=True
    )

    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=train_dataset.nx, ny=train_dataset.ny)

    batch_size = cfg.training.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Diagnostic probe batch (representative fixed batch)
    probe_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    diag_batch = next(iter(probe_loader))

    # Model
    model_cfg = {
        "architecture": cfg.model.architecture,
        "in_channels": len(CANONICAL_CHANNEL_NAMES),
        "out_channels": 1,
        "width": cfg.model.width,
        "depth": cfg.model.depth,
    }
    if cfg.model.architecture == "fno2d":
        model_cfg["modes_x"] = cfg.model.modes_x
        model_cfg["modes_y"] = cfg.model.modes_y

    model = build_model(model_cfg).to(device)
    param_counts = model.count_parameters()

    # Loss function
    loss_fn = build_loss(cfg.loss, grid=grid, normalizer=train_dataset.normalizer).to(device)

    # Optimizer & Scheduler
    total_epochs = epochs or cfg.training.epochs
    optimizer = AdamW(model.parameters(), lr=cfg.optimization.learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    ckpt_dir = Path(f"artifacts/checkpoints/{run_name}")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=ckpt_dir,
        checkpoint_frequency=10,
        gradient_clip_norm=cfg.training.gradient_clip_norm,
        patience=cfg.training.early_stopping.patience
        if cfg.training.early_stopping.enabled
        else None,
        model_config=model_cfg,
        normalizer_stats=train_dataset.normalizer.to_dict() if train_dataset.normalizer else None,
    )

    # Stage 1: Beginning gradient diagnostics (Epoch 0 / Before training)
    logger.info("Recording gradient diagnostics at beginning of training...")
    grad_diag_beginning = capture_gradient_diagnostics(model, loss_fn, diag_batch, device)

    # Custom training loop to capture middle and end diagnostics
    t_start = time.perf_counter()
    grad_diag_middle = None
    grad_diag_end = None

    epochs_without_improvement = 0
    middle_epoch = total_epochs // 2

    for ep in range(1, total_epochs + 1):
        t_ep_start = time.perf_counter()
        train_loss = trainer.train_epoch(ep)
        val_loss = trainer.validate()
        t_ep = time.perf_counter() - t_ep_start

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        trainer.history["train_loss"].append(train_loss)
        trainer.history["val_loss"].append(val_loss)
        trainer.history["train_loss_data"].append(
            trainer.last_train_components.get("loss_data", train_loss)
        )
        trainer.history["train_loss_pde"].append(trainer.last_train_components.get("loss_pde", 0.0))
        trainer.history["train_loss_bc"].append(trainer.last_train_components.get("loss_bc", 0.0))
        trainer.history["val_loss_data"].append(
            trainer.last_val_components.get("loss_data", val_loss)
        )
        trainer.history["val_loss_pde"].append(trainer.last_val_components.get("loss_pde", 0.0))
        trainer.history["val_loss_bc"].append(trainer.last_val_components.get("loss_bc", 0.0))
        trainer.history["grad_norm"].append(trainer.last_grad_norm)
        trainer.history["optimizer_steps"].append(trainer.optimizer_steps)
        trainer.history["lr"].append(current_lr)
        trainer.history["epoch_times"].append(t_ep)

        # Middle diagnostics
        if ep == middle_epoch:
            logger.info(f"Recording gradient diagnostics at middle of training (epoch {ep})...")
            grad_diag_middle = capture_gradient_diagnostics(model, loss_fn, diag_batch, device)

        # Checkpoint selection strictly on validation loss
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            trainer.best_epoch = ep
            epochs_without_improvement = 0
            from neural_process_twin.training.checkpoint import save_checkpoint

            save_checkpoint(
                path=ckpt_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=ep,
                val_metric=val_loss,
                model_config=model_cfg,
                normalizer_stats=train_dataset.normalizer.to_dict()
                if train_dataset.normalizer
                else None,
                metadata={
                    "is_best": True,
                    "early_stopping_metric": "val_loss",
                    "optimizer_steps": trainer.optimizer_steps,
                    "best_epoch": ep,
                },
            )
        else:
            epochs_without_improvement += 1

        if (
            cfg.training.early_stopping.enabled
            and epochs_without_improvement >= cfg.training.early_stopping.patience
        ):
            logger.info(
                f"Early stopping triggered at epoch {ep} (patience={cfg.training.early_stopping.patience})."
            )
            break

    total_training_time = time.perf_counter() - t_start
    stopping_epoch = len(trainer.history["train_loss"])

    # End diagnostics
    logger.info("Recording gradient diagnostics at end of training...")
    grad_diag_end = capture_gradient_diagnostics(model, loss_fn, diag_batch, device)
    if grad_diag_middle is None:
        grad_diag_middle = grad_diag_end

    # Stage 2: Load the verified best checkpoint for evaluation
    best_ckpt_path = ckpt_dir / "best_model.pt"
    ckpt_dict = load_checkpoint(best_ckpt_path, device=device)
    eval_model = ckpt_dict["model"]
    ckpt_info = ckpt_dict
    logger.info(
        f"Loaded best checkpoint from epoch {ckpt_info.get('epoch')} "
        f"(val_loss={ckpt_info.get('val_metric'):.6f}, steps={ckpt_info.get('metadata', {}).get('optimizer_steps')})"
    )

    # Stage 3: Independent evaluation across val, test_iid, and test_ood
    val_results = evaluate_split(eval_model, val_dataset, grid, device)
    iid_results = evaluate_split(eval_model, test_iid_dataset, grid, device)
    ood_results = evaluate_split(eval_model, test_ood_dataset, grid, device)

    report = {
        "run_name": run_name,
        "experiment_stem": exp_stem,
        "seed": seed,
        "model_architecture": cfg.model.architecture,
        "trainable_parameters": param_counts["trainable_params"],
        "loss_weights": {
            "w_data": float(cfg.loss.data_loss.weight),
            "w_pde": float(cfg.loss.physics_loss.weight) if cfg.loss.physics_loss.enabled else 0.0,
            "w_bc": float(cfg.loss.physics_loss.bc_weight)
            if cfg.loss.physics_loss.enabled
            else 0.0,
            "residual_type": cfg.loss.physics_loss.residual_type,
        },
        "training_summary": {
            "total_epochs": total_epochs,
            "epochs_completed": stopping_epoch,
            "best_epoch": trainer.best_epoch,
            "total_optimizer_steps": trainer.optimizer_steps,
            "training_time_seconds": total_training_time,
            "best_val_loss": trainer.best_val_loss,
            "checkpoint_evaluated": str(best_ckpt_path),
        },
        "gradient_telemetry": {
            "beginning": grad_diag_beginning,
            "middle": grad_diag_middle,
            "end": grad_diag_end,
        },
        "evaluation": {
            "validation": val_results,
            "test_iid": iid_results,
            "test_ood": ood_results,
        },
        "history": trainer.history,
    }

    # Save run report
    rep_dir = Path("artifacts/reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    out_rep_path = rep_dir / f"{run_name}.json"
    with open(out_rep_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved run report to {out_rep_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Milestone 04B scaled training ablations.")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to evaluate."
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "scaled_cnn_supervised",
            "scaled_cnn_physics",
            "scaled_cnn_physics_bc",
            "scaled_fno_supervised",
            "scaled_fno_physics",
            "scaled_fno_physics_bc",
        ],
        help="Model config names to run.",
    )
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/data/scaled_dataset.h5"))
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    all_reports: dict[str, list[dict[str, Any]]] = {m: [] for m in args.models}

    for model_name in args.models:
        cfg_file = Path(f"configs/experiments/{model_name}.yaml")
        if not cfg_file.is_file():
            logger.error(f"Config file not found: {cfg_file}")
            continue

        for seed in args.seeds:
            rep = run_experiment(
                config_path=cfg_file,
                seed=seed,
                dataset_path=args.dataset,
                epochs=args.epochs,
            )
            all_reports[model_name].append(rep)

    # Compute master aggregated statistics across seeds
    aggregated: dict[str, Any] = {}
    for model_name, reps in all_reports.items():
        if not reps:
            continue

        splits = ["validation", "test_iid", "test_ood"]
        metrics = [
            "relative_l2_mean",
            "rmse_kelvin_mean",
            "mae_kelvin_mean",
            "max_error_kelvin_mean",
            "pde_residual_w_mean",
            "flux_imbalance_w_mean",
        ]

        split_agg: dict[str, dict[str, dict[str, float]]] = {}
        for s in splits:
            split_agg[s] = {}
            for m in metrics:
                vals = [r["evaluation"][s][m] for r in reps]
                split_agg[s][m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }

        best_epochs = [r["training_summary"]["best_epoch"] for r in reps]
        stopping_epochs = [r["training_summary"]["epochs_completed"] for r in reps]
        train_times = [r["training_summary"]["training_time_seconds"] for r in reps]
        steps = [r["training_summary"]["total_optimizer_steps"] for r in reps]

        aggregated[model_name] = {
            "model_architecture": reps[0]["model_architecture"],
            "trainable_parameters": reps[0]["trainable_parameters"],
            "loss_weights": reps[0]["loss_weights"],
            "num_seeds": len(reps),
            "seeds": [r["seed"] for r in reps],
            "training_summary": {
                "mean_best_epoch": float(np.mean(best_epochs)),
                "std_best_epoch": float(np.std(best_epochs)),
                "mean_stopping_epoch": float(np.mean(stopping_epochs)),
                "mean_training_time_s": float(np.mean(train_times)),
                "mean_optimizer_steps": float(np.mean(steps)),
            },
            "evaluation_by_split": split_agg,
        }

    agg_path = Path("artifacts/reports/milestone_04b_aggregated_results.json")
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"Master aggregated results saved to {agg_path}")
    print("\nMaster Aggregated Results:")
    print(json.dumps(aggregated, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
