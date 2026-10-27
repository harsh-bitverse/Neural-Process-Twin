#!/usr/bin/env python3
"""Milestone 08B — Controlled Experimental Evaluation.

Executes rigorous comparison across three operator configurations:
- A0: Historical frozen supervised FNO baseline (scaled_fno_supervised)
- A1: FNO trained with Face-Consistent Boundary Loss (fno_face_consistent)
- A2: Boundary-Aware FNO with discrete boundary correction (boundary_aware_fno)

Evaluates on:
- test_iid (N=200)
- test_ood (N=200 total)
- test_ood High-Contrast Inclusion subgroup (N=153)
- test_ood Short-Correlation GRF subgroup (N=47, where the 59.6 kW flux anomaly occurs)

Outputs:
- Trained checkpoints in artifacts/checkpoints/milestone_08b/
- Comprehensive JSON results in artifacts/reports/milestone_08b_results.json
"""

from __future__ import annotations

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

from neural_process_twin.data.boundary_encoding import decode_boundary_conditions
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.models.boundary_aware_fno import BoundaryAwareFNO2D
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.training.face_consistent_loss import FaceConsistentPhysicsLoss
from neural_process_twin.training.loss import RelativeL2Loss
from neural_process_twin.training.physics_loss import DiscretePDEResidual
from neural_process_twin.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("milestone_08b")


def set_seed(seed: int) -> None:
    """Set deterministic seeds for PyTorch and NumPy."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_model(
    model_name: str,
    arch_type: str,
    seed: int,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> Path:
    """Train a single model under controlled conditions and return best checkpoint path."""
    set_seed(seed)
    dev = torch.device(device)
    save_dir = output_dir / f"{model_name}_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_model.pt"

    # Reuse existing trained checkpoint if already present
    if best_ckpt.is_file():
        logger.info(f"Checkpoint already exists: {best_ckpt}. Skipping training.")
        return best_ckpt

    logger.info(f"--- Training {model_name} (Seed {seed}) for {epochs} epochs ---")

    train_ds = HeatFieldDataset(dataset_path, split="train", include_coords=True, include_bcs=True)
    val_ds = HeatFieldDataset(
        dataset_path, split="validation", include_coords=True, include_bcs=True
    )
    grid = train_ds.grid
    normalizer = train_ds.normalizer

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model
    if arch_type == "fno2d":
        model = FourierNeuralOperator2D(
            in_channels=12,
            out_channels=1,
            modes_x=12,
            modes_y=12,
            width=32,
            depth=4,
            activation="gelu",
        )
    elif arch_type == "boundary_aware_fno":
        model = BoundaryAwareFNO2D(
            grid=grid,
            in_channels=12,
            out_channels=1,
            modes_x=12,
            modes_y=12,
            width=32,
            depth=4,
            activation="gelu",
            use_boundary_bypass=True,
            correction_mode="hard",
        )
    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")

    model.to(dev)

    # Configure loss function
    if model_name.startswith("fno_face_consistent"):
        loss_fn = FaceConsistentPhysicsLoss(
            grid=grid,
            normalizer=normalizer,
            data_weight=1.0,
            bc_weight=0.05,
            weight_dirichlet=1.0,
            weight_neumann=1.0,
            weight_flux_balance=0.1,
            delta_t_char=normalizer.t_stats.std if normalizer and normalizer.t_stats else 50.0,
        )
    else:
        # Standard supervised Relative L2 loss
        loss_fn = RelativeL2Loss()

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=dev,
        gradient_clip_norm=1.0,
        checkpoint_dir=save_dir,
        checkpoint_frequency=10,
        patience=10,
        normalizer_stats=normalizer.to_dict() if normalizer else None,
        model_config={
            "architecture": arch_type,
            "modes_x": 12,
            "modes_y": 12,
            "width": 32,
            "depth": 4,
            "in_channels": 12,
            "out_channels": 1,
        },
    )

    t0 = time.perf_counter()
    trainer.fit(epochs=epochs)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Completed {model_name} seed {seed}: Best val loss = {trainer.best_val_loss:.6f} "
        f"at epoch {trainer.best_epoch} (Elapsed: {elapsed:.1f} s)"
    )

    return best_ckpt


def evaluate_model_on_split(
    model: torch.nn.Module,
    dataset: HeatFieldDataset,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate temperature errors, boundary heat flows, flux imbalances, and residuals."""
    dev = device or torch.device("cpu")
    model.eval()
    grid = dataset.grid
    normalizer = dataset.normalizer
    t_mean = normalizer.t_stats.mean if normalizer and normalizer.t_stats else 0.0
    t_std = normalizer.t_stats.std if normalizer and normalizer.t_stats else 1.0

    pde_op = DiscretePDEResidual(grid=grid)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    latencies_ms: list[float] = []
    overall_rmse_list: list[float] = []
    interior_rmse_list: list[float] = []
    boundary_rmse_list: list[float] = []
    rel_l2_list: list[float] = []
    max_ae_list: list[float] = []

    flux_imbalance_abs_list: list[float] = []
    flux_imbalance_signed_list: list[float] = []
    rel_imbalance_list: list[float] = []
    q_gen_list: list[float] = []
    q_out_list: list[float] = []
    q_left_list: list[float] = []
    q_right_list: list[float] = []
    q_bottom_list: list[float] = []
    q_top_list: list[float] = []

    pde_res_l2_list: list[float] = []
    contrast_list: list[float] = []

    # Mask for interior vs boundary control volumes
    ny, nx = grid.ny, grid.nx
    interior_mask = np.zeros((ny, nx), dtype=bool)
    interior_mask[1:-1, 1:-1] = True
    boundary_mask = ~interior_mask

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(dev)
            y_true_norm = batch["y"].to(dev)
            k_raw = batch["k_raw"].to(dev)
            q_raw = batch["q_raw"].to(dev)

            # Benchmark inference latency
            t0 = time.perf_counter()
            if isinstance(model, BoundaryAwareFNO2D):
                y_pred_norm = model(x, k_raw=k_raw, t_mean=t_mean, t_std=t_std)
            else:
                y_pred_norm = model(x)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            latencies_ms.append(dt_ms)

            # Convert to physical Kelvin
            t_pred_k = y_pred_norm * t_std + t_mean
            t_true_k = y_true_norm * t_std + t_mean

            t_pred_np = t_pred_k.squeeze().cpu().numpy()
            t_true_np = t_true_k.squeeze().cpu().numpy()
            k_np = k_raw.squeeze().cpu().numpy()
            q_np = q_raw.squeeze().cpu().numpy()

            # Contrast ratio
            contrast = float(np.max(k_np) / max(np.min(k_np), 1e-12))
            contrast_list.append(contrast)

            # 1. Error metrics
            diff = t_pred_np - t_true_np
            overall_rmse = float(np.sqrt(np.mean(diff**2)))
            interior_rmse = float(np.sqrt(np.mean(diff[interior_mask] ** 2)))
            boundary_rmse = float(np.sqrt(np.mean(diff[boundary_mask] ** 2)))
            max_ae = float(np.max(np.abs(diff)))
            rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(t_true_np) * 100.0)

            overall_rmse_list.append(overall_rmse)
            interior_rmse_list.append(interior_rmse)
            boundary_rmse_list.append(boundary_rmse)
            max_ae_list.append(max_ae)
            rel_l2_list.append(rel_l2)

            # 2. Reconstruct Boundary Heat Flows via Reference Solver Diagnostics
            bc_tensor_np = x[:, 4:12].squeeze().cpu().numpy()
            bcs = decode_boundary_conditions(bc_tensor_np, grid=grid, normalizer=normalizer)
            diag = compute_heat_balance(
                grid=grid,
                temperature=t_pred_np,
                k_field=k_np,
                q_field=q_np,
                bcs=bcs,
            )

            q_out = diag.q_boundary_out_total
            q_gen = diag.q_generated_total
            imbalance_signed = q_out - q_gen
            imbalance_abs = abs(imbalance_signed)
            rel_imbalance = (imbalance_abs / max(abs(q_gen), 1.0)) * 100.0

            q_gen_list.append(q_gen)
            q_out_list.append(q_out)
            flux_imbalance_abs_list.append(imbalance_abs)
            flux_imbalance_signed_list.append(imbalance_signed)
            rel_imbalance_list.append(rel_imbalance)

            q_left_list.append(diag.boundary_heat_flow_out.get("left", 0.0))
            q_right_list.append(diag.boundary_heat_flow_out.get("right", 0.0))
            q_bottom_list.append(diag.boundary_heat_flow_out.get("bottom", 0.0))
            q_top_list.append(diag.boundary_heat_flow_out.get("top", 0.0))

            # 3. Discrete PDE Residual
            bc_torch = x[:, 4:12]
            r_torch, _ = pde_op(
                temperature_k=t_pred_k,
                k_field=k_raw,
                q_field=q_raw,
                bc_channels=bc_torch,
                t_mean=t_mean,
                t_std=t_std,
                residual_type="full",
            )
            res_l2 = float(torch.linalg.norm(r_torch.squeeze()).item())
            pde_res_l2_list.append(res_l2)

    def stats(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0,
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
        }

    # Aggregate full split
    results: dict[str, Any] = {
        "sample_count": len(latencies_ms),
        "latency_ms": stats(latencies_ms),
        "overall_rmse_k": stats(overall_rmse_list),
        "interior_rmse_k": stats(interior_rmse_list),
        "boundary_rmse_k": stats(boundary_rmse_list),
        "relative_l2_pct": stats(rel_l2_list),
        "max_ae_k": stats(max_ae_list),
        "flux_imbalance_abs_w": stats(flux_imbalance_abs_list),
        "flux_imbalance_signed_w": stats(flux_imbalance_signed_list),
        "relative_imbalance_pct": stats(rel_imbalance_list),
        "q_generated_w": stats(q_gen_list),
        "q_boundary_out_w": stats(q_out_list),
        "q_left_w": stats(q_left_list),
        "q_right_w": stats(q_right_list),
        "q_bottom_w": stats(q_bottom_list),
        "q_top_w": stats(q_top_list),
        "pde_residual_l2_w": stats(pde_res_l2_list),
    }

    # Subgroup breakdown if contrast variation is present (e.g. on test_ood)
    contrasts = np.array(contrast_list)
    high_mask = contrasts >= 20.0
    grf_mask = contrasts < 20.0

    if np.any(high_mask) and np.any(grf_mask):
        subgroups: dict[str, Any] = {
            "high_contrast_inclusions": {
                "count": int(np.sum(high_mask)),
                "overall_rmse_k": stats(np.array(overall_rmse_list)[high_mask].tolist()),
                "interior_rmse_k": stats(np.array(interior_rmse_list)[high_mask].tolist()),
                "boundary_rmse_k": stats(np.array(boundary_rmse_list)[high_mask].tolist()),
                "relative_l2_pct": stats(np.array(rel_l2_list)[high_mask].tolist()),
                "max_ae_k": stats(np.array(max_ae_list)[high_mask].tolist()),
                "flux_imbalance_abs_w": stats(
                    np.array(flux_imbalance_abs_list)[high_mask].tolist()
                ),
                "flux_imbalance_signed_w": stats(
                    np.array(flux_imbalance_signed_list)[high_mask].tolist()
                ),
            },
            "short_correlation_grf": {
                "count": int(np.sum(grf_mask)),
                "overall_rmse_k": stats(np.array(overall_rmse_list)[grf_mask].tolist()),
                "interior_rmse_k": stats(np.array(interior_rmse_list)[grf_mask].tolist()),
                "boundary_rmse_k": stats(np.array(boundary_rmse_list)[grf_mask].tolist()),
                "relative_l2_pct": stats(np.array(rel_l2_list)[grf_mask].tolist()),
                "max_ae_k": stats(np.array(max_ae_list)[grf_mask].tolist()),
                "flux_imbalance_abs_w": stats(np.array(flux_imbalance_abs_list)[grf_mask].tolist()),
                "flux_imbalance_signed_w": stats(
                    np.array(flux_imbalance_signed_list)[grf_mask].tolist()
                ),
                "q_left_w": stats(np.array(q_left_list)[grf_mask].tolist()),
                "q_right_w": stats(np.array(q_right_list)[grf_mask].tolist()),
            },
        }
        results["subgroups"] = subgroups

    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Milestone 08B Controlled Experiments")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to evaluate"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--configs", type=str, nargs="+", default=["A0", "A1", "A2"], help="Configs to run"
    )
    args = parser.parse_args()

    dataset_path = Path("artifacts/data/scaled_dataset.h5")
    if not dataset_path.is_file():
        logger.error(f"Dataset archive not found: {dataset_path}")
        return 1

    checkpoints_dir = Path("artifacts/checkpoints/milestone_08b")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("artifacts/reports")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "milestone_08b_results.json"

    device = torch.device("cpu")
    logger.info(
        f"Executing experiments on device: {device} | Seeds: {args.seeds} | Epochs: {args.epochs}"
    )

    # Seeds to evaluate
    seeds = args.seeds

    # All candidate configurations: (config_id, model_name, arch_type)
    all_configs = [
        ("A0", "scaled_fno_supervised", "fno2d"),
        ("A1", "fno_face_consistent", "fno2d"),
        ("A2", "boundary_aware_fno", "boundary_aware_fno"),
    ]
    configs = [c for c in all_configs if c[0] in args.configs]

    all_results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(dataset_path),
        "configurations": {},
    }
    if results_file.is_file():
        try:
            with open(results_file, encoding="utf-8") as f:
                existing = json.load(f)
                if "configurations" in existing:
                    all_results["configurations"].update(existing["configurations"])
        except Exception as e:
            logger.warning(f"Could not read existing results: {e}")

    # Load test datasets
    test_iid = HeatFieldDataset(
        dataset_path, split="test_iid", include_coords=True, include_bcs=True
    )
    test_ood = HeatFieldDataset(
        dataset_path, split="test_ood", include_coords=True, include_bcs=True
    )
    grid = test_iid.grid

    for config_id, model_name, arch_type in configs:
        logger.info("==================================================")
        logger.info(f"Processing Configuration {config_id}: {model_name}")
        logger.info("==================================================")

        config_data: dict[str, Any] = {
            "config_id": config_id,
            "model_name": model_name,
            "architecture": arch_type,
            "seeds": {},
        }

        for seed in seeds:
            # Checkpoint location
            if config_id == "A0":
                # Historical baseline checkpoints
                ckpt_path = Path(f"artifacts/checkpoints/{model_name}_seed{seed}/best_model.pt")
            else:
                ckpt_path = train_single_model(
                    model_name=model_name,
                    arch_type=arch_type,
                    seed=seed,
                    dataset_path=dataset_path,
                    output_dir=checkpoints_dir,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device="cpu",
                )

            if not ckpt_path.is_file():
                logger.error(f"Checkpoint not found for {model_name} seed {seed}: {ckpt_path}")
                continue

            # Load checkpoint
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)

            # Instantiate model
            if arch_type == "fno2d":
                model = FourierNeuralOperator2D(
                    in_channels=12,
                    out_channels=1,
                    modes_x=12,
                    modes_y=12,
                    width=32,
                    depth=4,
                    activation="gelu",
                )
            elif arch_type == "boundary_aware_fno":
                model = BoundaryAwareFNO2D(
                    grid=grid,
                    in_channels=12,
                    out_channels=1,
                    modes_x=12,
                    modes_y=12,
                    width=32,
                    depth=4,
                    activation="gelu",
                    use_boundary_bypass=True,
                    correction_mode="hard",
                )
            else:
                raise ValueError(arch_type)

            model.load_state_dict(state_dict)
            model.to(device)

            param_count = sum(p.numel() for p in model.parameters())

            # Evaluate on IID and OOD
            logger.info(f"Evaluating {model_name} seed {seed} on test_iid...")
            res_iid = evaluate_model_on_split(model, test_iid, device=device)

            logger.info(f"Evaluating {model_name} seed {seed} on test_ood...")
            res_ood = evaluate_model_on_split(model, test_ood, device=device)

            seed_entry = {
                "checkpoint": str(ckpt_path),
                "parameter_count": param_count,
                "test_iid": res_iid,
                "test_ood": res_ood,
            }
            config_data["seeds"][str(seed)] = seed_entry

            logger.info(
                f"[{config_id}|Seed {seed}] IID RMSE: {res_iid['overall_rmse_k']['mean']:.3f} K "
                f"(Boundary: {res_iid['boundary_rmse_k']['mean']:.3f} K, "
                f"Imbalance: {res_iid['flux_imbalance_abs_w']['mean']:.1f} W) | "
                f"OOD RMSE: {res_ood['overall_rmse_k']['mean']:.3f} K "
                f"(GRF Imbalance: {res_ood['subgroups']['short_correlation_grf']['flux_imbalance_abs_w']['mean']:.1f} W)"
            )

        # Aggregate across seeds
        seed_keys = list(config_data["seeds"].keys())
        if seed_keys:

            def agg_seeds(
                split_name: str,
                metric_path: list[str],
                s_keys: list[str] = seed_keys,
                c_data: dict[str, Any] = config_data,
            ) -> dict[str, float]:
                vals = []
                for sk in s_keys:
                    curr = c_data["seeds"][sk][split_name]
                    for p in metric_path:
                        curr = curr[p]
                    vals.append(float(curr))
                return {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }

            config_data["ensemble_summary"] = {
                "test_iid": {
                    "overall_rmse_k": agg_seeds("test_iid", ["overall_rmse_k", "mean"]),
                    "interior_rmse_k": agg_seeds("test_iid", ["interior_rmse_k", "mean"]),
                    "boundary_rmse_k": agg_seeds("test_iid", ["boundary_rmse_k", "mean"]),
                    "relative_l2_pct": agg_seeds("test_iid", ["relative_l2_pct", "mean"]),
                    "max_ae_k": agg_seeds("test_iid", ["max_ae_k", "mean"]),
                    "flux_imbalance_abs_w": agg_seeds("test_iid", ["flux_imbalance_abs_w", "mean"]),
                    "flux_imbalance_signed_w": agg_seeds(
                        "test_iid", ["flux_imbalance_signed_w", "mean"]
                    ),
                    "pde_residual_l2_w": agg_seeds("test_iid", ["pde_residual_l2_w", "mean"]),
                    "latency_ms": agg_seeds("test_iid", ["latency_ms", "mean"]),
                },
                "test_ood": {
                    "overall_rmse_k": agg_seeds("test_ood", ["overall_rmse_k", "mean"]),
                    "interior_rmse_k": agg_seeds("test_ood", ["interior_rmse_k", "mean"]),
                    "boundary_rmse_k": agg_seeds("test_ood", ["boundary_rmse_k", "mean"]),
                    "relative_l2_pct": agg_seeds("test_ood", ["relative_l2_pct", "mean"]),
                    "max_ae_k": agg_seeds("test_ood", ["max_ae_k", "mean"]),
                    "flux_imbalance_abs_w": agg_seeds("test_ood", ["flux_imbalance_abs_w", "mean"]),
                    "flux_imbalance_signed_w": agg_seeds(
                        "test_ood", ["flux_imbalance_signed_w", "mean"]
                    ),
                    "high_contrast_rmse_k": agg_seeds(
                        "test_ood",
                        ["subgroups", "high_contrast_inclusions", "overall_rmse_k", "mean"],
                    ),
                    "high_contrast_imbalance_w": agg_seeds(
                        "test_ood",
                        ["subgroups", "high_contrast_inclusions", "flux_imbalance_abs_w", "mean"],
                    ),
                    "grf_rmse_k": agg_seeds(
                        "test_ood", ["subgroups", "short_correlation_grf", "overall_rmse_k", "mean"]
                    ),
                    "grf_imbalance_w": agg_seeds(
                        "test_ood",
                        ["subgroups", "short_correlation_grf", "flux_imbalance_abs_w", "mean"],
                    ),
                    "pde_residual_l2_w": agg_seeds("test_ood", ["pde_residual_l2_w", "mean"]),
                },
            }

        all_results["configurations"][config_id] = config_data

    # Save to disk
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results successfully saved to {results_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
