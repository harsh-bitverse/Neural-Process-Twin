"""Milestone 08B.1 Phase 3: Loss-Weight Calibration Sweep and Pareto Analysis.

Performs a controlled parameter sweep of boundary loss weight w_bc in:
    [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
using strictly the training (N=1000) and validation (N=150) partitions of scaled_dataset.h5.

Evaluates on the validation partition:
- Validation Temperature RMSE (Overall, Interior, Boundary) [K]
- Validation Relative L2 Error [%]
- Validation Maximum Absolute Error [K]
- Validation Boundary Heat Flow Conservation Error (Flux Imbalance) [W]
- Validation Discrete PDE Residual (A*T - b) [W]

Constructs the empirical Pareto curve to determine optimal w_bc* before touching test partitions.
Outputs:
- Checkpoints in artifacts/checkpoints/milestone_08b1/
- Results in artifacts/reports/milestone_08b1_loss_calibration.json
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
logger = logging.getLogger("milestone_08b1_calibration")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_wbc_model(
    w_bc: float,
    seed: int,
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> Path:
    """Train FNO with given w_bc and return checkpoint path."""
    set_seed(seed)
    dev = torch.device(device)
    save_dir = output_dir / f"fno_wbc_{w_bc}_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_model.pt"

    if best_ckpt.is_file():
        logger.info(f"Checkpoint already exists: {best_ckpt}. Skipping training.")
        return best_ckpt

    logger.info(f"--- Training FNO with w_bc={w_bc} (Seed {seed}) for {epochs} epochs ---")

    train_ds = HeatFieldDataset(dataset_path, split="train", include_coords=True, include_bcs=True)
    val_ds = HeatFieldDataset(
        dataset_path, split="validation", include_coords=True, include_bcs=True
    )
    grid = train_ds.grid
    normalizer = train_ds.normalizer

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FourierNeuralOperator2D(
        in_channels=12,
        out_channels=1,
        modes_x=12,
        modes_y=12,
        width=32,
        depth=4,
        activation="gelu",
    )
    model.to(dev)

    if w_bc > 0.0:
        loss_fn = FaceConsistentPhysicsLoss(
            grid=grid,
            normalizer=normalizer,
            data_weight=1.0,
            bc_weight=w_bc,
            weight_dirichlet=1.0,
            weight_neumann=1.0,
            weight_flux_balance=0.1,
            delta_t_char=normalizer.t_stats.std if normalizer and normalizer.t_stats else 50.0,
        )
    else:
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
            "architecture": "fno2d",
            "modes_x": 12,
            "modes_y": 12,
            "width": 32,
            "depth": 4,
            "w_bc": w_bc,
        },
    )

    t0 = time.perf_counter()
    trainer.fit(epochs=epochs)
    elapsed = time.perf_counter() - t0
    logger.info(
        f"Completed w_bc={w_bc} in {elapsed:.1f}s | Best Val Loss: {trainer.best_val_loss:.6f} at Epoch {trainer.best_epoch}"
    )

    return best_ckpt


def evaluate_on_validation_split(
    checkpoint_path: Path,
    dataset_path: Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate trained model on the validation split (N=150) across physical & numerical metrics."""
    dev = torch.device(device)
    val_ds = HeatFieldDataset(
        dataset_path, split="validation", include_coords=True, include_bcs=True
    )
    grid = val_ds.grid
    normalizer = val_ds.normalizer

    if normalizer is None or normalizer.t_stats is None:
        raise ValueError("Normalizer stats missing from dataset.")

    t_mean = float(normalizer.t_stats.mean)
    t_std = float(normalizer.t_stats.std)

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model = FourierNeuralOperator2D(
        in_channels=12,
        out_channels=1,
        modes_x=12,
        modes_y=12,
        width=32,
        depth=4,
        activation="gelu",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)
    model.eval()

    pde_op = DiscretePDEResidual(grid=grid)

    overall_rmses: list[float] = []
    interior_rmses: list[float] = []
    boundary_rmses: list[float] = []
    rel_l2s: list[float] = []
    max_aes: list[float] = []
    flux_imbalances: list[float] = []
    rel_imbalances: list[float] = []
    pde_residuals: list[float] = []

    with torch.no_grad():
        for i in range(len(val_ds)):
            sample = val_ds[i]
            x_tensor = sample["x"].unsqueeze(0).to(dev)  # (1, 12, ny, nx)
            t_ref_k = sample["temperature_raw"].squeeze().numpy()  # (ny, nx)
            k_raw = sample["k_raw"].squeeze().numpy()
            q_raw = sample["q_raw"].squeeze().numpy()

            # Predict
            pred_norm = model(x_tensor).squeeze(0).squeeze(0).cpu().numpy()
            pred_k = pred_norm * t_std + t_mean

            # Errors
            diff = pred_k - t_ref_k
            overall_rmse = float(np.sqrt(np.mean(diff**2)))
            overall_rmses.append(overall_rmse)

            # Interior vs Boundary
            interior_diff = diff[1:-1, 1:-1]
            interior_rmse = float(np.sqrt(np.mean(interior_diff**2)))
            interior_rmses.append(interior_rmse)

            # Perimeter mask
            bnd_mask = np.ones_like(diff, dtype=bool)
            bnd_mask[1:-1, 1:-1] = False
            bnd_rmse = float(np.sqrt(np.mean(diff[bnd_mask] ** 2)))
            boundary_rmses.append(bnd_rmse)

            # Relative L2 (%)
            rel_l2 = float(np.linalg.norm(diff) / np.linalg.norm(t_ref_k) * 100.0)
            rel_l2s.append(rel_l2)

            # Max Absolute Error (K)
            max_ae = float(np.max(np.abs(diff)))
            max_aes.append(max_ae)

            # Physical flux balance
            bc_tensor_np = sample["x"][4:12].numpy()
            bcs = decode_boundary_conditions(bc_tensor_np, grid=grid, normalizer=normalizer)
            diag = compute_heat_balance(grid, pred_k, k_raw, q_raw, bcs)
            flux_imbalances.append(float(diag.flux_imbalance))
            rel_imbalances.append(float(diag.relative_imbalance * 100.0))

            # Discrete PDE residual (Watts)
            temp_k_tensor = torch.from_numpy(pred_k).float().unsqueeze(0).unsqueeze(0).to(dev)
            k_raw_tensor = sample["k_raw"].unsqueeze(0).to(dev)
            q_raw_tensor = sample["q_raw"].unsqueeze(0).to(dev)
            bc_chan_tensor = sample["x"][4:12].unsqueeze(0).to(dev)

            r_tensor, _ = pde_op(
                temperature_k=temp_k_tensor,
                k_field=k_raw_tensor,
                q_field=q_raw_tensor,
                bc_channels=bc_chan_tensor,
                t_mean=t_mean,
                t_std=t_std,
            )
            r_l2 = float(torch.norm(r_tensor).item())
            pde_residuals.append(r_l2)

    def stats(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "median": float(np.median(a)),
        }

    return {
        "sample_count": len(val_ds),
        "overall_rmse_k": stats(overall_rmses),
        "interior_rmse_k": stats(interior_rmses),
        "boundary_rmse_k": stats(boundary_rmses),
        "relative_l2_pct": stats(rel_l2s),
        "max_ae_k": stats(max_aes),
        "flux_imbalance_w": stats(flux_imbalances),
        "relative_imbalance_pct": stats(rel_imbalances),
        "discrete_pde_residual_w": stats(pde_residuals),
    }


def run_loss_calibration_sweep() -> dict[str, Any]:
    dataset_path = Path("artifacts/data/scaled_dataset.h5")
    output_dir = Path("artifacts/checkpoints/milestone_08b1")
    output_dir.mkdir(parents=True, exist_ok=True)

    w_bc_candidates = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    seed = 42

    sweep_results: dict[str, Any] = {}

    for w_bc in w_bc_candidates:
        logger.info(f"=== Starting calibration candidate w_bc = {w_bc} ===")
        ckpt_path = train_wbc_model(
            w_bc=w_bc,
            seed=seed,
            dataset_path=dataset_path,
            output_dir=output_dir,
            epochs=30,
            batch_size=32,
        )
        val_metrics = evaluate_on_validation_split(
            checkpoint_path=ckpt_path,
            dataset_path=dataset_path,
        )
        sweep_results[str(w_bc)] = {
            "w_bc": w_bc,
            "seed": seed,
            "checkpoint": str(ckpt_path),
            "validation_metrics": val_metrics,
        }
        logger.info(
            f"w_bc={w_bc} Results -> "
            f"Val RMSE: {val_metrics['overall_rmse_k']['mean']:.3f} K | "
            f"Boundary RMSE: {val_metrics['boundary_rmse_k']['mean']:.3f} K | "
            f"Flux Imbal: {val_metrics['flux_imbalance_w']['mean']:.1f} W | "
            f"PDE Res: {val_metrics['discrete_pde_residual_w']['mean']:.1f} W"
        )

    # Save results
    report_path = Path("artifacts/reports/milestone_08b1_loss_calibration.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
    logger.info(f"Loss calibration sweep saved to {report_path}")

    return sweep_results


if __name__ == "__main__":
    run_loss_calibration_sweep()
