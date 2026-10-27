#!/usr/bin/env python3
"""Train Efficient Uncertainty Model Candidates for Milestone 08D.

Milestone 08D Phase 3: Controlled Uncertainty Training.
Trains four matched-capacity uncertainty architectures on multi-resolution data:
- U1: Heteroscedastic FNO (Single-pass heteroscedastic scale head)
- U2: Deep Evidential Regression FNO (Normal-Inverse-Gamma prior with evidence regularizer)
- U3: Spatial Monte Carlo Dropout FNO (Stochastic epistemic sampling)
- U4: Distilled Student FNO (Single-pass epistemic student trained on residual scale)

Training Invariants:
- Partitions: train (1000 samples) on mixed 32x32 + 64x64, validation (150 samples) for model selection.
- Optimizer: AdamW(lr=1e-3, weight_decay=1e-4) with CosineAnnealingLR.
- Backbone initialization: Warm-started from frozen M3 feature layers for strict capacity parity.
- Preserves frozen M3 checkpoint untouched.
- Checkpoints persisted to artifacts/checkpoints/milestone_08d/{model_id}/best_model.pt.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.uncertainty_fno import (
    DistilledStudentFNO2D,
    EvidentialFNO2D,
    HeteroscedasticFNO2D,
    MCDropoutFNO2D,
)
from neural_process_twin.solvers.grid import Grid2D

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_milestone_08d")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)


class MultiResDataset(Dataset):
    """Loads resolution-consistent physical fields from multires_dataset.h5."""

    def __init__(
        self,
        h5_path: Path,
        split: str,
        resolutions: list[int],
        normalizer: FieldNormalizer,
        use_log_k: bool = True,
        log_k_mean: float = 0.5,
        log_k_std: float = 1.0,
    ) -> None:
        self.samples: list[dict[str, Any]] = []

        with h5py.File(h5_path, "r") as h5f:
            for res in resolutions:
                grp = h5f[f"res_{res}/{split}"]
                n_samples = grp["k"].shape[0]
                grid = Grid2D(0.0, 1.0, 0.0, 1.0, res, res, thickness=1.0)

                k_all = grp["k"][:]
                q_all = grp["q"][:]
                t_all = grp["temperature"][:]
                bc_grp = grp["boundary_conditions"]
                l_type_all = bc_grp["left_type"][:]
                r_type_all = bc_grp["right_type"][:]
                b_type_all = bc_grp["bottom_type"][:]
                t_type_all = bc_grp["top_type"][:]
                l_val_all = grp["boundary_conditions/left_value"][:]
                r_val_all = grp["boundary_conditions/right_value"][:]
                b_val_all = grp["boundary_conditions/bottom_value"][:]
                t_val_all = grp["boundary_conditions/top_value"][:]

                x_norm = ((grid.x_mesh - grid.x_min) / grid.lx).astype(np.float32)
                y_norm = ((grid.y_mesh - grid.y_min) / grid.ly).astype(np.float32)

                x_arr = np.zeros((n_samples, 12, res, res), dtype=np.float32)
                if use_log_k:
                    log_k = np.log(np.maximum(k_all, 1e-12))
                    x_arr[:, 0] = (
                        (log_k - log_k_mean) / (log_k_std if log_k_std > 1e-12 else 1.0)
                    ).astype(np.float32)
                else:
                    x_arr[:, 0] = normalizer.transform_k(k_all).astype(np.float32)

                x_arr[:, 1] = normalizer.transform_q(q_all).astype(np.float32)
                x_arr[:, 2] = x_norm[np.newaxis, :, :]
                x_arr[:, 3] = y_norm[np.newaxis, :, :]

                def to_str_list(arr: Any) -> list[str]:
                    return [
                        x.decode("utf-8").lower() if isinstance(x, bytes) else str(x).lower()
                        for x in arr
                    ]

                l_types = to_str_list(l_type_all)
                r_types = to_str_list(r_type_all)
                b_types = to_str_list(b_type_all)
                t_types = to_str_list(t_type_all)

                weight_west = (1.0 - x_norm)[np.newaxis, :, :]
                weight_east = x_norm[np.newaxis, :, :]
                weight_south = (1.0 - y_norm)[np.newaxis, :, :]
                weight_north = y_norm[np.newaxis, :, :]

                # West
                l_dir_mask = np.array([t == "dirichlet" for t in l_types])
                if np.any(l_dir_mask):
                    l_norm_dir = normalizer.transform_temperature(l_val_all[l_dir_mask]).astype(
                        np.float32
                    )
                    x_arr[l_dir_mask, 4] = weight_west * l_norm_dir[:, :, np.newaxis]
                if np.any(~l_dir_mask):
                    l_norm_neu = (l_val_all[~l_dir_mask] / 1000.0).astype(np.float32)
                    x_arr[~l_dir_mask, 6] = weight_west * l_norm_neu[:, :, np.newaxis]

                # East
                r_dir_mask = np.array([t == "dirichlet" for t in r_types])
                if np.any(r_dir_mask):
                    r_norm_dir = normalizer.transform_temperature(r_val_all[r_dir_mask]).astype(
                        np.float32
                    )
                    x_arr[r_dir_mask, 5] = weight_east * r_norm_dir[:, :, np.newaxis]
                if np.any(~r_dir_mask):
                    r_norm_neu = (r_val_all[~r_dir_mask] / 1000.0).astype(np.float32)
                    x_arr[~r_dir_mask, 7] = weight_east * r_norm_neu[:, :, np.newaxis]

                # South
                b_dir_mask = np.array([t == "dirichlet" for t in b_types])
                if np.any(b_dir_mask):
                    b_norm_dir = normalizer.transform_temperature(b_val_all[b_dir_mask]).astype(
                        np.float32
                    )
                    x_arr[b_dir_mask, 8] = weight_south * b_norm_dir[:, np.newaxis, :]
                if np.any(~b_dir_mask):
                    b_norm_neu = (b_val_all[~b_dir_mask] / 1000.0).astype(np.float32)
                    x_arr[~b_dir_mask, 10] = weight_south * b_norm_neu[:, np.newaxis, :]

                # North
                t_dir_mask = np.array([t == "dirichlet" for t in t_types])
                if np.any(t_dir_mask):
                    t_norm_dir = normalizer.transform_temperature(t_val_all[t_dir_mask]).astype(
                        np.float32
                    )
                    x_arr[t_dir_mask, 9] = weight_north * t_norm_dir[:, np.newaxis, :]
                if np.any(~t_dir_mask):
                    t_norm_neu = (t_val_all[~t_dir_mask] / 1000.0).astype(np.float32)
                    x_arr[~t_dir_mask, 11] = weight_north * t_norm_neu[:, np.newaxis, :]

                y_arr = normalizer.transform_temperature(t_all)[:, np.newaxis, :, :].astype(
                    np.float32
                )

                for i in range(n_samples):
                    self.samples.append(
                        {
                            "x": torch.from_numpy(x_arr[i]),
                            "y": torch.from_numpy(y_arr[i]),
                            "res": res,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class ResolutionBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Batches samples by identical resolution so tensors can be stacked."""

    def __init__(
        self,
        dataset: MultiResDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.res_to_indices: dict[int, list[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            res = sample["res"]
            if res not in self.res_to_indices:
                self.res_to_indices[res] = []
            self.res_to_indices[res].append(idx)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        batches: list[list[int]] = []
        for _res, indices in self.res_to_indices.items():
            inds = np.array(indices, copy=True)
            if self.shuffle:
                rng.shuffle(inds)
            for i in range(0, len(inds), self.batch_size):
                batch = inds[i : i + self.batch_size].tolist()
                if len(batch) > 0:
                    batches.append(batch)
        if self.shuffle:
            perm = rng.permutation(len(batches))
            batches = [batches[p] for p in perm]
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        total = 0
        for indices in self.res_to_indices.values():
            total += (len(indices) + self.batch_size - 1) // self.batch_size
        return total


def warm_start_backbone(
    target_model: nn.Module, source_state_dict: dict[str, torch.Tensor]
) -> None:
    """Warm-start common feature backbone layers (lifting, spectral, bypass) from frozen M3."""
    target_state = target_model.state_dict()
    transferred = 0
    for name, param in source_state_dict.items():
        if (
            name in target_state
            and target_state[name].shape == param.shape
            and not name.startswith("q_")
            and not name.startswith("q1")
        ):
            target_state[name].copy_(param)
            transferred += 1
    logger.info(f"Warm-started {transferred} backbone tensors from frozen M3.")


def train_single_model(
    model: nn.Module,
    model_id: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    normalizer: FieldNormalizer,
    epochs: int = 15,
    lr: float = 1e-3,
    device: torch.device | None = None,
    save_dir: Path = Path("artifacts/checkpoints/milestone_08d"),
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    logger.info(f"--- Training {model_id} for {epochs} epochs ---")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch = -1
    out_dir = save_dir / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()

            if model_id == "U1_heteroscedastic":
                mean, logvar = model(x)
                loss = HeteroscedasticFNO2D.nll_loss(mean, logvar, y)
            elif model_id == "U2_evidential":
                gamma, v, alpha, beta = model(x)
                loss = EvidentialFNO2D.evidential_loss(gamma, v, alpha, beta, y, lambda_reg=0.005)
            elif model_id == "U3_mc_dropout":
                pred = model(x)
                loss = nn.functional.smooth_l1_loss(pred, y) + 0.1 * nn.functional.mse_loss(pred, y)
            elif model_id == "U4_distilled_student":
                mean, std_pred = model(x)
                # Teacher residual supervision: target error |y - mean| provides student ground truth
                residual = torch.abs(y - mean.detach())
                loss = nn.functional.mse_loss(mean, y) + 0.5 * nn.functional.mse_loss(
                    std_pred, residual
                )
            else:
                raise ValueError(f"Unknown model_id: {model_id}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                if model_id == "U1_heteroscedastic":
                    mean, logvar = model(x)
                    v_loss = HeteroscedasticFNO2D.nll_loss(mean, logvar, y)
                elif model_id == "U2_evidential":
                    gamma, v, alpha, beta = model(x)
                    v_loss = EvidentialFNO2D.evidential_loss(
                        gamma, v, alpha, beta, y, lambda_reg=0.005
                    )
                elif model_id == "U3_mc_dropout":
                    pred = model(x)
                    v_loss = nn.functional.mse_loss(pred, y)
                elif model_id == "U4_distilled_student":
                    mean, std_pred = model(x)
                    residual = torch.abs(y - mean)
                    v_loss = nn.functional.mse_loss(mean, y) + 0.5 * nn.functional.mse_loss(
                        std_pred, residual
                    )
                val_losses.append(v_loss.item())

        mean_val = float(np.mean(val_losses))
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_epoch = epoch
            torch.save(
                {
                    "model_id": model_id,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "normalizer": normalizer.to_dict(),
                    "use_log_k": True,
                    "log_k_mean": 0.5,
                    "log_k_std": 1.0,
                },
                best_path,
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"{model_id} finished in {elapsed:.1f}s | Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}"
    )

    return {
        "model_id": model_id,
        "checkpoint": str(best_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
    }


def run_training_campaign() -> dict[str, Any]:
    """Execute complete controlled training campaign for Milestone 08D."""
    set_seed(42)
    logger.info("Initializing Milestone 08D Controlled Training Campaign...")

    # Load frozen M3 metadata and state dict
    m3_ckpt_path = Path("artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt")
    m3_ckpt = torch.load(m3_ckpt_path, map_location="cpu", weights_only=False)
    m3_state = m3_ckpt["model_state_dict"]
    normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])

    h5_path = Path("artifacts/data/multires_dataset.h5")

    # Load multi-res training and validation datasets
    logger.info("Loading multi-resolution training partition (mixed 32x32 + 64x64)...")
    train_dataset = MultiResDataset(
        h5_path=h5_path,
        split="train",
        resolutions=[32, 64],
        normalizer=normalizer,
        use_log_k=True,
    )
    val_dataset = MultiResDataset(
        h5_path=h5_path,
        split="validation",
        resolutions=[32, 64],
        normalizer=normalizer,
        use_log_k=True,
    )

    train_sampler = ResolutionBatchSampler(train_dataset, batch_size=32, shuffle=True)
    val_sampler = ResolutionBatchSampler(val_dataset, batch_size=32, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)

    summary: dict[str, Any] = {}

    # 1. U1: Heteroscedastic FNO
    u1 = HeteroscedasticFNO2D(in_channels=12, modes_x=12, modes_y=12, width=32, depth=4)
    warm_start_backbone(u1, m3_state)
    summary["U1_heteroscedastic"] = train_single_model(
        u1, "U1_heteroscedastic", train_loader, val_loader, normalizer, epochs=12
    )

    # 2. U2: Deep Evidential Regression FNO
    u2 = EvidentialFNO2D(in_channels=12, modes_x=12, modes_y=12, width=32, depth=4)
    warm_start_backbone(u2, m3_state)
    summary["U2_evidential"] = train_single_model(
        u2, "U2_evidential", train_loader, val_loader, normalizer, epochs=12
    )

    # 3. U3: MC Dropout FNO
    u3 = MCDropoutFNO2D(in_channels=12, modes_x=12, modes_y=12, width=32, depth=4, dropout_p=0.10)
    warm_start_backbone(u3, m3_state)
    summary["U3_mc_dropout"] = train_single_model(
        u3, "U3_mc_dropout", train_loader, val_loader, normalizer, epochs=12
    )

    # 4. U4: Distilled Student FNO
    u4 = DistilledStudentFNO2D(in_channels=12, modes_x=12, modes_y=12, width=32, depth=4)
    warm_start_backbone(u4, m3_state)
    summary["U4_distilled_student"] = train_single_model(
        u4, "U4_distilled_student", train_loader, val_loader, normalizer, epochs=12
    )

    out_json = Path("artifacts/checkpoints/milestone_08d/training_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved training summary to {out_json}")
    return summary


if __name__ == "__main__":
    run_training_campaign()
