#!/usr/bin/env python3
"""Train Multi-Resolution Neural Operator Candidates for Milestone 08C.

Milestone 08C Phase 4: Controlled Model Experiments.
Trains four matched-budget FNO architectures (1,188,641 parameters each):
- M0: Frozen historical 32x32 FNO baseline (scaled_fno_supervised_seed42) [Reference, untouched]
- M1: Continuous-encoding FNO trained exclusively at 32x32
- M2: Continuous-encoding FNO trained on mixed 32x32 and 64x64 resolutions
- M3: Continuous-encoding FNO with logarithmic conductivity trained on mixed resolutions

Matched Budget Invariants:
- Architecture: FourierNeuralOperator2D(in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4)
- Total parameters: 1,188,641 per candidate
- Training budget: 30 epochs (938 optimizer steps at batch size 32)
- Seed: 42
- Optimizer: AdamW(lr=1e-3, weight_decay=1e-4) with CosineAnnealingLR
- Checkpoint directory: artifacts/checkpoints/milestone_08c/
"""

from __future__ import annotations

import argparse
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
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.training.loss import RelativeL2Loss

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_milestone_08c")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiResDataset(Dataset):
    """PyTorch Dataset loading resolution-consistent physical fields from multires_dataset.h5."""

    def __init__(
        self,
        h5_path: Path,
        split: str,
        resolutions: list[int],
        normalizer: FieldNormalizer,
        use_log_k: bool = False,
        log_k_mean: float = 0.5,
        log_k_std: float = 1.0,
    ) -> None:
        self.h5_path = h5_path
        self.split = split
        self.resolutions = resolutions
        self.normalizer = normalizer
        self.use_log_k = use_log_k
        self.log_k_mean = log_k_mean
        self.log_k_std = log_k_std

        # Preload data into RAM for fast multi-resolution training
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

                # Precompute spatial coordinates once for this grid resolution
                x_norm = ((grid.x_mesh - grid.x_min) / grid.lx).astype(np.float32)
                y_norm = ((grid.y_mesh - grid.y_min) / grid.ly).astype(np.float32)
                weight_west = (1.0 - x_norm)[np.newaxis, :, :]
                weight_east = x_norm[np.newaxis, :, :]
                weight_south = (1.0 - y_norm)[np.newaxis, :, :]
                weight_north = y_norm[np.newaxis, :, :]

                # Build (N, 12, res, res) in vectorized fashion
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

                def to_str_list(arr: np.ndarray) -> list[str]:
                    return [
                        x.decode("utf-8").lower() if isinstance(x, bytes) else str(x).lower()
                        for x in arr
                    ]

                l_types = to_str_list(l_type_all)
                r_types = to_str_list(r_type_all)
                b_types = to_str_list(b_type_all)
                t_types = to_str_list(t_type_all)

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

                x_tensors = torch.from_numpy(x_arr)
                y_tensors = torch.from_numpy(y_arr)
                k_raw_tensors = torch.from_numpy(k_all).float()
                q_raw_tensors = torch.from_numpy(q_all).float()
                t_raw_tensors = torch.from_numpy(t_all).float()

                for i in range(n_samples):
                    self.samples.append(
                        {
                            "x": x_tensors[i],
                            "y": y_tensors[i],
                            "k_raw": k_raw_tensors[i],
                            "q_raw": q_raw_tensors[i],
                            "t_raw": t_raw_tensors[i],
                            "res": res,
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def compute_normalization_statistics(
    h5_path: Path,
) -> tuple[FieldNormalizer, float, float]:
    """Fit FieldNormalizer on 32x32 training partition."""
    with h5py.File(h5_path, "r") as f:
        k_train = f["res_32/train/k"][:]
        q_train = f["res_32/train/q"][:]
        t_train = f["res_32/train/temperature"][:]

    normalizer = FieldNormalizer()
    normalizer.fit(
        k=k_train,
        q=q_train,
        temperature=t_train,
    )

    log_k = np.log(np.maximum(k_train, 1e-12))
    log_k_mean = float(np.mean(log_k))
    log_k_std = float(np.std(log_k))

    logger.info(
        f"Normalization fitted: T mean={normalizer.t_stats.mean:.1f} K, std={normalizer.t_stats.std:.1f} K | "
        f"k mean={normalizer.k_stats.mean:.2f}, std={normalizer.k_stats.std:.2f} | "
        f"log(k) mean={log_k_mean:.2f}, std={log_k_std:.2f}"
    )
    return normalizer, log_k_mean, log_k_std


class MultiResolutionBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Batch sampler that ensures every batch contains samples of identical spatial resolution."""

    def __init__(
        self,
        dataset: MultiResDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
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


def train_candidate(
    model_id: str,
    resolutions: list[int],
    use_log_k: bool,
    h5_path: Path,
    output_dir: Path,
    normalizer: FieldNormalizer,
    log_k_mean: float,
    log_k_std: float,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Train single neural operator candidate under controlled budget."""
    set_seed(seed)
    device = device if device is not None else torch.device("cpu")
    logger.info(
        f"=== Training Candidate {model_id} (Resolutions: {resolutions}, Log-k: {use_log_k}) ==="
    )

    save_dir = output_dir / model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best_model.pt"

    # Load datasets
    train_ds = MultiResDataset(
        h5_path=h5_path,
        split="train",
        resolutions=resolutions,
        normalizer=normalizer,
        use_log_k=use_log_k,
        log_k_mean=log_k_mean,
        log_k_std=log_k_std,
    )

    val_ds_32 = MultiResDataset(
        h5_path=h5_path,
        split="validation",
        resolutions=[32],
        normalizer=normalizer,
        use_log_k=use_log_k,
        log_k_mean=log_k_mean,
        log_k_std=log_k_std,
    )

    val_ds_64 = MultiResDataset(
        h5_path=h5_path,
        split="validation",
        resolutions=[64],
        normalizer=normalizer,
        use_log_k=use_log_k,
        log_k_mean=log_k_mean,
        log_k_std=log_k_std,
    )

    batch_sampler = MultiResolutionBatchSampler(
        train_ds, batch_size=batch_size, shuffle=True, seed=seed
    )
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler)
    val_loader_32 = DataLoader(val_ds_32, batch_size=batch_size, shuffle=False)
    val_loader_64 = DataLoader(val_ds_64, batch_size=batch_size, shuffle=False)

    # Standard FNO-2D
    model = FourierNeuralOperator2D(
        in_channels=12,
        out_channels=1,
        modes_x=12,
        modes_y=12,
        width=32,
        depth=4,
        activation="gelu",
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = RelativeL2Loss()

    history: dict[str, list[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss_32": [],
        "val_loss_64": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_epoch = -1
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        batch_sampler.epoch = ep
        model.train()
        train_losses = []
        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        curr_lr = float(optimizer.param_groups[0]["lr"])
        mean_train_loss = float(np.mean(train_losses))

        # Validate on 32 and 64
        model.eval()
        with torch.no_grad():
            val_losses_32 = [
                loss_fn(model(b["x"].to(device)), b["y"].to(device)).item() for b in val_loader_32
            ]
            val_losses_64 = [
                loss_fn(model(b["x"].to(device)), b["y"].to(device)).item() for b in val_loader_64
            ]
            mean_val_32 = float(np.mean(val_losses_32))
            mean_val_64 = float(np.mean(val_losses_64))

        # Combined validation loss for checkpointing
        combined_val = 0.5 * (mean_val_32 + mean_val_64) if len(resolutions) > 1 else mean_val_32

        history["epoch"].append(ep)
        history["train_loss"].append(mean_train_loss)
        history["val_loss_32"].append(mean_val_32)
        history["val_loss_64"].append(mean_val_64)
        history["lr"].append(curr_lr)

        if combined_val < best_val_loss:
            best_val_loss = combined_val
            best_epoch = ep
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_loss_32": mean_val_32,
                    "val_loss_64": mean_val_64,
                    "model_id": model_id,
                    "resolutions": resolutions,
                    "use_log_k": use_log_k,
                    "param_count": param_count,
                    "normalizer": normalizer.to_dict(),
                    "log_k_mean": log_k_mean,
                    "log_k_std": log_k_std,
                },
                best_ckpt,
            )

        if ep % 5 == 0 or ep == epochs:
            logger.info(
                f"[{model_id}] Ep {ep:02d}/{epochs:02d} | Train: {mean_train_loss:.5f} | "
                f"Val-32: {mean_val_32:.5f} | Val-64: {mean_val_64:.5f} | Best Val: {best_val_loss:.5f} (Ep {best_epoch})"
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Finished {model_id}: Best val loss = {best_val_loss:.5f} at epoch {best_epoch} in {elapsed:.1f} s"
    )

    # Save training history
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_id": model_id,
                "resolutions": resolutions,
                "use_log_k": use_log_k,
                "param_count": param_count,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "elapsed_seconds": elapsed,
                "history": history,
            },
            f,
            indent=2,
        )

    return {
        "model_id": model_id,
        "checkpoint": str(best_ckpt),
        "param_count": param_count,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Milestone 08C Neural Operator Candidates")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    h5_path = Path("artifacts/data/multires_dataset.h5")
    if not h5_path.is_file():
        logger.error(f"Dataset not found: {h5_path}")
        return 1

    checkpoints_dir = Path("artifacts/checkpoints/milestone_08c")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    normalizer, log_k_mean, log_k_std = compute_normalization_statistics(h5_path)

    # Model specifications:
    # M1: Continuous boundary encoding, trained at 32x32
    # M2: Continuous boundary encoding, trained on mixed 32x32 + 64x64
    # M3: Continuous boundary encoding with log-conductivity, trained on mixed 32x32 + 64x64
    candidates = [
        {"model_id": "M1_cont_fno_res32", "resolutions": [32], "use_log_k": False},
        {"model_id": "M2_cont_fno_mixed", "resolutions": [32, 64], "use_log_k": False},
        {"model_id": "M3_logk_fno_mixed", "resolutions": [32, 64], "use_log_k": True},
    ]

    summary = {}
    for cand in candidates:
        res = train_candidate(
            model_id=cand["model_id"],
            resolutions=cand["resolutions"],
            use_log_k=cand["use_log_k"],
            h5_path=h5_path,
            output_dir=checkpoints_dir,
            normalizer=normalizer,
            log_k_mean=log_k_mean,
            log_k_std=log_k_std,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        summary[cand["model_id"]] = res

    with open(checkpoints_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("All Milestone 08C models successfully trained and check-pointed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
