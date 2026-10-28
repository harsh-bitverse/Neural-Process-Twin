"""Training pipeline for Milestone 08F candidate neural operators.

Trains controlled candidate models under matched data, resolution, and optimization budgets:
- A1_loss_refined_fno: Standard FNO trained with RefinedLocalizedLoss (penalizing smooth-max, boundary, and interface gradient errors)
- A3_hybrid_refined_fno: Coupled Global-Local RefinedFNO2D trained with RefinedLocalizedLoss
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
from neural_process_twin.models.refined_fno import RefinedFNO2D
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.training.refined_loss import RefinedLocalizedLoss

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_milestone_08f")


def set_seed(seed: int) -> None:
    """Set random seed for PyTorch, NumPy, and thread execution."""
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
        use_log_k: bool = True,
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
                weight_west = (1.0 - x_norm)[np.newaxis, :, :]
                weight_east = x_norm[np.newaxis, :, :]
                weight_south = (1.0 - y_norm)[np.newaxis, :, :]
                weight_north = y_norm[np.newaxis, :, :]

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

                y_arr = normalizer.transform_temperature(t_all).astype(np.float32)
                y_arr = y_arr[:, np.newaxis, :, :]  # (N, 1, res, res)

                for i in range(n_samples):
                    self.samples.append(
                        {
                            "x": torch.from_numpy(x_arr[i]),
                            "y": torch.from_numpy(y_arr[i]),
                            "k_raw": torch.from_numpy(k_all[i : i + 1]).float(),
                            "resolution": res,
                        }
                    )

        logger.info(
            f"Loaded {len(self.samples)} samples for split='{split}' across resolutions {resolutions}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


class MultiResolutionBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Batch sampler ensuring uniform resolution within each minibatch."""

    def __init__(
        self,
        dataset: MultiResDataset,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.res_indices: dict[int, list[int]] = {}
        for idx, item in enumerate(dataset.samples):
            r = item["resolution"]
            self.res_indices.setdefault(r, []).append(idx)

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        all_batches: list[list[int]] = []

        for _r, indices in self.res_indices.items():
            inds = list(indices)
            if self.shuffle:
                rng.shuffle(inds)
            for i in range(0, len(inds), self.batch_size):
                batch = inds[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        return sum(len(inds) // self.batch_size for inds in self.res_indices.values())


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
    return normalizer, log_k_mean, log_k_std


def train_candidate(
    model_id: str,
    model_class: type[nn.Module],
    model_kwargs: dict[str, Any],
    h5_path: Path,
    output_dir: Path,
    normalizer: FieldNormalizer,
    log_k_mean: float,
    log_k_std: float,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Train a single candidate neural operator under matched budget."""
    set_seed(seed)
    device = torch.device("cpu")
    logger.info(f"--- Training Candidate: {model_id} (Seed={seed}, Epochs={epochs}) ---")

    ckpt_dir = output_dir / model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = MultiResDataset(
        h5_path=h5_path,
        split="train",
        resolutions=[32, 64],
        normalizer=normalizer,
        use_log_k=True,
        log_k_mean=log_k_mean,
        log_k_std=log_k_std,
    )

    val_ds_32 = MultiResDataset(
        h5_path=h5_path,
        split="validation",
        resolutions=[32],
        normalizer=normalizer,
        use_log_k=True,
        log_k_mean=log_k_mean,
        log_k_std=log_k_std,
    )

    val_ds_64 = MultiResDataset(
        h5_path=h5_path,
        split="validation",
        resolutions=[64],
        normalizer=normalizer,
        use_log_k=True,
        log_k_mean=log_k_mean,
        log_k_std=log_k_std,
    )

    batch_sampler = MultiResolutionBatchSampler(
        train_ds, batch_size=batch_size, shuffle=True, seed=seed
    )
    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler)
    val_loader_32 = DataLoader(val_ds_32, batch_size=batch_size, shuffle=False)
    val_loader_64 = DataLoader(val_ds_64, batch_size=batch_size, shuffle=False)

    model = model_class(**model_kwargs).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Refined loss formulation targeting MaxAE and boundary errors
    loss_fn = RefinedLocalizedLoss(
        lambda_max=0.05,
        lambda_bnd=0.10,
        lambda_grad=0.05,
        beta=2.0,
        bnd_ring_width=2,
    )

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
            k_raw = batch["k_raw"].to(device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss, _ = loss_fn(y_pred, y, k_field=k_raw)
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
                loss_fn(model(b["x"].to(device)), b["y"].to(device), b["k_raw"].to(device))[
                    0
                ].item()
                for b in val_loader_32
            ]
            val_losses_64 = [
                loss_fn(model(b["x"].to(device)), b["y"].to(device), b["k_raw"].to(device))[
                    0
                ].item()
                for b in val_loader_64
            ]
            mean_val_32 = float(np.mean(val_losses_32))
            mean_val_64 = float(np.mean(val_losses_64))

        combined_val = 0.5 * (mean_val_32 + mean_val_64)

        if combined_val < best_val_loss:
            best_val_loss = combined_val
            best_epoch = ep
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "model_id": model_id,
                    "model_class": model_class.__name__,
                    "model_kwargs": model_kwargs,
                    "normalizer": normalizer.to_dict(),
                    "log_k_mean": log_k_mean,
                    "log_k_std": log_k_std,
                    "param_count": param_count,
                    "seed": seed,
                },
                ckpt_dir / "best_model.pt",
            )

        if ep % 5 == 0 or ep == epochs:
            logger.info(
                f"[{model_id}] Ep {ep:02d}/{epochs} | Train Loss: {mean_train_loss:.5f} | "
                f"Val-32: {mean_val_32:.5f} | Val-64: {mean_val_64:.5f} | Best Ep: {best_epoch:02d} (LR: {curr_lr:.2e})"
            )

    wall_time = time.perf_counter() - t0
    logger.info(f"Training completed for {model_id} in {wall_time:.1f}s. Best Epoch: {best_epoch}")

    return {
        "model_id": model_id,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "wall_time_s": wall_time,
        "param_count": param_count,
        "checkpoint_path": str(ckpt_dir / "best_model.pt"),
    }


def main() -> int:
    """Execute training of Milestone 08F candidate neural operators."""
    parser = argparse.ArgumentParser(description="Train Milestone 08F candidate models")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/multires_dataset.h5"),
        help="Path to multiresolution HDF5 dataset",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs per candidate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size per step",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="all",
        choices=["all", "A1", "A3"],
        help="Specific candidate to train (A1, A3, or all)",
    )
    args = parser.parse_args()

    output_dir = Path("artifacts/checkpoints/milestone_08f")
    output_dir.mkdir(parents=True, exist_ok=True)

    normalizer, log_k_mean, log_k_std = compute_normalization_statistics(args.dataset)

    candidates: list[dict[str, Any]] = [
        {
            "model_id": "A1_loss_refined_fno",
            "model_class": FourierNeuralOperator2D,
            "model_kwargs": {
                "in_channels": 12,
                "out_channels": 1,
                "modes_x": 12,
                "modes_y": 12,
                "width": 32,
                "depth": 4,
                "activation": "gelu",
            },
        },
        {
            "model_id": "A3_hybrid_refined_fno",
            "model_class": RefinedFNO2D,
            "model_kwargs": {
                "in_channels": 12,
                "out_channels": 1,
                "modes_x": 12,
                "modes_y": 12,
                "width": 32,
                "depth": 4,
                "refinement_hidden": 32,
                "refinement_layers": 3,
                "activation": "gelu",
            },
        },
    ]

    if args.candidate == "A1":
        candidates = [c for c in candidates if "A1" in c["model_id"]]
    elif args.candidate == "A3":
        candidates = [c for c in candidates if "A3" in c["model_id"]]

    summary: dict[str, Any] = {}
    for cand in candidates:
        res = train_candidate(
            model_id=cand["model_id"],
            model_class=cand["model_class"],
            model_kwargs=cand["model_kwargs"],
            h5_path=args.dataset,
            output_dir=output_dir,
            normalizer=normalizer,
            log_k_mean=log_k_mean,
            log_k_std=log_k_std,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        summary[cand["model_id"]] = res

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Milestone 08F training complete. Checkpoints saved to artifacts/checkpoints/milestone_08f/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
