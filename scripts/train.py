#!/usr/bin/env python3
"""Surrogate model training CLI for Neural Process Twin.

Trains Fourier Neural Operator (FNO-2D) and CNN baseline models on synthetic
heat conduction datasets with canonical 12-channel physical representations,
validation-based checkpointing, learning rate scheduling, and early stopping.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from neural_process_twin.data.boundary_encoding import CANONICAL_CHANNEL_NAMES
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.models.factory import build_model
from neural_process_twin.training.loss import RelativeL2Loss, build_loss
from neural_process_twin.training.trainer import Trainer
from neural_process_twin.utils.config import load_training_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train surrogate neural operator models on 2D heat conduction datasets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training.yaml"),
        help="Path to YAML training configuration (default: configs/training.yaml).",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/data/smoke_dataset.h5"),
        help="Path to HDF5 training dataset (default: artifacts/data/smoke_dataset.h5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["fno2d", "cnn2d", "cnn"],
        help="Override model architecture type ('fno2d' or 'cnn2d').",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override mini-batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override initial learning rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device ('auto', 'cpu', or 'cuda').",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for saved checkpoints (default: artifacts/checkpoints/<model_arch>).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Execute rapid smoke test (2 epochs, minimal batch size) for pipeline verification.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Determine execution device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Set random seeds
    torch.manual_seed(args.seed)

    # Load configuration if present
    cfg = None
    if args.config.is_file():
        logger.info(f"Loading configuration from {args.config}")
        cfg = load_training_config(args.config)

    # Determine architecture
    arch = args.model or (cfg.model.architecture if cfg else "fno2d")
    arch = "cnn2d" if arch == "cnn" else arch

    epochs = args.epochs or (cfg.training.epochs if cfg else 50)
    batch_size = args.batch_size or (cfg.training.batch_size if cfg else 16)
    lr = args.lr or (cfg.optimization.learning_rate if cfg else 1e-3)

    if args.smoke_test:
        epochs = 2
        batch_size = min(batch_size, 4)
        logger.info(f"Smoke test mode enabled: epochs={epochs}, batch_size={batch_size}")

    if not args.dataset.is_file():
        logger.error(f"Dataset file not found at {args.dataset}")
        return 1

    logger.info(f"Loading datasets from {args.dataset} with 12-channel canonical representation...")
    train_dataset = HeatFieldDataset(
        file_path=args.dataset,
        split="train",
        include_coords=True,
        include_bcs=True,
    )
    val_dataset = HeatFieldDataset(
        file_path=args.dataset,
        split="val",
        include_coords=True,
        include_bcs=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=0,
    )

    # Build model with 12 canonical input channels
    in_channels = len(CANONICAL_CHANNEL_NAMES)  # 12
    out_channels = 1

    model_config = {
        "architecture": arch,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "width": cfg.model.width if cfg else 32,
        "depth": cfg.model.depth if cfg else 4,
    }
    if arch == "fno2d":
        nx, ny = train_dataset.nx, train_dataset.ny
        max_mx = nx // 2 + 1
        max_my = ny // 2
        mx = min(cfg.model.modes_x if cfg else 12, max_mx)
        my = min(cfg.model.modes_y if cfg else 12, max_my)
        model_config["modes_x"] = mx
        model_config["modes_y"] = my

    model = build_model(model_config)
    param_counts = model.count_parameters()
    logger.info(
        f"Built {arch} surrogate model with {param_counts['trainable_params']:,} trainable parameters."
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    if cfg is not None:
        loss_fn = build_loss(
            cfg.loss,
            grid=train_dataset.grid,
            normalizer=train_dataset.normalizer,
        )
    else:
        loss_fn = RelativeL2Loss()

    # Output directory
    output_dir = args.output_dir or Path(f"artifacts/checkpoints/{arch}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalizer statistics extraction for checkpoint
    norm_stats = {}
    if train_dataset.normalizer is not None and train_dataset.normalizer.is_fitted:
        norm = train_dataset.normalizer
        if norm.t_stats:
            norm_stats["t_mean"] = norm.t_stats.mean
            norm_stats["t_std"] = norm.t_stats.std
            norm_stats["t_min"] = norm.t_stats.min
            norm_stats["t_max"] = norm.t_stats.max

    # Trainer instantiation
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        device=device,
        gradient_clip_norm=1.0,
        checkpoint_dir=output_dir,
        checkpoint_frequency=max(1, epochs // 5),
        patience=15,
        normalizer_stats=norm_stats,
        model_config=model_config,
    )

    logger.info(f"Commencing training: {epochs} epochs on {len(train_dataset)} training samples...")
    t0 = time.perf_counter()
    _ = trainer.fit(epochs=epochs)
    total_time = time.perf_counter() - t0

    logger.info(
        f"Training completed in {total_time:.2f}s. "
        f"Best epoch: {trainer.best_epoch}, Best val loss: {trainer.best_val_loss:.6e}."
    )
    logger.info(f"Best checkpoint saved to: {output_dir / 'best_model.pt'}")

    # Close open dataset handles
    train_dataset.close()
    val_dataset.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
