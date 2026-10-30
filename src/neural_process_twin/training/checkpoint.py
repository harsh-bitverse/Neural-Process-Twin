"""Checkpoint persistence and restoration for surrogate models.

Saves and loads complete reproducible model states, including model weights,
optimizer state, scheduler state, model architecture hyperparameters, input channel
names, normalization statistics, and runtime environment metadata.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from neural_process_twin.data.boundary_encoding import CANONICAL_CHANNEL_NAMES
from neural_process_twin.models.base import SurrogateModel
from neural_process_twin.models.factory import build_model


def save_checkpoint(
    path: Path | str,
    model: SurrogateModel | nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    epoch: int = 0,
    val_metric: float | None = None,
    model_config: dict[str, Any] | None = None,
    channel_names: list[str] | None = None,
    normalizer_stats: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save model checkpoint and experimental reproducibility metadata.

    Args:
        path: Destination file path (e.g. 'checkpoints/best_model.pt').
        model: Trained surrogate model.
        optimizer: Optimizer instance (optional).
        scheduler: Learning rate scheduler instance (optional).
        epoch: Current training epoch index.
        val_metric: Best validation metric achieved.
        model_config: Dictionary of architecture hyperparameters.
        channel_names: Ordered list of input channel names. Defaults to CANONICAL_CHANNEL_NAMES.
        normalizer_stats: Statistics for physical unit inversion (mean/std or min/max).
        metadata: Arbitrary run metadata (git hash, run ID, notes).

    Returns:
        Path to saved checkpoint.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Infer model config if not provided
    if model_config is None:
        arch_name = "fno2d" if "Fourier" in type(model).__name__ else "cnn2d"
        model_config = {
            "architecture": arch_name,
            "in_channels": getattr(model, "in_channels", 12),
            "out_channels": getattr(model, "out_channels", 1),
            "width": getattr(model, "width", 32),
            "depth": getattr(model, "depth", 4),
        }
        if hasattr(model, "modes_x"):
            model_config["modes_x"] = model.modes_x
        if hasattr(model, "modes_y"):
            model_config["modes_y"] = model.modes_y

    env_metadata = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    if metadata:
        env_metadata.update(metadata)

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "val_metric": val_metric,
        "model_config": model_config,
        "channel_names": channel_names or CANONICAL_CHANNEL_NAMES,
        "normalizer_stats": normalizer_stats or {},
        "metadata": env_metadata,
    }

    torch.save(payload, target_path)
    return target_path


def load_checkpoint(
    path: Path | str,
    model: SurrogateModel | nn.Module | None = None,
    optimizer: Optimizer | None = None,
    scheduler: Any | None = None,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    """Load model checkpoint and restore state.

    If `model` is None and `model_config` is present in the checkpoint, this function
    will automatically instantiate the model using `build_model(checkpoint['model_config'])`.

    Args:
        path: Path to checkpoint file.
        model: Optional pre-instantiated model to populate.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        device: Device to map loaded tensors to.
        strict: Whether to enforce strict key matching in state_dict.

    Returns:
        Checkpoint dictionary containing loaded metadata, and instantiated `model` under 'model'.
    """
    chk_path = Path(path)
    if not chk_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {chk_path}")

    device_obj = torch.device(device) if isinstance(device, str) else device
    try:
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint: dict[str, Any] = torch.load(
                chk_path, map_location=device_obj, weights_only=True
            )
    except Exception:
        checkpoint = torch.load(chk_path, map_location=device_obj, weights_only=False)

    # Instantiate model if not provided
    if model is None:
        if "model_config" not in checkpoint or not checkpoint["model_config"]:
            raise ValueError(
                "Cannot automatically instantiate model: 'model_config' missing from checkpoint."
            )
        model = build_model(checkpoint["model_config"])

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    model.to(device_obj)
    checkpoint["model"] = model

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
