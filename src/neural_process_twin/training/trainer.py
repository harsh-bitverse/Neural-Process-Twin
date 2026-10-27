"""Trainer implementation for supervised surrogate model optimization.

Manages training iterations, gradient norm clipping, learning rate scheduling,
validation monitoring, early stopping, and reproducible checkpoint persistence.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from neural_process_twin.training.checkpoint import save_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """End-to-end training manager for Neural Process Twin surrogate models.

    Coordinates training epochs, mini-batch forward/backward passes, validation
    evaluation, checkpoint management, and early stopping.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        scheduler: Any | None = None,
        device: str | torch.device = "cpu",
        gradient_clip_norm: float | None = None,
        checkpoint_dir: Path | str | None = None,
        checkpoint_frequency: int = 10,
        patience: int | None = None,
        normalizer_stats: dict[str, Any] | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Trainer.

        Args:
            model: Surrogate model instance to train.
            optimizer: PyTorch optimizer (Adam, AdamW, etc.).
            loss_fn: Loss function module (e.g. RelativeL2Loss or MSELoss).
            train_loader: DataLoader yielding training batches.
            val_loader: Optional DataLoader yielding validation batches.
            scheduler: Optional learning rate scheduler.
            device: Computing device ('cpu', 'cuda', etc.).
            gradient_clip_norm: Maximum gradient norm for clipping.
            checkpoint_dir: Directory where checkpoints will be saved.
            checkpoint_frequency: Frequency in epochs for saving periodic checkpoints.
            patience: Number of epochs to wait for validation improvement before early stopping.
            normalizer_stats: Field normalizer statistics to persist with checkpoints.
            model_config: Model architecture hyperparameters to persist with checkpoints.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_frequency = max(1, checkpoint_frequency)
        self.patience = patience
        self.normalizer_stats = normalizer_stats
        self.model_config = model_config

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.optimizer_steps = 0
        self.last_grad_norm = 0.0
        self.last_train_components: dict[str, float] = {}
        self.last_val_components: dict[str, float] = {}
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_loss_data": [],
            "train_loss_pde": [],
            "train_loss_bc": [],
            "val_loss_data": [],
            "val_loss_pde": [],
            "val_loss_bc": [],
            "grad_norm": [],
            "optimizer_steps": [],
            "lr": [],
            "epoch_times": [],
        }

    def _extract_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract (input, target) tensors from DataLoader batch."""
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
        elif isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}.")
        return x.to(self.device), y.to(self.device)

    def train_epoch(self, epoch: int) -> float:
        """Execute one training epoch across the training dataset.

        Args:
            epoch: Current epoch index (1-based).

        Returns:
            Mean training loss over all batches.

        Raises:
            RuntimeError: If NaN or infinite loss/gradient is detected.
        """
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        component_totals: dict[str, float] = {}
        num_batches = 0

        for batch in self.train_loader:
            x, y = self._extract_batch(batch)

            self.optimizer.zero_grad()
            y_pred = self.model(x)

            if getattr(self.loss_fn, "is_physics_aware", False) and isinstance(batch, dict):
                loss, comp_dict = self.loss_fn(y_pred, y, batch=batch)
            else:
                loss_out = self.loss_fn(y_pred, y)
                if isinstance(loss_out, tuple):
                    loss, comp_dict = loss_out
                else:
                    loss = loss_out
                    comp_dict = {"loss_total": float(loss.detach().item())}

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss ({loss.item()}) encountered at epoch {epoch}, batch {num_batches}."
                )

            loss.backward()

            if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
                total_grad_norm += (
                    float(gnorm.detach().cpu().item()) if hasattr(gnorm, "detach") else float(gnorm)
                )
            else:
                gnorm = torch.linalg.norm(
                    torch.stack(
                        [
                            torch.linalg.norm(p.grad.detach())
                            for p in self.model.parameters()
                            if p.grad is not None
                        ]
                    )
                )
                total_grad_norm += float(gnorm.detach().cpu().item())

            self.optimizer.step()
            self.optimizer_steps += 1

            total_loss += loss.item()
            for k_comp, v_comp in comp_dict.items():
                component_totals[k_comp] = component_totals.get(k_comp, 0.0) + float(v_comp)
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        self.last_grad_norm = total_grad_norm / max(1, num_batches)
        self.last_train_components = {
            k: v / max(1, num_batches) for k, v in component_totals.items()
        }
        return avg_loss

    def validate(self) -> float:
        """Evaluate current model performance across the validation dataset.

        Returns:
            Mean validation loss/metric.
        """
        if self.val_loader is None:
            return float("nan")

        self.model.eval()
        total_loss = 0.0
        component_totals: dict[str, float] = {}
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                x, y = self._extract_batch(batch)
                y_pred = self.model(x)

                if getattr(self.loss_fn, "is_physics_aware", False) and isinstance(batch, dict):
                    loss, comp_dict = self.loss_fn(y_pred, y, batch=batch)
                else:
                    loss_out = self.loss_fn(y_pred, y)
                    if isinstance(loss_out, tuple):
                        loss, comp_dict = loss_out
                    else:
                        loss = loss_out
                        comp_dict = {"loss_total": float(loss.detach().item())}

                total_loss += loss.item()
                for k_comp, v_comp in comp_dict.items():
                    component_totals[k_comp] = component_totals.get(k_comp, 0.0) + float(v_comp)
                num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        self.last_val_components = {k: v / max(1, num_batches) for k, v in component_totals.items()}
        return avg_loss

    def fit(self, epochs: int) -> dict[str, list[float]]:
        """Run complete training loop with validation, checkpointing, and early stopping.

        Args:
            epochs: Total number of training epochs to execute.

        Returns:
            Dictionary containing history of train_loss, val_loss, lr, and epoch_times.
        """
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            # Train single epoch
            train_loss = self.train_epoch(epoch)

            # Evaluate validation
            val_loss = self.validate()

            t_epoch = time.perf_counter() - t0

            # Current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_loss_data"].append(
                self.last_train_components.get("loss_data", train_loss)
            )
            self.history["train_loss_pde"].append(self.last_train_components.get("loss_pde", 0.0))
            self.history["train_loss_bc"].append(self.last_train_components.get("loss_bc", 0.0))
            self.history["val_loss_data"].append(
                self.last_val_components.get("loss_data", val_loss)
            )
            self.history["val_loss_pde"].append(self.last_val_components.get("loss_pde", 0.0))
            self.history["val_loss_bc"].append(self.last_val_components.get("loss_bc", 0.0))
            self.history["grad_norm"].append(self.last_grad_norm)
            self.history["optimizer_steps"].append(self.optimizer_steps)
            self.history["lr"].append(current_lr)
            self.history["epoch_times"].append(t_epoch)

            # Checkpoint management and early stopping evaluation
            if self.val_loader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                epochs_without_improvement = 0

                if self.checkpoint_dir is not None:
                    save_checkpoint(
                        path=self.checkpoint_dir / "best_model.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        val_metric=val_loss,
                        model_config=self.model_config,
                        normalizer_stats=self.normalizer_stats,
                        metadata={
                            "is_best": True,
                            "early_stopping_metric": "val_loss",
                            "optimizer_steps": self.optimizer_steps,
                            "best_epoch": epoch,
                        },
                    )
            elif self.val_loader is not None:
                epochs_without_improvement += 1

            # Periodic checkpoint
            if self.checkpoint_dir is not None and (epoch % self.checkpoint_frequency == 0):
                save_checkpoint(
                    path=self.checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_metric=val_loss,
                    model_config=self.model_config,
                    normalizer_stats=self.normalizer_stats,
                )

            # Early stopping check
            if self.patience is not None and epochs_without_improvement >= self.patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch} (patience={self.patience}, "
                    f"best_epoch={self.best_epoch}, best_val={self.best_val_loss:.6e})."
                )
                break

        return self.history
