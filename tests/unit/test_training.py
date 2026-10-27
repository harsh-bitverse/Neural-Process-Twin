"""Unit tests for loss functions, checkpoint persistence, and training loops."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from neural_process_twin.models.cnn import CNNBaseline2D
from neural_process_twin.training.checkpoint import load_checkpoint, save_checkpoint
from neural_process_twin.training.loss import RelativeL2Loss, build_loss
from neural_process_twin.training.trainer import Trainer


class TestLossFormulations:
    """Test suite for RelativeL2Loss and build_loss."""

    def test_relative_l2_exact(self) -> None:
        loss_fn = RelativeL2Loss(reduction="mean")
        y_true = torch.full((1, 1, 4, 4), 2.0)
        y_pred = torch.full((1, 1, 4, 4), 3.0)

        # diff = 1.0 everywhere, diff_norm = sqrt(16 * 1.0) = 4.0
        # ref = 2.0 everywhere, ref_norm = sqrt(16 * 4.0) = 8.0
        # rel = 4.0 / 8.0 = 0.5
        loss = loss_fn(y_pred, y_true)
        assert pytest.approx(loss.item(), rel=1e-5) == 0.5

    def test_relative_l2_zero_error(self) -> None:
        loss_fn = RelativeL2Loss()
        y = torch.randn(2, 1, 8, 8)
        loss = loss_fn(y, y)
        assert pytest.approx(loss.item(), abs=1e-6) == 0.0

    def test_relative_l2_zero_reference_epsilon(self) -> None:
        loss_fn = RelativeL2Loss(eps=1e-4)
        y_true = torch.zeros(1, 1, 4, 4)
        y_pred = torch.ones(1, 1, 4, 4)
        loss = loss_fn(y_pred, y_true)
        assert torch.isfinite(loss)
        # diff_norm = 4.0, ref_norm = 0.0, loss = 4.0 / 1e-4 = 40000.0
        assert pytest.approx(loss.item(), rel=1e-3) == 40000.0

    def test_relative_l2_reductions(self) -> None:
        y_true = torch.tensor([[[[2.0]]], [[[4.0]]]])  # batch of 2
        y_pred = torch.tensor([[[[3.0]]], [[[6.0]]]])  # batch of 2
        # sample 0: diff=1, ref=2, rel=0.5
        # sample 1: diff=2, ref=4, rel=0.5

        loss_none = RelativeL2Loss(reduction="none")(y_pred, y_true)
        assert loss_none.shape == (2,)
        assert pytest.approx(loss_none[0].item(), rel=1e-5) == 0.5
        assert pytest.approx(loss_none[1].item(), rel=1e-5) == 0.5

        loss_sum = RelativeL2Loss(reduction="sum")(y_pred, y_true)
        assert pytest.approx(loss_sum.item(), rel=1e-5) == 1.0

        loss_mean = RelativeL2Loss(reduction="mean")(y_pred, y_true)
        assert pytest.approx(loss_mean.item(), rel=1e-5) == 0.5

    def test_build_loss(self) -> None:
        loss_rel = build_loss({"data_loss": {"type": "relative_l2"}})
        assert isinstance(loss_rel, RelativeL2Loss)

        loss_mse = build_loss({"data_loss": {"type": "mse"}})
        assert isinstance(loss_mse, nn.MSELoss)

        loss_l1 = build_loss({"data_loss": {"type": "l1"}})
        assert isinstance(loss_l1, nn.L1Loss)


class TestCheckpointPersistence:
    """Test suite for checkpoint save, load, and bitwise reproducibility."""

    def test_save_load_roundtrip_bitwise(self, tmp_path: Path) -> None:
        model = CNNBaseline2D(in_channels=12, out_channels=1, width=16, depth=2)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        chk_path = tmp_path / "checkpoint.pt"

        # Generate sample prediction before saving
        x = torch.randn(2, 12, 16, 16)
        with torch.no_grad():
            pred_before = model(x)

        # Save checkpoint
        save_checkpoint(
            path=chk_path,
            model=model,
            optimizer=optimizer,
            epoch=5,
            val_metric=0.042,
            metadata={"notes": "unit test checkpoint"},
        )
        assert chk_path.is_file()

        # Load checkpoint into new instance
        model_loaded = CNNBaseline2D(in_channels=12, out_channels=1, width=16, depth=2)
        chk_loaded = load_checkpoint(chk_path, model=model_loaded)

        assert chk_loaded["epoch"] == 5
        assert chk_loaded["val_metric"] == 0.042
        assert chk_loaded["metadata"]["notes"] == "unit test checkpoint"

        with torch.no_grad():
            pred_after = model_loaded(x)

        # Bitwise identical predictions
        assert torch.equal(pred_before, pred_after)

    def test_auto_model_instantiation_on_load(self, tmp_path: Path) -> None:
        model = CNNBaseline2D(in_channels=12, out_channels=1, width=16, depth=2)
        chk_path = tmp_path / "auto_chk.pt"
        save_checkpoint(path=chk_path, model=model, epoch=1)

        # Load passing model=None
        loaded_chk = load_checkpoint(chk_path, model=None)
        assert "model" in loaded_chk
        assert isinstance(loaded_chk["model"], CNNBaseline2D)


class TestTrainer:
    """Test suite for Trainer training loops, loss descent, and error checks."""

    def test_trainer_loss_decreases(self, tmp_path: Path) -> None:
        # Create tiny synthetic dataset
        torch.manual_seed(42)
        x = torch.randn(16, 12, 8, 8)
        # Target is linear transformation of channel 0
        y = 2.0 * x[:, 0:1, :, :] + 0.5
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = CNNBaseline2D(in_channels=12, out_channels=1, width=8, depth=2)
        optimizer = AdamW(model.parameters(), lr=0.01)
        loss_fn = RelativeL2Loss()

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_loader=loader,
            val_loader=loader,
            checkpoint_dir=tmp_path / "chk",
            patience=10,
        )

        history = trainer.fit(epochs=6)
        train_losses = history["train_loss"]

        assert len(train_losses) == 6
        # Loss after 6 epochs should be significantly lower than initial loss
        assert train_losses[-1] < train_losses[0]

    def test_non_finite_loss_raises_runtime_error(self) -> None:
        class NanModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.full_like(x[:, 0:1, :, :], float("nan"))

        x = torch.randn(4, 12, 8, 8)
        y = torch.randn(4, 1, 8, 8)
        loader = DataLoader(TensorDataset(x, y), batch_size=2)

        model = NanModel()
        optimizer = AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.MSELoss(),
            train_loader=loader,
        )

        with pytest.raises(RuntimeError, match="Non-finite loss"):
            trainer.train_epoch(epoch=1)

    def test_early_stopping_triggered(self, tmp_path: Path) -> None:
        x = torch.randn(8, 12, 8, 8)
        y = torch.randn(8, 1, 8, 8)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)

        # Frozen model where validation loss will never improve
        model = CNNBaseline2D(in_channels=12, out_channels=1, width=8, depth=2)
        optimizer = AdamW(model.parameters(), lr=0.0)  # Zero learning rate

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=RelativeL2Loss(),
            train_loader=loader,
            val_loader=loader,
            checkpoint_dir=tmp_path / "chk",
            patience=3,
        )

        history = trainer.fit(epochs=10)
        # Should stop after epoch 1 (best) + 3 without improvement = 4 epochs
        assert len(history["train_loss"]) <= 5
