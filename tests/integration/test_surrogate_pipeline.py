"""Integration tests for end-to-end surrogate training, evaluation, and CLI automation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from torch.optim import AdamW
from torch.utils.data import DataLoader

from neural_process_twin.data.boundary_encoding import NUM_CANONICAL_CHANNELS
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.evaluation.evaluator import evaluate_all_splits, evaluate_surrogate
from neural_process_twin.models.cnn import CNNBaseline2D
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.training.loss import RelativeL2Loss
from neural_process_twin.training.trainer import Trainer

SMOKE_DATASET_PATH = Path("artifacts/data/smoke_dataset.h5")


@pytest.mark.skipif(not SMOKE_DATASET_PATH.is_file(), reason="Smoke dataset archive not found.")
class TestSurrogatePipelineIntegration:
    """End-to-end pipeline integration testing on synthetic HDF5 datasets."""

    def test_cnn_and_fno_smoke_train_and_eval(self, tmp_path: Path) -> None:
        """Verify training, checkpointing, and evaluation of CNN and FNO surrogates."""
        train_ds = HeatFieldDataset(
            file_path=SMOKE_DATASET_PATH,
            split="train",
            include_coords=True,
            include_bcs=True,
        )
        val_ds = HeatFieldDataset(
            file_path=SMOKE_DATASET_PATH,
            split="val",
            include_coords=True,
            include_bcs=True,
        )

        try:
            sample = train_ds[0]
            assert sample["x"].shape[0] == NUM_CANONICAL_CHANNELS
            assert sample["y"].shape[0] == 1

            train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

            # 1. Train CNNBaseline2D
            cnn_dir = tmp_path / "cnn_run"
            cnn = CNNBaseline2D(
                in_channels=NUM_CANONICAL_CHANNELS, out_channels=1, width=16, depth=2
            )
            trainer_cnn = Trainer(
                model=cnn,
                optimizer=AdamW(cnn.parameters(), lr=1e-3),
                loss_fn=RelativeL2Loss(),
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=cnn_dir,
                checkpoint_frequency=1,
            )
            history_cnn = trainer_cnn.fit(epochs=2)
            assert len(history_cnn["train_loss"]) == 2
            assert (cnn_dir / "best_model.pt").is_file()

            # 2. Train FourierNeuralOperator2D
            fno_dir = tmp_path / "fno_run"
            fno = FourierNeuralOperator2D(
                in_channels=NUM_CANONICAL_CHANNELS,
                out_channels=1,
                modes_x=6,
                modes_y=6,
                width=16,
                depth=2,
            )
            trainer_fno = Trainer(
                model=fno,
                optimizer=AdamW(fno.parameters(), lr=1e-3),
                loss_fn=RelativeL2Loss(),
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=fno_dir,
                checkpoint_frequency=1,
            )
            history_fno = trainer_fno.fit(epochs=2)
            assert len(history_fno["train_loss"]) == 2
            assert (fno_dir / "best_model.pt").is_file()

            # 3. Evaluate on validation partition
            eval_result = evaluate_surrogate(
                model=cnn,
                dataset=val_ds,
                split_name="val",
                device="cpu",
                compute_residuals=True,
                max_residual_samples=5,
            )
            assert eval_result.num_samples == len(val_ds)
            assert "relative_l2" in eval_result.aggregate_metrics
            assert "rmse_k" in eval_result.aggregate_metrics
            assert eval_result.aggregate_metrics["rmse_k"]["mean"] > 0.0

            # 4. Multi-split evaluation across val, test_iid, test_ood
            all_results = evaluate_all_splits(
                model=fno,
                dataset_path=SMOKE_DATASET_PATH,
                splits=("val", "test_iid", "test_ood"),
                device="cpu",
                compute_residuals=True,
            )
            assert "val" in all_results
            assert "test_iid" in all_results
            assert "test_ood" in all_results
            for _split_name, res in all_results.items():
                assert res.num_samples > 0
                assert res.aggregate_metrics["relative_l2"]["mean"] > 0.0
                assert res.residual_metrics is not None
                assert "pde_residual_l2_w" in res.residual_metrics

        finally:
            train_ds.close()
            val_ds.close()

    def test_cli_scripts_execution(self, tmp_path: Path) -> None:
        """Verify train, evaluate, and benchmark CLI scripts execute cleanly with returncode 0."""
        chk_dir = tmp_path / "cli_checkpoints"

        # 1. scripts/train.py
        cmd_train = [
            sys.executable,
            "scripts/train.py",
            "--smoke-test",
            "--model",
            "cnn2d",
            "--dataset",
            str(SMOKE_DATASET_PATH),
            "--output-dir",
            str(chk_dir),
            "--device",
            "cpu",
        ]
        proc_train = subprocess.run(cmd_train, capture_output=True, text=True, check=False)
        assert proc_train.returncode == 0, f"train.py failed: {proc_train.stderr}"
        best_chk = chk_dir / "best_model.pt"
        assert best_chk.is_file()

        # 2. scripts/evaluate.py
        cmd_eval = [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(best_chk),
            "--dataset",
            str(SMOKE_DATASET_PATH),
            "--splits",
            "val",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "eval_report.json"),
        ]
        proc_eval = subprocess.run(cmd_eval, capture_output=True, text=True, check=False)
        assert proc_eval.returncode == 0, f"evaluate.py failed: {proc_eval.stderr}"
        assert (tmp_path / "eval_report.json").is_file()

        # 3. scripts/benchmark.py
        cmd_bench = [
            sys.executable,
            "scripts/benchmark.py",
            "--grid-size",
            "16",
            "16",
            "--repeats",
            "5",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "bench_report.json"),
        ]
        proc_bench = subprocess.run(cmd_bench, capture_output=True, text=True, check=False)
        assert proc_bench.returncode == 0, f"benchmark.py failed: {proc_bench.stderr}"
        assert (tmp_path / "bench_report.json").is_file()
