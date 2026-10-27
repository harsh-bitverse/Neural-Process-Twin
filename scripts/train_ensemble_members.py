"""Train additional ensemble members for FNO-2D using the standardized pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_scaled_experiments import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train additional ensemble members.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/scaled_fno_supervised.yaml"),
        help="Path to experiment config file.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[45, 46],
        help="Random seeds to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs per run.",
    )
    args = parser.parse_args()

    for seed in args.seeds:
        run_name = f"{args.config.stem}_seed{seed}"
        ckpt_path = Path(f"artifacts/checkpoints/{run_name}/best_model.pt")
        if ckpt_path.is_file():
            logger.info(f"Checkpoint {ckpt_path} already exists. Skipping training.")
            continue

        logger.info(f">>> Training ensemble member: {run_name} (seed {seed})")
        res = run_experiment(
            config_path=args.config,
            seed=seed,
            dataset_path=Path("artifacts/data/scaled_dataset.h5"),
            epochs=args.epochs,
        )
        logger.info(
            f"Run {run_name} completed in {res['training_summary']['training_time_seconds']:.1f}s. "
            f"Best Val Loss: {res['training_summary']['best_val_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
