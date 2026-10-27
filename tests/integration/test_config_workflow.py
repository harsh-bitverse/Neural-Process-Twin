"""Integration tests verifying configuration workflows against repository config files."""

from __future__ import annotations

from pathlib import Path

import pytest

from neural_process_twin.utils.config import (
    DataConfig,
    TrainingConfig,
    deep_merge,
    load_data_config,
    load_training_config,
    load_yaml_file,
    parse_training_config,
)


@pytest.fixture
def repo_root() -> Path:
    """Return absolute Path to the repository root."""
    return Path(__file__).resolve().parent.parent.parent


@pytest.mark.integration
def test_repository_default_data_config(repo_root: Path) -> None:
    """Verify that the repository configs/data.yaml loads and validates successfully."""
    config_path = repo_root / "configs" / "data.yaml"
    assert config_path.is_file(), f"Expected configs/data.yaml at {config_path}"

    cfg = load_data_config(config_path)
    assert isinstance(cfg, DataConfig)
    assert cfg.config_status == "provisional"
    assert cfg.grid.nx == 64
    assert cfg.grid.ny == 64
    assert cfg.physics.conductivity.k_min > 0.0
    assert cfg.physics.conductivity.k_max >= cfg.physics.conductivity.k_min
    assert cfg.boundary_conditions.left.type in {"dirichlet", "neumann"}
    assert cfg.dataset.num_train >= 1


@pytest.mark.integration
def test_repository_default_training_config(repo_root: Path) -> None:
    """Verify that the repository configs/training.yaml loads and validates successfully."""
    config_path = repo_root / "configs" / "training.yaml"
    assert config_path.is_file(), f"Expected configs/training.yaml at {config_path}"

    cfg = load_training_config(config_path)
    assert isinstance(cfg, TrainingConfig)
    assert cfg.config_status == "provisional"
    assert cfg.model.architecture in {"fno2d", "unet2d"}
    assert cfg.model.modes_x > 0
    assert cfg.optimization.learning_rate > 0.0
    assert cfg.training.batch_size >= 1
    assert cfg.training.epochs >= 1


@pytest.mark.integration
def test_experiment_config_override_workflow(repo_root: Path) -> None:
    """Verify that experiment overrides in configs/experiments/ merge cleanly with training.yaml."""
    base_path = repo_root / "configs" / "training.yaml"
    experiment_path = repo_root / "configs" / "experiments" / "baseline_heat2d.yaml"
    assert experiment_path.is_file(), f"Expected experiment config at {experiment_path}"

    base_dict = load_yaml_file(base_path)
    exp_dict = load_yaml_file(experiment_path)

    merged = deep_merge(base_dict, exp_dict)
    cfg = parse_training_config(merged)

    assert isinstance(cfg, TrainingConfig)
    assert cfg.model.architecture == "fno2d"
    assert cfg.model.modes_x == 12
    assert cfg.training.batch_size == 32
