"""Unit tests for configuration parsing, schema validation, and error handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neural_process_twin.utils.config import (
    BoundaryConditionsConfig,
    ConductivityConfig,
    ConfigurationError,
    ConfigurationMissingFieldError,
    ConfigurationTypeError,
    ConfigurationValueError,
    DataConfig,
    DomainConfig,
    GridConfig,
    ModelConfig,
    OptimizationConfig,
    SourceTermConfig,
    TrainingConfig,
    deep_merge,
    load_yaml_file,
    parse_data_config,
    parse_training_config,
)


@pytest.fixture
def minimal_data_dict() -> dict[str, Any]:
    """Return a valid minimal raw dictionary for data configuration."""
    return {
        "version": "0.1.0",
        "config_status": "provisional",
        "domain": {
            "x_min": 0.0,
            "x_max": 1.0,
            "y_min": 0.0,
            "y_max": 1.0,
            "length_unit": "meter",
        },
        "grid": {
            "nx": 64,
            "ny": 64,
            "coordinate_system": "cartesian_2d",
        },
        "physics": {
            "problem_type": "steady_state_heat_conduction",
            "governing_equation": "-div(k*grad(T)) = q",
            "conductivity": {
                "distribution": "gaussian_random_field",
                "k_min": 0.1,
                "k_max": 10.0,
                "length_scale": 0.2,
            },
            "source_term": {
                "distribution": "gaussian_blobs",
                "q_min": 0.0,
                "q_max": 1000.0,
                "num_peaks": 3,
            },
        },
        "boundary_conditions": {
            "left": {"type": "dirichlet", "value": 300.0},
            "right": {"type": "dirichlet", "value": 350.0},
            "bottom": {"type": "neumann", "value": 0.0},
            "top": {"type": "neumann", "value": 0.0},
        },
        "dataset": {
            "seed": 42,
            "num_train": 1000,
            "num_val": 200,
            "num_test_iid": 200,
            "num_test_ood": 100,
            "storage_format": "hdf5",
        },
    }


@pytest.fixture
def minimal_training_dict() -> dict[str, Any]:
    """Return a valid minimal raw dictionary for training configuration."""
    return {
        "version": "0.1.0",
        "config_status": "provisional",
        "model": {
            "architecture": "fno2d",
            "in_channels": 2,
            "out_channels": 1,
            "modes_x": 12,
            "modes_y": 12,
            "width": 32,
            "depth": 4,
        },
        "optimization": {
            "optimizer": "adamw",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": {
                "type": "cosine_annealing",
                "warmup_epochs": 5,
                "min_lr": 1.0e-6,
            },
        },
        "loss": {
            "data_loss": {
                "type": "relative_l2",
                "weight": 1.0,
            },
            "physics_loss": {
                "enabled": False,
                "weight": 0.01,
                "residual_type": "interior_pde_l2",
            },
        },
        "training": {
            "seed": 42,
            "batch_size": 32,
            "epochs": 100,
            "gradient_clip_norm": 1.0,
            "checkpoint_frequency": 10,
            "early_stopping": {
                "enabled": True,
                "patience": 15,
                "metric": "val_relative_l2",
            },
        },
        "hardware": {
            "device": "auto",
            "num_workers": 2,
            "mixed_precision": False,
        },
    }


@pytest.mark.unit
def test_parse_valid_data_config(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that a valid dictionary parses into a populated DataConfig."""
    cfg = parse_data_config(minimal_data_dict)
    assert isinstance(cfg, DataConfig)
    assert isinstance(cfg.domain, DomainConfig)
    assert cfg.domain.x_max == 1.0
    assert isinstance(cfg.grid, GridConfig)
    assert cfg.grid.nx == 64
    assert isinstance(cfg.physics.conductivity, ConductivityConfig)
    assert cfg.physics.conductivity.k_min == 0.1
    assert isinstance(cfg.physics.source_term, SourceTermConfig)
    assert isinstance(cfg.boundary_conditions, BoundaryConditionsConfig)
    assert cfg.boundary_conditions.left.type == "dirichlet"
    assert cfg.boundary_conditions.bottom.type == "neumann"
    assert cfg.dataset.num_train == 1000


@pytest.mark.unit
def test_parse_valid_training_config(minimal_training_dict: dict[str, Any]) -> None:
    """Verify that a valid dictionary parses into a populated TrainingConfig."""
    cfg = parse_training_config(minimal_training_dict)
    assert isinstance(cfg, TrainingConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert cfg.model.architecture == "fno2d"
    assert isinstance(cfg.optimization, OptimizationConfig)
    assert cfg.optimization.learning_rate == 0.001
    assert cfg.loss.data_loss.type == "relative_l2"
    assert cfg.training.batch_size == 32
    assert cfg.hardware.device == "auto"


@pytest.mark.unit
def test_load_yaml_file_not_found(tmp_path: Path) -> None:
    """Verify that loading a non-existent YAML file raises FileNotFoundError."""
    non_existent = tmp_path / "does_not_exist.yaml"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_yaml_file(non_existent)


@pytest.mark.unit
def test_load_yaml_file_malformed(tmp_path: Path) -> None:
    """Verify that malformed YAML syntax raises ConfigurationError."""
    bad_yaml = tmp_path / "malformed.yaml"
    bad_yaml.write_text("domain: [unclosed list\n  key: val", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
        load_yaml_file(bad_yaml)


@pytest.mark.unit
def test_load_yaml_file_non_dict(tmp_path: Path) -> None:
    """Verify that a YAML file containing a list or scalar raises ConfigurationError."""
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="must contain a key-value mapping"):
        load_yaml_file(list_yaml)


@pytest.mark.unit
def test_missing_mandatory_field(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that omitting a mandatory field raises ConfigurationMissingFieldError."""
    del minimal_data_dict["grid"]["nx"]
    with pytest.raises(
        ConfigurationMissingFieldError, match=r"Missing mandatory configuration field 'grid\.nx'"
    ):
        parse_data_config(minimal_data_dict)


@pytest.mark.unit
def test_wrong_section_type(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that supplying a non-dict for a section raises ConfigurationTypeError."""
    minimal_data_dict["domain"] = "not_a_dictionary"
    with pytest.raises(
        ConfigurationTypeError, match=r"Expected section 'root\.domain' to be a dictionary"
    ):
        parse_data_config(minimal_data_dict)


@pytest.mark.unit
def test_domain_bounds_validation(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that invalid domain coordinates (x_min >= x_max) raise ConfigurationValueError."""
    minimal_data_dict["domain"]["x_min"] = 2.0
    minimal_data_dict["domain"]["x_max"] = 1.0
    with pytest.raises(
        ConfigurationValueError, match=r"domain\.x_min .* must be strictly less than"
    ):
        parse_data_config(minimal_data_dict)


@pytest.mark.unit
def test_grid_resolution_validation(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that degenerate grid resolutions (< 2) raise ConfigurationValueError."""
    minimal_data_dict["grid"]["nx"] = 1
    with pytest.raises(ConfigurationValueError, match=r"grid\.nx .* must be >= 2"):
        parse_data_config(minimal_data_dict)


@pytest.mark.unit
def test_grid_type_validation(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that unsupported grid_type raises ConfigurationValueError."""
    minimal_data_dict["grid"]["grid_type"] = "unsupported_staggered"
    with pytest.raises(ConfigurationValueError, match=r"grid\.grid_type .* is not supported"):
        parse_data_config(minimal_data_dict)


@pytest.mark.unit
def test_conductivity_positivity_validation(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that non-positive k_min violates ellipticity and raises ConfigurationValueError."""
    minimal_data_dict["physics"]["conductivity"]["k_min"] = 0.0
    with pytest.raises(
        ConfigurationValueError, match=r"must be strictly positive to ensure ellipticity"
    ):
        parse_data_config(minimal_data_dict)


@pytest.mark.unit
def test_conductivity_ordering_validation(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that k_min > k_max raises ConfigurationValueError."""
    minimal_data_dict["physics"]["conductivity"]["k_min"] = 15.0
    minimal_data_dict["physics"]["conductivity"]["k_max"] = 5.0
    with pytest.raises(ConfigurationValueError, match=r"k_min .* must be <= k_max"):
        parse_data_config(minimal_data_dict)


@pytest.mark.unit
def test_unsupported_boundary_condition_type(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that an unsupported boundary condition type raises ConfigurationValueError."""
    minimal_data_dict["boundary_conditions"]["left"]["type"] = "robin_unsupported"
    with pytest.raises(
        ConfigurationValueError,
        match=r"Boundary condition type 'robin_unsupported' is not supported",
    ):
        parse_data_config(minimal_data_dict)


@pytest.mark.unit
def test_empty_training_dataset_validation(minimal_data_dict: dict[str, Any]) -> None:
    """Verify that num_train < 1 raises ConfigurationValueError."""
    minimal_data_dict["dataset"]["num_train"] = 0
    with pytest.raises(ConfigurationValueError, match=r"num_train .* must be at least 1"):
        parse_data_config(minimal_data_dict)


@pytest.mark.unit
def test_unrecognized_model_architecture(minimal_training_dict: dict[str, Any]) -> None:
    """Verify that an unrecognized model architecture raises ConfigurationValueError."""
    minimal_training_dict["model"]["architecture"] = "unsupported_transformer"
    with pytest.raises(
        ConfigurationValueError,
        match=r"Model architecture 'unsupported_transformer' not recognized",
    ):
        parse_training_config(minimal_training_dict)


@pytest.mark.unit
def test_invalid_learning_rate(minimal_training_dict: dict[str, Any]) -> None:
    """Verify that non-positive learning rate raises ConfigurationValueError."""
    minimal_training_dict["optimization"]["learning_rate"] = -0.01
    with pytest.raises(
        ConfigurationValueError, match=r"learning_rate .* must be strictly positive"
    ):
        parse_training_config(minimal_training_dict)


@pytest.mark.unit
def test_unsupported_data_loss_type(minimal_training_dict: dict[str, Any]) -> None:
    """Verify that an unsupported loss type raises ConfigurationValueError."""
    minimal_training_dict["loss"]["data_loss"]["type"] = "cross_entropy"
    with pytest.raises(
        ConfigurationValueError, match=r"data_loss\.type 'cross_entropy' not supported"
    ):
        parse_training_config(minimal_training_dict)


@pytest.mark.unit
def test_deep_merge_utility() -> None:
    """Verify that deep_merge recursively overlays override dictionaries while keeping base keys."""
    base = {
        "model": {"arch": "fno", "modes": 12, "width": 32},
        "training": {"lr": 0.001, "epochs": 50},
    }
    override = {
        "model": {"modes": 16},
        "training": {"epochs": 100},
        "new_section": {"active": True},
    }
    merged = deep_merge(base, override)

    # Overridden values
    assert merged["model"]["modes"] == 16
    assert merged["training"]["epochs"] == 100
    # Retained base values
    assert merged["model"]["arch"] == "fno"
    assert merged["model"]["width"] == 32
    assert merged["training"]["lr"] == 0.001
    # Added new section
    assert merged["new_section"]["active"] is True
    # Ensure immutability of base
    assert base["model"]["modes"] == 12
