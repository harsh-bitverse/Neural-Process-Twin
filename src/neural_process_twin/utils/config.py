"""Configuration loading and validation utilities for Neural Process Twin.

This module provides strictly typed and validated configuration structures for
both physical data generation and surrogate model training.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(Exception):
    """Base exception for all configuration loading and validation errors."""


class ConfigurationMissingFieldError(ConfigurationError):
    """Raised when a mandatory configuration field or section is missing."""


class ConfigurationTypeError(ConfigurationError):
    """Raised when a configuration field has an incompatible type."""


class ConfigurationValueError(ConfigurationError):
    """Raised when a configuration value is out of bounds or semantically invalid."""


@dataclass(frozen=True)
class DomainConfig:
    """Spatial 2D rectangular domain [x_min, x_max] x [y_min, y_max]."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    length_unit: str = "meter"

    def __post_init__(self) -> None:
        if self.x_min >= self.x_max:
            raise ConfigurationValueError(
                f"domain.x_min ({self.x_min}) must be strictly less than domain.x_max ({self.x_max})."
            )
        if self.y_min >= self.y_max:
            raise ConfigurationValueError(
                f"domain.y_min ({self.y_min}) must be strictly less than domain.y_max ({self.y_max})."
            )


@dataclass(frozen=True)
class GridConfig:
    """Discretization grid dimensions for solvers and field representations.

    Attributes:
        nx: Number of control volumes (cells) along x. Must be >= 2.
        ny: Number of control volumes (cells) along y. Must be >= 2.
        coordinate_system: Coordinate system identifier, default 'cartesian_2d'.
        grid_type: Grid centering convention ('cell_centered' or 'node_centered').
            The finite-volume reference solver uses 'cell_centered'.
    """

    nx: int
    ny: int
    coordinate_system: str = "cartesian_2d"
    grid_type: str = "cell_centered"

    def __post_init__(self) -> None:
        if self.nx < 2:
            raise ConfigurationValueError(
                f"grid.nx ({self.nx}) must be >= 2 for spatial discretization."
            )
        if self.ny < 2:
            raise ConfigurationValueError(
                f"grid.ny ({self.ny}) must be >= 2 for spatial discretization."
            )
        allowed_grid_types = {"cell_centered", "node_centered"}
        if self.grid_type not in allowed_grid_types:
            raise ConfigurationValueError(
                f"grid.grid_type '{self.grid_type}' is not supported. Must be one of {allowed_grid_types}."
            )


@dataclass(frozen=True)
class ConductivityConfig:
    """Thermal conductivity k(x,y) distribution specifications."""

    distribution: str
    k_min: float
    k_max: float
    length_scale: float | None = None

    def __post_init__(self) -> None:
        if self.k_min <= 0.0:
            raise ConfigurationValueError(
                f"k_min ({self.k_min}) must be strictly positive to ensure ellipticity."
            )
        if self.k_min > self.k_max:
            raise ConfigurationValueError(f"k_min ({self.k_min}) must be <= k_max ({self.k_max}).")


@dataclass(frozen=True)
class SourceTermConfig:
    """Volumetric heat generation q(x,y) distribution specifications."""

    distribution: str
    q_min: float
    q_max: float
    num_peaks: int | None = None

    def __post_init__(self) -> None:
        if self.q_min > self.q_max:
            raise ConfigurationValueError(f"q_min ({self.q_min}) must be <= q_max ({self.q_max}).")


@dataclass(frozen=True)
class PhysicsDataConfig:
    """Governing physics and parameter field configuration."""

    problem_type: str
    governing_equation: str
    conductivity: ConductivityConfig
    source_term: SourceTermConfig


@dataclass(frozen=True)
class BoundarySideConfig:
    """Boundary condition specification on a single domain boundary side."""

    type: str
    value: float

    def __post_init__(self) -> None:
        allowed_types = {"dirichlet", "neumann"}
        if self.type not in allowed_types:
            raise ConfigurationValueError(
                f"Boundary condition type '{self.type}' is not supported. Must be one of {allowed_types}."
            )


@dataclass(frozen=True)
class BoundaryConditionsConfig:
    """Boundary conditions for all 4 boundaries of the 2D rectangular domain."""

    left: BoundarySideConfig
    right: BoundarySideConfig
    bottom: BoundarySideConfig
    top: BoundarySideConfig


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset split sizes, random seed, and archival format."""

    seed: int
    num_train: int
    num_val: int
    num_test_iid: int
    num_test_ood: int
    storage_format: str = "hdf5"

    def __post_init__(self) -> None:
        if self.num_train < 1:
            raise ConfigurationValueError(f"num_train ({self.num_train}) must be at least 1.")
        if self.num_val < 0 or self.num_test_iid < 0 or self.num_test_ood < 0:
            raise ConfigurationValueError("Dataset split sample counts cannot be negative.")


@dataclass(frozen=True)
class DataConfig:
    """Full validated configuration for data synthesis and ingestion."""

    version: str
    config_status: str
    domain: DomainConfig
    grid: GridConfig
    physics: PhysicsDataConfig
    boundary_conditions: BoundaryConditionsConfig
    dataset: DatasetConfig


# ==============================================================================
# Training Configuration Data Structures
# ==============================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Surrogate model architecture hyperparameters."""

    architecture: str
    in_channels: int
    out_channels: int
    width: int
    depth: int
    modes_x: int = 12
    modes_y: int = 12

    def __post_init__(self) -> None:
        allowed = {"fno2d", "unet2d", "cnn2d", "cnn"}
        if self.architecture not in allowed:
            raise ConfigurationValueError(
                f"Model architecture '{self.architecture}' not recognized. Must be one of {allowed}."
            )
        if self.in_channels < 1 or self.out_channels < 1:
            raise ConfigurationValueError("Model in_channels and out_channels must be >= 1.")
        if self.architecture == "fno2d" and (self.modes_x < 1 or self.modes_y < 1):
            raise ConfigurationValueError("modes_x and modes_y must be >= 1 for fno2d.")
        if self.width < 1 or self.depth < 1:
            raise ConfigurationValueError("Model width and depth must be >= 1.")


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning rate scheduler settings."""

    type: str
    warmup_epochs: int = 0
    min_lr: float = 1e-6

    def __post_init__(self) -> None:
        if self.warmup_epochs < 0:
            raise ConfigurationValueError("warmup_epochs cannot be negative.")
        if self.min_lr < 0:
            raise ConfigurationValueError("min_lr cannot be negative.")


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer algorithm and learning rate settings."""

    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler: SchedulerConfig | None = None

    def __post_init__(self) -> None:
        allowed_optimizers = {"adam", "adamw"}
        if self.optimizer.lower() not in allowed_optimizers:
            raise ConfigurationValueError(
                f"Optimizer '{self.optimizer}' not supported. Allowed: {allowed_optimizers}."
            )
        if self.learning_rate <= 0.0:
            raise ConfigurationValueError(
                f"learning_rate ({self.learning_rate}) must be strictly positive."
            )
        if self.weight_decay < 0.0:
            raise ConfigurationValueError("weight_decay cannot be negative.")


@dataclass(frozen=True)
class DataLossConfig:
    """Supervised data misfit loss configuration."""

    type: str
    weight: float

    def __post_init__(self) -> None:
        allowed = {"relative_l2", "mse", "l1"}
        if self.type not in allowed:
            raise ConfigurationValueError(
                f"data_loss.type '{self.type}' not supported. Allowed: {allowed}."
            )
        if self.weight < 0.0:
            raise ConfigurationValueError("data_loss weight cannot be negative.")


@dataclass(frozen=True)
class PhysicsLossConfig:
    """Physics-informed PDE residual loss configuration."""

    enabled: bool
    weight: float
    residual_type: str = "interior_pde_l2"
    bc_weight: float = 0.0
    q_floor: float = 1.0
    norm_type: str = "l2"

    def __post_init__(self) -> None:
        if self.weight < 0.0:
            raise ConfigurationValueError("physics_loss weight cannot be negative.")
        if self.bc_weight < 0.0:
            raise ConfigurationValueError("physics_loss bc_weight cannot be negative.")
        if self.q_floor <= 0.0:
            raise ConfigurationValueError("physics_loss q_floor must be strictly positive.")


@dataclass(frozen=True)
class LossConfig:
    """Combined loss formulation."""

    data_loss: DataLossConfig
    physics_loss: PhysicsLossConfig


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early stopping criteria."""

    enabled: bool
    patience: int
    metric: str

    def __post_init__(self) -> None:
        if self.enabled and self.patience < 1:
            raise ConfigurationValueError("early_stopping patience must be >= 1 when enabled.")


@dataclass(frozen=True)
class TrainingRunConfig:
    """Execution parameters for model training."""

    seed: int
    batch_size: int
    epochs: int
    gradient_clip_norm: float | None
    checkpoint_frequency: int
    early_stopping: EarlyStoppingConfig

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ConfigurationValueError(f"batch_size ({self.batch_size}) must be >= 1.")
        if self.epochs < 1:
            raise ConfigurationValueError(f"epochs ({self.epochs}) must be >= 1.")
        if self.checkpoint_frequency < 1:
            raise ConfigurationValueError(
                f"checkpoint_frequency ({self.checkpoint_frequency}) must be >= 1."
            )


@dataclass(frozen=True)
class HardwareConfig:
    """Device and worker hardware allocation."""

    device: str
    num_workers: int
    mixed_precision: bool

    def __post_init__(self) -> None:
        allowed = {"auto", "cpu", "cuda"}
        if self.device not in allowed:
            raise ConfigurationValueError(
                f"device '{self.device}' not supported. Allowed: {allowed}."
            )
        if self.num_workers < 0:
            raise ConfigurationValueError("num_workers cannot be negative.")


@dataclass(frozen=True)
class TrainingConfig:
    """Full validated configuration for surrogate model training."""

    version: str
    config_status: str
    model: ModelConfig
    optimization: OptimizationConfig
    loss: LossConfig
    training: TrainingRunConfig
    hardware: HardwareConfig


# ==============================================================================
# Parsing and Loading Helpers
# ==============================================================================


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML file from disk into a Python dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed dictionary.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ConfigurationError: If the file is not a valid YAML document or does not parse as a dict.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML file '{file_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Configuration file '{file_path}' must contain a key-value mapping (dictionary)."
        )

    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries, returning a new dictionary.

    Keys in `override` take precedence over keys in `base`. Nested dictionaries
    are recursively merged.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _require_field(d: dict[str, Any], key: str, context: str) -> Any:
    """Retrieve mandatory field from dictionary or raise ConfigurationMissingFieldError."""
    if key not in d:
        raise ConfigurationMissingFieldError(
            f"Missing mandatory configuration field '{context}.{key}'."
        )
    return d[key]


def _require_dict(d: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    """Retrieve mandatory dictionary section from dictionary."""
    val = _require_field(d, key, context)
    if not isinstance(val, dict):
        raise ConfigurationTypeError(
            f"Expected section '{context}.{key}' to be a dictionary, got {type(val).__name__}."
        )
    return val


def parse_data_config(raw: dict[str, Any]) -> DataConfig:
    """Validate and parse a raw dictionary into a typed DataConfig.

    Args:
        raw: Dictionary parsed from data configuration YAML.

    Returns:
        Validated DataConfig instance.

    Raises:
        ConfigurationError: If validation fails.
    """
    version = str(_require_field(raw, "version", "root"))
    config_status = str(_require_field(raw, "config_status", "root"))

    dom_raw = _require_dict(raw, "domain", "root")
    domain = DomainConfig(
        x_min=float(_require_field(dom_raw, "x_min", "domain")),
        x_max=float(_require_field(dom_raw, "x_max", "domain")),
        y_min=float(_require_field(dom_raw, "y_min", "domain")),
        y_max=float(_require_field(dom_raw, "y_max", "domain")),
        length_unit=str(dom_raw.get("length_unit", "meter")),
    )

    grid_raw = _require_dict(raw, "grid", "root")
    grid = GridConfig(
        nx=int(_require_field(grid_raw, "nx", "grid")),
        ny=int(_require_field(grid_raw, "ny", "grid")),
        coordinate_system=str(grid_raw.get("coordinate_system", "cartesian_2d")),
        grid_type=str(grid_raw.get("grid_type", "cell_centered")),
    )

    phys_raw = _require_dict(raw, "physics", "root")
    cond_raw = _require_dict(phys_raw, "conductivity", "physics")
    conductivity = ConductivityConfig(
        distribution=str(_require_field(cond_raw, "distribution", "physics.conductivity")),
        k_min=float(_require_field(cond_raw, "k_min", "physics.conductivity")),
        k_max=float(_require_field(cond_raw, "k_max", "physics.conductivity")),
        length_scale=(
            float(cond_raw["length_scale"])
            if "length_scale" in cond_raw and cond_raw["length_scale"] is not None
            else None
        ),
    )

    src_raw = _require_dict(phys_raw, "source_term", "physics")
    source_term = SourceTermConfig(
        distribution=str(_require_field(src_raw, "distribution", "physics.source_term")),
        q_min=float(_require_field(src_raw, "q_min", "physics.source_term")),
        q_max=float(_require_field(src_raw, "q_max", "physics.source_term")),
        num_peaks=int(src_raw["num_peaks"])
        if "num_peaks" in src_raw and src_raw["num_peaks"] is not None
        else None,
    )

    physics = PhysicsDataConfig(
        problem_type=str(_require_field(phys_raw, "problem_type", "physics")),
        governing_equation=str(_require_field(phys_raw, "governing_equation", "physics")),
        conductivity=conductivity,
        source_term=source_term,
    )

    bc_raw = _require_dict(raw, "boundary_conditions", "root")
    bcs = {}
    for side in ("left", "right", "bottom", "top"):
        side_raw = _require_dict(bc_raw, side, "boundary_conditions")
        bcs[side] = BoundarySideConfig(
            type=str(_require_field(side_raw, "type", f"boundary_conditions.{side}")).lower(),
            value=float(_require_field(side_raw, "value", f"boundary_conditions.{side}")),
        )
    boundary_conditions = BoundaryConditionsConfig(
        left=bcs["left"],
        right=bcs["right"],
        bottom=bcs["bottom"],
        top=bcs["top"],
    )

    ds_raw = _require_dict(raw, "dataset", "root")
    dataset = DatasetConfig(
        seed=int(_require_field(ds_raw, "seed", "dataset")),
        num_train=int(_require_field(ds_raw, "num_train", "dataset")),
        num_val=int(_require_field(ds_raw, "num_val", "dataset")),
        num_test_iid=int(_require_field(ds_raw, "num_test_iid", "dataset")),
        num_test_ood=int(_require_field(ds_raw, "num_test_ood", "dataset")),
        storage_format=str(ds_raw.get("storage_format", "hdf5")),
    )

    return DataConfig(
        version=version,
        config_status=config_status,
        domain=domain,
        grid=grid,
        physics=physics,
        boundary_conditions=boundary_conditions,
        dataset=dataset,
    )


def parse_training_config(raw: dict[str, Any]) -> TrainingConfig:
    """Validate and parse a raw dictionary into a typed TrainingConfig.

    Args:
        raw: Dictionary parsed from training configuration YAML.

    Returns:
        Validated TrainingConfig instance.

    Raises:
        ConfigurationError: If validation fails.
    """
    version = str(_require_field(raw, "version", "root"))
    config_status = str(_require_field(raw, "config_status", "root"))

    model_raw = _require_dict(raw, "model", "root")
    arch = str(_require_field(model_raw, "architecture", "model")).lower()
    modes_x = int(model_raw.get("modes_x", 12))
    modes_y = int(model_raw.get("modes_y", 12))
    model = ModelConfig(
        architecture=arch,
        in_channels=int(_require_field(model_raw, "in_channels", "model")),
        out_channels=int(_require_field(model_raw, "out_channels", "model")),
        width=int(_require_field(model_raw, "width", "model")),
        depth=int(_require_field(model_raw, "depth", "model")),
        modes_x=modes_x,
        modes_y=modes_y,
    )

    opt_raw = _require_dict(raw, "optimization", "root")
    sched_raw = opt_raw.get("scheduler")
    scheduler: SchedulerConfig | None = None
    if isinstance(sched_raw, dict):
        scheduler = SchedulerConfig(
            type=str(_require_field(sched_raw, "type", "optimization.scheduler")),
            warmup_epochs=int(sched_raw.get("warmup_epochs", 0)),
            min_lr=float(sched_raw.get("min_lr", 1e-6)),
        )

    optimization = OptimizationConfig(
        optimizer=str(_require_field(opt_raw, "optimizer", "optimization")),
        learning_rate=float(_require_field(opt_raw, "learning_rate", "optimization")),
        weight_decay=float(_require_field(opt_raw, "weight_decay", "optimization")),
        scheduler=scheduler,
    )

    loss_raw = _require_dict(raw, "loss", "root")
    data_loss_raw = _require_dict(loss_raw, "data_loss", "loss")
    data_loss = DataLossConfig(
        type=str(_require_field(data_loss_raw, "type", "loss.data_loss")).lower(),
        weight=float(_require_field(data_loss_raw, "weight", "loss.data_loss")),
    )

    physics_loss_raw = _require_dict(loss_raw, "physics_loss", "loss")
    physics_loss = PhysicsLossConfig(
        enabled=bool(_require_field(physics_loss_raw, "enabled", "loss.physics_loss")),
        weight=float(_require_field(physics_loss_raw, "weight", "loss.physics_loss")),
        residual_type=str(physics_loss_raw.get("residual_type", "interior_pde_l2")),
        bc_weight=float(physics_loss_raw.get("bc_weight", 0.0)),
        q_floor=float(physics_loss_raw.get("q_floor", 1.0)),
        norm_type=str(physics_loss_raw.get("norm_type", "l2")),
    )
    loss = LossConfig(data_loss=data_loss, physics_loss=physics_loss)

    train_raw = _require_dict(raw, "training", "root")
    es_raw = _require_dict(train_raw, "early_stopping", "training")
    early_stopping = EarlyStoppingConfig(
        enabled=bool(_require_field(es_raw, "enabled", "training.early_stopping")),
        patience=int(_require_field(es_raw, "patience", "training.early_stopping")),
        metric=str(_require_field(es_raw, "metric", "training.early_stopping")),
    )

    training = TrainingRunConfig(
        seed=int(_require_field(train_raw, "seed", "training")),
        batch_size=int(_require_field(train_raw, "batch_size", "training")),
        epochs=int(_require_field(train_raw, "epochs", "training")),
        gradient_clip_norm=(
            float(train_raw["gradient_clip_norm"])
            if "gradient_clip_norm" in train_raw and train_raw["gradient_clip_norm"] is not None
            else None
        ),
        checkpoint_frequency=int(_require_field(train_raw, "checkpoint_frequency", "training")),
        early_stopping=early_stopping,
    )

    hw_raw = _require_dict(raw, "hardware", "root")
    hardware = HardwareConfig(
        device=str(_require_field(hw_raw, "device", "hardware")).lower(),
        num_workers=int(_require_field(hw_raw, "num_workers", "hardware")),
        mixed_precision=bool(_require_field(hw_raw, "mixed_precision", "hardware")),
    )

    return TrainingConfig(
        version=version,
        config_status=config_status,
        model=model,
        optimization=optimization,
        loss=loss,
        training=training,
        hardware=hardware,
    )


def load_data_config(path: str | Path) -> DataConfig:
    """Load, parse, and validate a DataConfig from a YAML file path.

    Args:
        path: Path to the data configuration YAML file.

    Returns:
        Validated DataConfig.
    """
    raw = load_yaml_file(path)
    return parse_data_config(raw)


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load, parse, and validate a TrainingConfig from a YAML file path.

    Args:
        path: Path to the training configuration YAML file.

    Returns:
        Validated TrainingConfig.
    """
    raw = load_yaml_file(path)
    return parse_training_config(raw)
