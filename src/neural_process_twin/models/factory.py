"""Model factory for instantiating surrogate architectures from typed configurations.

Supports standard model types:
- 'fno2d': Fourier Neural Operator 2D for continuous operator learning
- 'cnn2d' / 'cnn' / 'unet2d': Convolutional neural network baseline
"""

from __future__ import annotations

from typing import Any

from neural_process_twin.models.base import SurrogateModel
from neural_process_twin.models.boundary_aware_fno import BoundaryAwareFNO2D
from neural_process_twin.models.cnn import CNNBaseline2D
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.utils.config import ModelConfig


def build_model(config: ModelConfig | dict[str, Any]) -> SurrogateModel:
    """Instantiate a SurrogateModel from a ModelConfig or dictionary.

    Args:
        config: ModelConfig dataclass or dictionary specifying model architecture
            and hyperparameters.

    Returns:
        Instantiated SurrogateModel.

    Raises:
        ValueError: If the requested architecture is unknown.
    """
    if isinstance(config, ModelConfig):
        arch = config.architecture.lower()
        in_channels = config.in_channels
        out_channels = config.out_channels
        width = config.width
        depth = config.depth
        modes_x = config.modes_x
        modes_y = config.modes_y
        extra: dict[str, Any] = {}
    elif isinstance(config, dict):
        arch = str(config.get("architecture", "")).lower()
        in_channels = int(config.get("in_channels", 12))
        out_channels = int(config.get("out_channels", 1))
        width = int(config.get("width", 32))
        depth = int(config.get("depth", 4))
        modes_x = int(config.get("modes_x", 12))
        modes_y = int(config.get("modes_y", 12))
        extra = {
            k: v
            for k, v in config.items()
            if k
            not in {
                "architecture",
                "in_channels",
                "out_channels",
                "width",
                "depth",
                "modes_x",
                "modes_y",
            }
        }
    else:
        raise TypeError(f"Expected ModelConfig or dict, got {type(config)}.")

    if arch in {"fno2d", "fno"}:
        activation = extra.get("activation", "gelu")
        return FourierNeuralOperator2D(
            in_channels=in_channels,
            out_channels=out_channels,
            modes_x=modes_x,
            modes_y=modes_y,
            width=width,
            depth=depth,
            activation=activation,
        )
    elif arch in {"boundary_aware_fno", "ba_fno2d", "ba_fno", "boundary_fno"}:
        activation = extra.get("activation", "gelu")
        use_boundary_bypass = bool(extra.get("use_boundary_bypass", True))
        correction_mode = str(extra.get("correction_mode", "hard"))
        learnable_gate = bool(extra.get("learnable_gate", False))
        flux_scale = float(extra.get("flux_scale", 1000.0))
        grid = extra.get("grid")
        return BoundaryAwareFNO2D(
            grid=grid,
            in_channels=in_channels,
            out_channels=out_channels,
            modes_x=modes_x,
            modes_y=modes_y,
            width=width,
            depth=depth,
            activation=activation,
            use_boundary_bypass=use_boundary_bypass,
            correction_mode=correction_mode,
            learnable_gate=learnable_gate,
            flux_scale=flux_scale,
        )
    elif arch in {"cnn2d", "cnn", "unet2d"}:
        activation = extra.get("activation", "gelu")
        padding_mode = extra.get("padding_mode", "replicate")
        kernel_size = int(extra.get("kernel_size", 3))
        use_residual = bool(extra.get("use_residual", True))
        return CNNBaseline2D(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            depth=depth,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            activation=activation,
            use_residual=use_residual,
        )
    else:
        raise ValueError(
            f"Unknown model architecture '{arch}'. Supported architectures: 'fno2d', 'boundary_aware_fno', 'cnn2d', 'cnn'."
        )
