"""Convolutional Neural Network (CNN) 2D baseline surrogate for physical field prediction.

Provides a compact, highly interpretable fully-convolutional baseline mapping
input parameter and boundary fields (batch, in_channels, ny, nx) to predicted
temperature fields (batch, 1, ny, nx).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_process_twin.models.base import SurrogateModel


class ConvBlock2D(nn.Module):
    """Convolutional block with configurable activation, normalization, and residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        activation: str = "gelu",
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )

        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

        # Stable Kaiming normal initialization
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv(x))
        if self.use_residual:
            return x + out
        return out


class CNNBaseline2D(SurrogateModel):
    """Convolutional baseline architecture for 2D physical field surrogates.

    Architecture:
    1. Input projection: Conv2D(in_channels -> width, kernel_size=3) + Activation.
    2. Hidden backbone: `depth` ConvBlock2D layers with residual connections.
    3. Output projection: Conv2D(width -> out_channels=1, kernel_size=1).

    Note on Boundary Effects and Padding Behavior:
    In physical field simulation, standard zero-padding treats the space outside the
    physical domain boundaries as a fictitious zero-valued reservoir, which can induce
    artificial temperature dips along the domain perimeter. To mitigate boundary artifacts,
    the default padding mode is set to 'replicate', which continuously projects border
    values outward across the stencil receptive field.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        width: int = 32,
        depth: int = 4,
        kernel_size: int = 3,
        padding_mode: str = "replicate",
        activation: str = "gelu",
        use_residual: bool = True,
    ) -> None:
        """Initialize CNN baseline model.

        Args:
            in_channels: Number of input channels (e.g. 12 for canonical k, q, coords, BCs).
            out_channels: Number of output channels (1 for temperature field).
            width: Number of hidden feature channels across convolutional blocks.
            depth: Number of convolutional layers in the hidden backbone. Must be >= 1.
            kernel_size: Convolution kernel size (default 3x3).
            padding_mode: Padding mode ('replicate' or 'zeros').
            activation: Nonlinear activation function ('gelu', 'relu', 'silu').
            use_residual: If True, adds residual skip connections across same-width layers.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}.")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")

        self.width = width
        self.depth = depth
        self.padding_mode = padding_mode

        # 1. Input projection
        padding = kernel_size // 2
        self.in_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode=padding_mode,
        )
        nn.init.kaiming_normal_(self.in_proj.weight, nonlinearity="relu")
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        act = activation.lower()
        if act == "gelu":
            self.in_act = nn.GELU()
        elif act == "relu":
            self.in_act = nn.ReLU(inplace=True)
        elif act == "silu":
            self.in_act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

        # 2. Hidden convolutional backbone
        blocks = []
        for _ in range(depth - 1):
            blocks.append(
                ConvBlock2D(
                    in_channels=width,
                    out_channels=width,
                    kernel_size=kernel_size,
                    padding_mode=padding_mode,
                    activation=activation,
                    use_residual=use_residual,
                )
            )
        self.backbone = nn.Sequential(*blocks)

        # 3. Final projection to temperature field
        self.out_proj = nn.Conv2d(
            in_channels=width,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
        )
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, ny, nx).

        Returns:
            Output tensor of shape (batch, 1, ny, nx).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (batch, C, ny, nx), got {x.dim()}D.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.in_channels} channels, got {x.shape[1]}."
            )

        h = self.in_act(self.in_proj(x))
        h = self.backbone(h)
        out = self.out_proj(h)
        return out
