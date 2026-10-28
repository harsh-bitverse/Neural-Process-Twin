"""Refined Multi-Resolution Fourier Neural Operator with Local High-Frequency Refinement.

Addresses localized maximum absolute error (MaxAE) spikes and boundary layer
distortions in multi-resolution FNO predictions by coupling a global frequency-domain
spectral operator with a localized spatial refinement residual network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_process_twin.models.base import SurrogateModel
from neural_process_twin.models.fno2d import FourierNeuralOperator2D


class LocalRefinementBlock(nn.Module):
    """Lightweight convolutional residual block for local boundary and interface refinement.

    Operates directly on local 3x3 spatial stencils across arbitrary grid resolutions,
    learning high-frequency corrections that cannot be captured by truncated Fourier modes.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        out_channels: int = 1,
        num_layers: int = 3,
    ) -> None:
        """Initialize LocalRefinementBlock.

        Args:
            in_channels: Number of input channels (input features + FNO output).
            hidden_channels: Width of intermediate convolutional feature maps.
            out_channels: Number of output channels (1 for temperature correction).
            num_layers: Number of conv layers in the refinement branch.
        """
        super().__init__()
        layers: list[nn.Module] = []
        cur_ch = in_channels
        for _i in range(num_layers - 1):
            layers.append(
                nn.Conv2d(
                    cur_ch,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                    padding_mode="replicate",
                )
            )
            layers.append(nn.GELU())
            cur_ch = hidden_channels

        # Final projection layer to delta_T
        final_conv = nn.Conv2d(
            cur_ch,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="replicate",
        )
        # Initialize final conv with small weights to ensure smooth warm-start from FNO
        nn.init.normal_(final_conv.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(final_conv.bias)
        layers.append(final_conv)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing localized residual correction.

        Args:
            x: Concatenated input tensor of shape (B, C, H, W).

        Returns:
            Additive temperature correction of shape (B, 1, H, W).
        """
        return self.net(x)


class RefinedFNO2D(SurrogateModel):
    """Coupled Global-Local Multi-Resolution Fourier Neural Operator.

    Combines:
    1. Global spectral convolution (FNO-2D) for continuous domain-wide thermal transport.
    2. Local convolutional refinement (3x3 stencil) for boundary layers and material interfaces.

    Architecture:
        y_fno = FNO2D(x)
        delta_y = RefinementBlock(concat([x, y_fno]))
        y_out = y_fno + delta_y
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        modes_x: int = 12,
        modes_y: int = 12,
        width: int = 32,
        depth: int = 4,
        refinement_hidden: int = 32,
        refinement_layers: int = 3,
        activation: str = "gelu",
    ) -> None:
        """Initialize RefinedFNO2D.

        Args:
            in_channels: Total input physical channels (default: 12).
            out_channels: Output temperature channels (default: 1).
            modes_x: Truncated Fourier modes along x.
            modes_y: Truncated Fourier modes along y.
            width: FNO spectral hidden channel width.
            depth: Number of Fourier neural operator layers.
            refinement_hidden: Convolutional channel width of refinement branch.
            refinement_layers: Number of conv layers in refinement branch.
            activation: Activation function name.
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.depth = depth

        # 1. Global FNO branch
        self.fno = FourierNeuralOperator2D(
            in_channels=in_channels,
            out_channels=out_channels,
            modes_x=modes_x,
            modes_y=modes_y,
            width=width,
            depth=depth,
            activation=activation,
        )

        # 2. Local refinement branch: accepts (x, y_fno) -> shape (B, in_channels + out_channels, H, W)
        self.local_refinement = LocalRefinementBlock(
            in_channels=in_channels + out_channels,
            hidden_channels=refinement_hidden,
            out_channels=out_channels,
            num_layers=refinement_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through coupled global-local neural operator.

        Args:
            x: Input tensor of shape (batch, in_channels, ny, nx).

        Returns:
            Refined output temperature field of shape (batch, out_channels, ny, nx).
        """
        # Global Fourier neural operator prediction
        y_fno = self.fno(x)

        # Local high-frequency residual correction
        feat_combined = torch.cat([x, y_fno], dim=1)
        delta_y = self.local_refinement(feat_combined)

        return y_fno + delta_y

    def forward_decomposed(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning decomposed global, local, and total predictions.

        Returns:
            Tuple of (y_total, y_fno, delta_local).
        """
        y_fno = self.fno(x)
        feat_combined = torch.cat([x, y_fno], dim=1)
        delta_y = self.local_refinement(feat_combined)
        y_total = y_fno + delta_y
        return y_total, y_fno, delta_y

    def get_num_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
