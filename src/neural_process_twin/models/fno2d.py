"""Fourier Neural Operator (FNO-2D) for continuous physical field surrogate modeling.

Implements resolution-invariant operator learning via parameterization in frequency
space using 2D real Fourier transforms (rfft2 / irfft2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neural_process_twin.models.base import SurrogateModel


class SpectralConv2D(nn.Module):
    """2D Spectral Convolution Layer for real-valued spatial fields.

    Computes:
        out = irfft2(R * rfft2(x))
    where R represents learned complex-valued frequency weights for retained modes.

    Handles real FFT conjugate symmetry by truncating non-negative x-frequencies
    and positive/negative y-frequencies across odd and even spatial resolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_x: int,
        modes_y: int,
    ) -> None:
        """Initialize spectral convolution layer.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            modes_x: Number of retained Fourier modes along the x-axis (0 <= modes_x <= nx // 2 + 1).
            modes_y: Number of retained Fourier modes along the y-axis (0 <= modes_y <= ny // 2).
        """
        super().__init__()
        if modes_x < 1 or modes_y < 1:
            raise ValueError(f"modes_x ({modes_x}) and modes_y ({modes_y}) must be >= 1.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        # Scaling factor for weight initialization
        scale = 1.0 / (in_channels * out_channels)

        # Complex weights for positive y-frequencies (top-left in Fourier spectrum)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat)
        )
        # Complex weights for negative y-frequencies (bottom-left in Fourier spectrum)
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes_y, modes_x, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward spectral convolution pass.

        Args:
            x: Input tensor of shape (batch, in_channels, ny, nx).

        Returns:
            Output tensor of shape (batch, out_channels, ny, nx).
        """
        batch_size, _, ny, nx = x.shape

        # Validate mode counts against current grid resolution
        max_modes_x = nx // 2 + 1
        max_modes_y = ny // 2
        if self.modes_x > max_modes_x:
            raise ValueError(
                f"Requested modes_x={self.modes_x} exceeds Nyquist limit {max_modes_x} for nx={nx}."
            )
        if self.modes_y > max_modes_y:
            raise ValueError(
                f"Requested modes_y={self.modes_y} exceeds Nyquist limit {max_modes_y} for ny={ny}."
            )

        # 1. 2D Real FFT: (batch, in_channels, ny, nx // 2 + 1) complex64
        x_ft = torch.fft.rfft2(x, norm="ortho")

        # 2. Allocate output frequency tensor (zero-padded for truncated modes)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            ny,
            nx // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # 3. Complex multiplication on retained Fourier modes:
        # Top-left quadrant (+ky, +kx)
        out_ft[:, :, : self.modes_y, : self.modes_x] = torch.einsum(
            "biyx,ioyx->boyx",
            x_ft[:, :, : self.modes_y, : self.modes_x],
            self.weights1,
        )
        # Bottom-left quadrant (-ky, +kx)
        out_ft[:, :, -self.modes_y :, : self.modes_x] = torch.einsum(
            "biyx,ioyx->boyx",
            x_ft[:, :, -self.modes_y :, : self.modes_x],
            self.weights2,
        )

        # 4. Inverse 2D Real FFT back to spatial domain with exact shape specification
        out = torch.fft.irfft2(out_ft, s=(ny, nx), norm="ortho")
        return out


class FourierNeuralOperator2D(SurrogateModel):
    """Fourier Neural Operator (FNO-2D) for continuous physical field predictions.

    Architecture:
    1. Lifting projection: P: R^d_in -> R^width via Conv2D(kernel_size=1).
    2. Fourier layers: L blocks computing v_{l+1} = GELU(W_l(v_l) + K_l(v_l))
       where W_l is local 1x1 conv and K_l is SpectralConv2D.
    3. Projection decoder: Q: R^width -> R^128 -> R^out_channels via 1x1 convs.

    Non-Periodic Physical Boundary Conditions:
    Standard FFT spectral operators inherently assume periodic boundary conditions.
    To learn non-periodic Dirichlet and Neumann boundary value problems, the model
    receives explicit boundary condition channels (prescribed values and boundary masks)
    concatenated with physical coordinates and parameter fields. The lifting projection P
    encodes this localized boundary information into the continuous latent representation.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 1,
        modes_x: int = 12,
        modes_y: int = 12,
        width: int = 32,
        depth: int = 4,
        activation: str = "gelu",
    ) -> None:
        """Initialize FNO-2D model.

        Args:
            in_channels: Number of input channels (e.g. 12 for canonical k, q, coords, BCs).
            out_channels: Number of output channels (1 for temperature field).
            modes_x: Retained Fourier modes along x.
            modes_y: Retained Fourier modes along y.
            width: Latent channel width / hidden feature dimension.
            depth: Number of Fourier neural operator layers. Must be >= 1.
            activation: Nonlinear activation function ('gelu' or 'relu').
        """
        super().__init__(in_channels=in_channels, out_channels=out_channels)
        if modes_x < 1 or modes_y < 1:
            raise ValueError("modes_x and modes_y must be >= 1.")
        if width < 1:
            raise ValueError(f"width must be >= 1, got {width}.")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}.")

        self.modes_x = modes_x
        self.modes_y = modes_y
        self.width = width
        self.depth = depth

        # 1. Lifting projection: Conv2D 1x1
        self.p = nn.Conv2d(in_channels, width, kernel_size=1)
        nn.init.kaiming_normal_(self.p.weight, nonlinearity="relu")
        if self.p.bias is not None:
            nn.init.zeros_(self.p.bias)

        # 2. Spectral convolution layers and local linear bypass transforms
        self.spectral_layers = nn.ModuleList()
        self.bypass_layers = nn.ModuleList()
        for _ in range(depth):
            self.spectral_layers.append(
                SpectralConv2D(
                    in_channels=width,
                    out_channels=width,
                    modes_x=modes_x,
                    modes_y=modes_y,
                )
            )
            bypass = nn.Conv2d(width, width, kernel_size=1)
            nn.init.kaiming_normal_(bypass.weight, nonlinearity="relu")
            if bypass.bias is not None:
                nn.init.zeros_(bypass.bias)
            self.bypass_layers.append(bypass)

        # Activation
        act = activation.lower()
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation '{activation}'.")

        # 3. Projection decoder: Conv2D 1x1 -> Activation -> Conv2D 1x1
        self.q0 = nn.Conv2d(width, 128, kernel_size=1)
        self.q1 = nn.Conv2d(128, out_channels, kernel_size=1)
        nn.init.kaiming_normal_(self.q0.weight, nonlinearity="relu")
        nn.init.xavier_uniform_(self.q1.weight)
        if self.q0.bias is not None:
            nn.init.zeros_(self.q0.bias)
        if self.q1.bias is not None:
            nn.init.zeros_(self.q1.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, ny, nx).

        Returns:
            Output solution field of shape (batch, 1, ny, nx).
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (batch, C, ny, nx), got {x.dim()}D.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Input channel mismatch: expected {self.in_channels} channels, got {x.shape[1]}."
            )

        # 1. Lift to latent width
        h = self.p(x)

        # 2. Fourier layers
        for spec_conv, bypass in zip(self.spectral_layers, self.bypass_layers, strict=True):
            h_spec = spec_conv(h)
            h_lin = bypass(h)
            h = self.act(h_spec + h_lin)

        # 3. Project to output field
        h = self.act(self.q0(h))
        out = self.q1(h)
        return out
