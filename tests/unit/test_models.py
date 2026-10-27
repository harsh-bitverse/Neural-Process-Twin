"""Unit tests for surrogate model architectures, spectral layers, and model factory."""

from __future__ import annotations

import pytest
import torch

from neural_process_twin.models.base import SurrogateModel
from neural_process_twin.models.cnn import CNNBaseline2D, ConvBlock2D
from neural_process_twin.models.factory import build_model
from neural_process_twin.models.fno2d import FourierNeuralOperator2D, SpectralConv2D
from neural_process_twin.utils.config import ModelConfig


class TestCNNBaseline2D:
    """Test suite for 2D convolutional baseline model."""

    def test_forward_output_shape(self) -> None:
        model = CNNBaseline2D(in_channels=12, out_channels=1, width=16, depth=3)
        x = torch.randn(2, 12, 32, 32)
        out = model(x)
        assert out.shape == (2, 1, 32, 32)

    def test_parameter_count(self) -> None:
        model = CNNBaseline2D(in_channels=12, out_channels=1, width=16, depth=3)
        counts = model.count_parameters()
        assert "total_params" in counts
        assert "trainable_params" in counts
        assert counts["total_params"] > 0
        assert counts["total_params"] == counts["trainable_params"]

    def test_gradient_flow(self) -> None:
        model = CNNBaseline2D(in_channels=12, out_channels=1, width=16, depth=2)
        x = torch.randn(2, 12, 16, 16, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"Parameter {name} has no gradient."
            assert torch.isfinite(p.grad).all(), f"Parameter {name} has non-finite gradient."
            assert (p.grad != 0).any(), f"Parameter {name} gradient is all zeros."

    def test_invalid_input_shapes(self) -> None:
        model = CNNBaseline2D(in_channels=12, out_channels=1)
        with pytest.raises(ValueError, match="Expected 4D input tensor"):
            model(torch.randn(12, 32, 32))

        with pytest.raises(ValueError, match="Input channel mismatch"):
            model(torch.randn(1, 4, 32, 32))

    def test_replicate_padding_preserves_resolution(self) -> None:
        block = ConvBlock2D(in_channels=8, out_channels=8, kernel_size=3, padding_mode="replicate")
        x = torch.randn(1, 8, 25, 27)
        out = block(x)
        assert out.shape == (1, 8, 25, 27)


class TestFourierNeuralOperator2D:
    """Test suite for 2D Fourier Neural Operator model and spectral layers."""

    @pytest.mark.parametrize(
        ("ny", "nx"),
        [
            (16, 16),  # Even-even
            (15, 15),  # Odd-odd
            (16, 17),  # Even-odd
            (17, 16),  # Odd-even
            (32, 32),  # Standard baseline resolution
        ],
    )
    def test_fno_odd_even_resolutions(self, ny: int, nx: int) -> None:
        model = FourierNeuralOperator2D(
            in_channels=12,
            out_channels=1,
            modes_x=6,
            modes_y=6,
            width=16,
            depth=2,
        )
        x = torch.randn(2, 12, ny, nx)
        out = model(x)
        assert out.shape == (2, 1, ny, nx)

    def test_spectral_conv_gradient_flow(self) -> None:
        layer = SpectralConv2D(in_channels=8, out_channels=8, modes_x=4, modes_y=4)
        x = torch.randn(2, 8, 16, 16, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weights1.grad is not None
        assert layer.weights2.grad is not None
        assert torch.isfinite(layer.weights1.grad).all()
        assert torch.isfinite(layer.weights2.grad).all()

    def test_nyquist_mode_validation(self) -> None:
        layer = SpectralConv2D(in_channels=4, out_channels=4, modes_x=10, modes_y=10)
        # For nx=16, max_modes_x is 16//2 + 1 = 9
        x = torch.randn(1, 4, 16, 16)
        with pytest.raises(ValueError, match="exceeds Nyquist limit"):
            layer(x)

    def test_fno_parameter_count(self) -> None:
        model = FourierNeuralOperator2D(
            in_channels=12,
            out_channels=1,
            modes_x=8,
            modes_y=8,
            width=24,
            depth=3,
        )
        counts = model.count_parameters()
        assert counts["total_params"] > 0
        assert counts["total_params"] == counts["trainable_params"]


class TestModelFactory:
    """Test suite for build_model factory function."""

    def test_build_fno_from_dataclass(self) -> None:
        cfg = ModelConfig(
            architecture="fno2d",
            in_channels=12,
            out_channels=1,
            width=16,
            depth=2,
            modes_x=6,
            modes_y=6,
        )
        model = build_model(cfg)
        assert isinstance(model, SurrogateModel)
        assert isinstance(model, FourierNeuralOperator2D)
        assert model.in_channels == 12
        assert model.out_channels == 1

    def test_build_cnn_from_dict(self) -> None:
        cfg = {
            "architecture": "cnn2d",
            "in_channels": 12,
            "out_channels": 1,
            "width": 16,
            "depth": 3,
        }
        model = build_model(cfg)
        assert isinstance(model, CNNBaseline2D)
        assert model.in_channels == 12

    def test_unknown_architecture_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model architecture"):
            build_model({"architecture": "transformer2d"})
