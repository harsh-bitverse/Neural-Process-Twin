"""Surrogate model and neural operator architectures for Neural Process Twin.

This package houses machine learning architectures capable of learning mappings
between function spaces (e.g., coefficient and boundary fields to solution fields):
    - Fourier Neural Operator 2D (FNO-2D) for continuous operator learning.
    - CNN Baseline (CNNBaseline2D) for localized convolutional field mapping.
    - Spectral convolution layers with configurable Fourier mode truncation.
    - Unified SurrogateModel interface with forward, parameter counting, and device placement.
    - Model factory (build_model) for configuration-driven instantiation.
"""

from __future__ import annotations

from neural_process_twin.models.base import SurrogateModel
from neural_process_twin.models.boundary_aware_fno import (
    BoundaryAwareFNO2D,
    DiscreteBoundaryCorrection,
)
from neural_process_twin.models.cnn import CNNBaseline2D, ConvBlock2D
from neural_process_twin.models.factory import build_model
from neural_process_twin.models.fno2d import FourierNeuralOperator2D, SpectralConv2D
from neural_process_twin.models.uncertainty_fno import (
    DistilledStudentFNO2D,
    EvidentialFNO2D,
    HeteroscedasticFNO2D,
    MCDropoutFNO2D,
)

__all__ = [
    "BoundaryAwareFNO2D",
    "CNNBaseline2D",
    "ConvBlock2D",
    "DiscreteBoundaryCorrection",
    "DistilledStudentFNO2D",
    "EvidentialFNO2D",
    "FourierNeuralOperator2D",
    "HeteroscedasticFNO2D",
    "MCDropoutFNO2D",
    "SpectralConv2D",
    "SurrogateModel",
    "build_model",
]
