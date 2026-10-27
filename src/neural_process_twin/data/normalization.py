"""Normalization and feature scaling for physical PDE fields.

Provides strict training-partition fitting to prevent data leakage, with reversible
transformations for temperature predictions and surrogate model evaluation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class NormalizationMethod(StrEnum):
    """Method for field normalization."""

    STANDARD = "standard"  # z-score: (x - mean) / std
    MINMAX = "minmax"  # affine min-max scaling to [0, 1] or [-1, 1]


@dataclass(frozen=True)
class FieldStatistics:
    """Empirical statistics of a scalar field evaluated over a dataset partition."""

    mean: float
    std: float
    min: float
    max: float

    def __post_init__(self) -> None:
        if self.std < 0.0:
            raise ValueError(f"Standard deviation cannot be negative, got {self.std}.")
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) cannot exceed max ({self.max}).")


class FieldNormalizer:
    """Physics-aware scalar field normalizer.

    IMPORTANT: To maintain rigorous scientific evaluation and prevent data leakage,
    instances of FieldNormalizer must be fitted exclusively on the 'train' partition.
    Validation and test partitions must be transformed using the statistics fitted from train.
    """

    def __init__(
        self,
        method: NormalizationMethod | str = NormalizationMethod.STANDARD,
        eps: float = 1e-8,
    ) -> None:
        """Initialize field normalizer.

        Args:
            method: Normalization method (standard or minmax).
            eps: Small epsilon to prevent division by zero for uniform fields.
        """
        self.method = NormalizationMethod(method)
        self.eps = float(eps)
        self.k_stats: FieldStatistics | None = None
        self.q_stats: FieldStatistics | None = None
        self.t_stats: FieldStatistics | None = None

    @property
    def is_fitted(self) -> bool:
        """Return True if statistics have been fitted."""
        return self.k_stats is not None and self.q_stats is not None and self.t_stats is not None

    def fit(
        self,
        k: np.ndarray,
        q: np.ndarray,
        temperature: np.ndarray,
    ) -> FieldNormalizer:
        """Fit empirical normalizer statistics on training partition fields.

        Args:
            k: Conductivity fields of shape (N, ny, nx) or (ny, nx) [W/(m K)].
            q: Volumetric heat source fields of shape (N, ny, nx) or (ny, nx) [W/m^3].
            temperature: Solution temperature fields of shape (N, ny, nx) or (ny, nx) [K].

        Returns:
            Fitted FieldNormalizer instance.
        """

        def compute_stats(arr: np.ndarray) -> FieldStatistics:
            arr_f = np.asarray(arr, dtype=np.float64)
            mean_val = float(np.mean(arr_f))
            std_val = float(np.std(arr_f))
            min_val = float(np.min(arr_f))
            max_val = float(np.max(arr_f))
            # Guard against zero std
            if std_val < self.eps:
                std_val = 1.0
            return FieldStatistics(mean=mean_val, std=std_val, min=min_val, max=max_val)

        self.k_stats = compute_stats(k)
        self.q_stats = compute_stats(q)
        self.t_stats = compute_stats(temperature)
        return self

    def _transform_field(self, arr: np.ndarray, stats: FieldStatistics) -> np.ndarray:
        """Transform an arbitrary field using the specified statistics."""
        if self.method == NormalizationMethod.STANDARD:
            return (arr - stats.mean) / (stats.std + self.eps)
        if self.method == NormalizationMethod.MINMAX:
            span = stats.max - stats.min
            if span < self.eps:
                return np.zeros_like(arr)
            return (arr - stats.min) / span
        raise NotImplementedError(f"Unsupported normalization method {self.method}")

    def _inverse_transform_field(self, arr: np.ndarray, stats: FieldStatistics) -> np.ndarray:
        """Invert transformation on a field using the specified statistics."""
        if self.method == NormalizationMethod.STANDARD:
            return arr * (stats.std + self.eps) + stats.mean
        if self.method == NormalizationMethod.MINMAX:
            span = stats.max - stats.min
            return arr * span + stats.min
        raise NotImplementedError(f"Unsupported normalization method {self.method}")

    def transform_k(self, k: np.ndarray) -> np.ndarray:
        """Transform conductivity field to normalized scale."""
        if self.k_stats is None:
            raise RuntimeError("FieldNormalizer is not fitted yet.")
        return self._transform_field(k, self.k_stats)

    def transform_q(self, q: np.ndarray) -> np.ndarray:
        """Transform volumetric heat source field to normalized scale."""
        if self.q_stats is None:
            raise RuntimeError("FieldNormalizer is not fitted yet.")
        return self._transform_field(q, self.q_stats)

    def transform_temperature(self, temperature: np.ndarray) -> np.ndarray:
        """Transform solution temperature field to normalized scale."""
        if self.t_stats is None:
            raise RuntimeError("FieldNormalizer is not fitted yet.")
        return self._transform_field(temperature, self.t_stats)

    def inverse_transform_temperature(self, temp_norm: np.ndarray) -> np.ndarray:
        """Invert normalized temperature prediction back to physical temperature in Kelvin [K]."""
        if self.t_stats is None:
            raise RuntimeError("FieldNormalizer is not fitted yet.")
        return self._inverse_transform_field(temp_norm, self.t_stats)

    def inverse_transform_k(self, k_norm: np.ndarray) -> np.ndarray:
        """Invert normalized conductivity back to physical units [W/(m K)]."""
        if self.k_stats is None:
            raise RuntimeError("FieldNormalizer is not fitted yet.")
        return self._inverse_transform_field(k_norm, self.k_stats)

    def inverse_transform_q(self, q_norm: np.ndarray) -> np.ndarray:
        """Invert normalized heat source back to physical units [W/m^3]."""
        if self.q_stats is None:
            raise RuntimeError("FieldNormalizer is not fitted yet.")
        return self._inverse_transform_field(q_norm, self.q_stats)

    def to_dict(self) -> dict[str, Any]:
        """Serialize normalizer configuration and fitted statistics to dictionary."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted FieldNormalizer.")
        return {
            "method": self.method.value,
            "eps": self.eps,
            "k_stats": asdict(self.k_stats) if self.k_stats else None,
            "q_stats": asdict(self.q_stats) if self.q_stats else None,
            "t_stats": asdict(self.t_stats) if self.t_stats else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldNormalizer:
        """Reconstruct FieldNormalizer from dictionary."""
        normalizer = cls(method=data["method"], eps=float(data.get("eps", 1e-8)))
        if data.get("k_stats"):
            normalizer.k_stats = FieldStatistics(**data["k_stats"])
        if data.get("q_stats"):
            normalizer.q_stats = FieldStatistics(**data["q_stats"])
        if data.get("t_stats"):
            normalizer.t_stats = FieldStatistics(**data["t_stats"])
        return normalizer
