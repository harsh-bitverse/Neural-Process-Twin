"""Unit tests for FieldNormalizer and reversible feature scaling."""

from __future__ import annotations

import numpy as np
import pytest

from neural_process_twin.data.normalization import (
    FieldNormalizer,
    NormalizationMethod,
)


@pytest.fixture
def sample_fields() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    # 10 samples of shape (16, 16)
    k = rng.uniform(0.5, 5.0, size=(10, 16, 16))
    q = rng.uniform(0.0, 1000.0, size=(10, 16, 16))
    temp = rng.uniform(300.0, 400.0, size=(10, 16, 16))
    return k, q, temp


@pytest.mark.unit
def test_normalizer_unfitted_guard() -> None:
    """Test that unfitted normalizer raises RuntimeError on transform attempts."""
    norm = FieldNormalizer()
    assert not norm.is_fitted
    with pytest.raises(RuntimeError, match="not fitted"):
        norm.transform_k(np.ones((4, 4)))
    with pytest.raises(RuntimeError, match="not fitted"):
        norm.inverse_transform_temperature(np.ones((4, 4)))
    with pytest.raises(RuntimeError, match="unfitted"):
        norm.to_dict()


@pytest.mark.unit
def test_standard_normalization_and_inversion(
    sample_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Test z-score standard normalization and exact temperature inversion."""
    k, q, temp = sample_fields
    norm = FieldNormalizer(method=NormalizationMethod.STANDARD).fit(k, q, temp)
    assert norm.is_fitted

    # Verify fitted statistics
    assert norm.t_stats is not None
    assert np.isclose(norm.t_stats.mean, np.mean(temp))
    assert np.isclose(norm.t_stats.std, np.std(temp))

    # Transform
    temp_norm = norm.transform_temperature(temp)
    assert np.isclose(np.mean(temp_norm), 0.0, atol=1e-5)
    assert np.isclose(np.std(temp_norm), 1.0, atol=1e-5)

    # Inverse transform
    temp_recon = norm.inverse_transform_temperature(temp_norm)
    assert np.allclose(temp_recon, temp, atol=1e-5)


@pytest.mark.unit
def test_minmax_normalization_and_inversion(
    sample_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Test min-max normalization and exact temperature inversion."""
    k, q, temp = sample_fields
    norm = FieldNormalizer(method=NormalizationMethod.MINMAX).fit(k, q, temp)

    temp_norm = norm.transform_temperature(temp)
    assert np.min(temp_norm) >= 0.0 - 1e-6
    assert np.max(temp_norm) <= 1.0 + 1e-6

    temp_recon = norm.inverse_transform_temperature(temp_norm)
    assert np.allclose(temp_recon, temp, atol=1e-5)


@pytest.mark.unit
def test_normalizer_serialization_roundtrip(
    sample_fields: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Test serialization to dict and reconstruction."""
    k, q, temp = sample_fields
    norm = FieldNormalizer(method=NormalizationMethod.STANDARD).fit(k, q, temp)
    state = norm.to_dict()

    restored = FieldNormalizer.from_dict(state)
    assert restored.is_fitted
    assert restored.method == norm.method

    temp_norm_orig = norm.transform_temperature(temp)
    temp_norm_rest = restored.transform_temperature(temp)
    assert np.allclose(temp_norm_orig, temp_norm_rest)
