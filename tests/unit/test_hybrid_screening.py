"""Unit tests for SafetyScreener multi-tiered acceptance logic."""

from __future__ import annotations

import numpy as np
import pytest

from neural_process_twin.inference.contracts import (
    InferenceOptions,
    PhysicalDiagnosticsOutput,
    RejectionCode,
)
from neural_process_twin.inference.screening import SafetyScreener


@pytest.fixture
def clean_surrogate_data() -> dict[str, np.ndarray]:
    """Fixture providing clean surrogate predictions in-domain."""
    return {
        "t_pred": np.full((32, 32), 320.0),
        "s_pred": np.full((32, 32), 0.5),  # mean 0.5 K < 0.9978 K, max 0.5 K < 4.4526 K
        "k_field": np.full((32, 32), 2.0),  # contrast 1.0 < 7.16
        "q_field": np.full((32, 32), 100.0),  # max 100.0 < 912.8
    }


@pytest.mark.unit
def test_screener_accepts_clean_prediction(clean_surrogate_data: dict[str, np.ndarray]) -> None:
    """Test that in-domain prediction with low disagreement is accepted."""
    screener = SafetyScreener()
    options = InferenceOptions()
    res = screener.screen(
        predicted_temperature_k=clean_surrogate_data["t_pred"],
        ensemble_std_k=clean_surrogate_data["s_pred"],
        k_field=clean_surrogate_data["k_field"],
        q_field=clean_surrogate_data["q_field"],
        diagnostics=None,
        options=options,
    )
    assert res.accepted is True
    assert res.rejection.code == RejectionCode.NONE
    assert np.isclose(res.mean_disagreement_kelvin, 0.5)
    assert np.isclose(res.max_disagreement_kelvin, 0.5)


@pytest.mark.unit
def test_screener_rejects_non_finite_surrogate_output(
    clean_surrogate_data: dict[str, np.ndarray],
) -> None:
    """Test rejection when surrogate produces NaN values."""
    screener = SafetyScreener()
    t_nan = clean_surrogate_data["t_pred"].copy()
    t_nan[10, 10] = np.nan

    res = screener.screen(
        predicted_temperature_k=t_nan,
        ensemble_std_k=clean_surrogate_data["s_pred"],
        k_field=clean_surrogate_data["k_field"],
        q_field=clean_surrogate_data["q_field"],
        diagnostics=None,
        options=InferenceOptions(),
    )
    assert res.accepted is False
    assert res.rejection.code == RejectionCode.SURROGATE_OUTPUT_NON_FINITE


@pytest.mark.unit
def test_screener_rejects_high_mean_disagreement(
    clean_surrogate_data: dict[str, np.ndarray],
) -> None:
    """Test rejection when spatial mean disagreement exceeds 0.9978 K."""
    screener = SafetyScreener()
    s_high = np.full((32, 32), 1.25)  # 1.25 K > 0.9978 K

    res = screener.screen(
        predicted_temperature_k=clean_surrogate_data["t_pred"],
        ensemble_std_k=s_high,
        k_field=clean_surrogate_data["k_field"],
        q_field=clean_surrogate_data["q_field"],
        diagnostics=None,
        options=InferenceOptions(),
    )
    assert res.accepted is False
    assert res.rejection.code == RejectionCode.HIGH_ENSEMBLE_DISAGREEMENT
    assert res.rejection.triggering_metric == "spatial_mean_disagreement_k"
    assert res.rejection.observed_value == 1.25
    assert res.rejection.threshold_value == 0.9978


@pytest.mark.unit
def test_screener_rejects_high_peak_disagreement(
    clean_surrogate_data: dict[str, np.ndarray],
) -> None:
    """Test rejection when single cell disagreement exceeds 4.4526 K despite low mean."""
    screener = SafetyScreener()
    s_spike = np.full((32, 32), 0.2)
    s_spike[5, 5] = 5.5  # Peak 5.5 K > 4.4526 K, but mean is ~0.205 K < 0.9978 K

    res = screener.screen(
        predicted_temperature_k=clean_surrogate_data["t_pred"],
        ensemble_std_k=s_spike,
        k_field=clean_surrogate_data["k_field"],
        q_field=clean_surrogate_data["q_field"],
        diagnostics=None,
        options=InferenceOptions(),
    )
    assert res.accepted is False
    assert res.rejection.code == RejectionCode.HIGH_PEAK_DISAGREEMENT
    assert res.rejection.observed_value == 5.5
    assert res.rejection.threshold_value == 4.4526


@pytest.mark.unit
def test_screener_rejects_high_contrast(clean_surrogate_data: dict[str, np.ndarray]) -> None:
    """Test rejection when conductivity contrast ratio exceeds 7.16."""
    screener = SafetyScreener()
    k_contrast = np.full((32, 32), 1.0)
    k_contrast[0, 0] = 20.0  # Contrast 20:1 > 7.16:1

    res = screener.screen(
        predicted_temperature_k=clean_surrogate_data["t_pred"],
        ensemble_std_k=clean_surrogate_data["s_pred"],
        k_field=k_contrast,
        q_field=clean_surrogate_data["q_field"],
        diagnostics=None,
        options=InferenceOptions(),
    )
    assert res.accepted is False
    assert res.rejection.code == RejectionCode.HIGH_INPUT_CONTRAST


@pytest.mark.unit
def test_screener_optional_pde_residual_gate(clean_surrogate_data: dict[str, np.ndarray]) -> None:
    """Test that configurable PDE residual threshold rejects when violated."""
    screener = SafetyScreener()
    diag = PhysicalDiagnosticsOutput(
        pde_residual_l2_w=150.0,
        pde_residual_relative=0.1,
        pde_residual_linf_w=20.0,
        q_generated_total_w=100.0,
        q_boundary_out_total_w=105.0,
        signed_net_imbalance_w=5.0,
        absolute_flux_imbalance_w=5.0,
        relative_flux_imbalance=0.05,
        boundary_heat_flow_out_w={},
        boundary_flux_densities_w_m2={},
    )

    # 1. Disabled threshold (default None): passes
    res_none = screener.screen(
        predicted_temperature_k=clean_surrogate_data["t_pred"],
        ensemble_std_k=clean_surrogate_data["s_pred"],
        k_field=clean_surrogate_data["k_field"],
        q_field=clean_surrogate_data["q_field"],
        diagnostics=diag,
        options=InferenceOptions(pde_residual_threshold_w=None),
    )
    assert res_none.accepted is True

    # 2. Strict threshold (100.0 W < 150.0 W): rejects
    res_strict = screener.screen(
        predicted_temperature_k=clean_surrogate_data["t_pred"],
        ensemble_std_k=clean_surrogate_data["s_pred"],
        k_field=clean_surrogate_data["k_field"],
        q_field=clean_surrogate_data["q_field"],
        diagnostics=diag,
        options=InferenceOptions(pde_residual_threshold_w=100.0),
    )
    assert res_strict.accepted is False
    assert res_strict.rejection.code == RejectionCode.EXCESSIVE_PDE_RESIDUAL
    assert res_strict.rejection.observed_value == 150.0
