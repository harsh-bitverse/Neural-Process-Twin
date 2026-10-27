"""Regression tests for Milestone 08B.2 Canonical Evaluation & Reconciliation.

Verifies:
1. Physical contrast subgroup identification (exact 153 HC vs 47 GRF breakdown).
2. Proof of 08B.1 slice misclassification ([153:200] containing 36 HC and 11 GRF).
3. Canonical heat balance sign conventions and bounded relative imbalance.
4. Exact numerical reproduction of historical GRF metrics across A0 and 5-member Deep Ensemble.
5. Mathematical consistency of pooled RMSE vs sample-mean RMSE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_process_twin.data.boundary_encoding import decode_boundary_conditions
from neural_process_twin.data.dataset import HeatFieldDataset
from neural_process_twin.solvers.diagnostics import compute_heat_balance


@pytest.fixture(scope="module")
def ood_dataset() -> HeatFieldDataset:
    ds_path = Path("artifacts/data/scaled_dataset.h5")
    if not ds_path.is_file():
        pytest.skip(f"Dataset missing: {ds_path}")
    return HeatFieldDataset(ds_path, split="test_ood", include_coords=True, include_bcs=True)


@pytest.fixture(scope="module")
def reconciled_json() -> dict:
    json_path = Path("artifacts/reports/milestone_08b2_reconciled_metrics.json")
    if not json_path.is_file():
        pytest.skip(f"Reconciled results missing: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def test_test_ood_contrast_subgroup_distribution(ood_dataset: HeatFieldDataset):
    """Verify that test_ood contains exactly 153 high-contrast and 47 GRF samples."""
    contrasts = []
    for i in range(len(ood_dataset)):
        k_raw = ood_dataset[i]["k_raw"].squeeze().numpy()
        c = float(np.max(k_raw) / max(np.min(k_raw), 1e-12))
        contrasts.append(c)

    contrasts = np.array(contrasts)
    hc_mask = contrasts >= 20.0
    grf_mask = contrasts < 20.0

    assert len(ood_dataset) == 200
    assert int(np.sum(hc_mask)) == 153, "High-contrast inclusion count must be exactly 153"
    assert int(np.sum(grf_mask)) == 47, "Short-correlation GRF count must be exactly 47"

    # Verify interleaved distribution: first sample is GRF, second is HC
    assert grf_mask[0] is np.True_, "Sample 0 must be GRF"
    assert hc_mask[1] is np.True_, "Sample 1 must be High-Contrast Inclusion"


def test_historical_08b1_slice_misclassification_proof(ood_dataset: HeatFieldDataset):
    """Verify the exact mathematical mechanism of the 08B.1 subgroup discrepancy."""
    contrasts = []
    for i in range(len(ood_dataset)):
        k_raw = ood_dataset[i]["k_raw"].squeeze().numpy()
        c = float(np.max(k_raw) / max(np.min(k_raw), 1e-12))
        contrasts.append(c)

    contrasts = np.array(contrasts)

    # Slice [153:200] was labeled 'GRF' in 08B.1
    slice_153_200 = contrasts[153:200]
    hc_in_slice = int(np.sum(slice_153_200 >= 20.0))
    grf_in_slice = int(np.sum(slice_153_200 < 20.0))

    assert len(slice_153_200) == 47
    assert hc_in_slice == 36, (
        "Historical slice [153:200] must contain 36 high-contrast samples (76.6%)"
    )
    assert grf_in_slice == 11, "Historical slice [153:200] must contain only 11 GRF samples (23.4%)"

    # Slice [0:153] was labeled 'High-Contrast' in 08B.1
    slice_0_153 = contrasts[0:153]
    hc_in_first = int(np.sum(slice_0_153 >= 20.0))
    grf_in_first = int(np.sum(slice_0_153 < 20.0))

    assert len(slice_0_153) == 153
    assert hc_in_first == 117, (
        "Historical slice [0:153] must contain 117 high-contrast samples (76.5%)"
    )
    assert grf_in_first == 36, "Historical slice [0:153] must contain 36 GRF samples (23.5%)"


def test_energy_balance_sign_convention(ood_dataset: HeatFieldDataset):
    """Verify energy balance sign convention and non-negative absolute imbalance."""
    sample = ood_dataset[0]
    grid = ood_dataset.grid
    normalizer = ood_dataset.normalizer

    t_ref = sample["temperature_raw"].squeeze().numpy()
    k_raw = sample["k_raw"].squeeze().numpy()
    q_raw = sample["q_raw"].squeeze().numpy()
    bc_tensor = sample["x"][4:12].numpy()

    bcs = decode_boundary_conditions(bc_tensor, grid=grid, normalizer=normalizer)
    diag = compute_heat_balance(grid, t_ref, k_raw, q_raw, bcs)

    # Reference solver must balance to high precision (< 0.05 W)
    assert diag.flux_imbalance < 0.05, (
        f"Reference solve energy balance error {diag.flux_imbalance} W"
    )
    assert 0.0 <= diag.relative_imbalance <= 2.0, "Relative imbalance must be bounded in [0, 2]"


def test_reconciled_metrics_numerical_consistency(reconciled_json: dict):
    """Verify exact reproduction of historical metrics in reconciled JSON."""
    cands = reconciled_json["candidates"]

    # 1. A0 Seed 42
    a0_s42 = cands["A0_supervised_seed42"]
    canonical_grf_imb = a0_s42["test_ood_subgroups_canonical"]["short_correlation_grf"][
        "flux_imbalance_abs_w"
    ]["mean"]
    slice_grf_imb = a0_s42["test_ood_subgroups_historical_08b1_slice"]["slice_153_200"][
        "flux_imbalance_abs_w"
    ]["mean"]

    assert canonical_grf_imb == pytest.approx(90192.2, rel=1e-3), (
        "Canonical GRF imbalance must reproduce 08B reported value (90,192.2 W)"
    )
    assert slice_grf_imb == pytest.approx(24301.2, rel=1e-3), (
        "Slice [153:200] must reproduce 08B.1 reported value (24,301.2 W)"
    )

    # 2. Deep Ensemble 59.6 kW Anomaly Reproduction
    ens = reconciled_json["deep_ensemble"]
    ens_grf_imb = ens["test_ood_subgroups_canonical"]["short_correlation_grf"][
        "flux_imbalance_abs_w"
    ]["mean"]
    assert ens_grf_imb == pytest.approx(59591.9, abs=0.5), (
        f"Deep Ensemble must reproduce Milestone 05B anomaly of 59,591.9 W (got {ens_grf_imb:.1f} W)"
    )

    # 3. Pooled RMSE >= Sample Mean RMSE (Jensen's inequality)
    pooled_rmse = a0_s42["test_iid"]["pooled_rmse_k"]
    mean_rmse = a0_s42["test_iid"]["overall_rmse_k"]["mean"]
    assert pooled_rmse >= mean_rmse - 1e-6, (
        "Pooled RMSE must be >= sample mean RMSE by Jensen's inequality"
    )
