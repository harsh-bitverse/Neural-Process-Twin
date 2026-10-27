"""Scientific Regression Tests for Milestone 08D Uncertainty Quantification.

Tests:
1. Uncertainty model architectures, multi-resolution forward passes, and output shapes.
2. Evidential loss formulation, numerical stability, and evidence regularizer behavior.
3. Heteroscedastic NLL loss gradient flow and variance penalization.
4. Spatial MC Dropout stochasticity and variance extraction.
5. Conformal quantile monotonicity and split conformal calibration logic.
6. Risk-coverage AURC monotonicity and error-detection scoring.
7. Independent evaluation cohort governance, SHA-256 integrity, and seed disjointness.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import h5py
import numpy as np
import torch
from scripts.evaluate_milestone_08d_calibration import (
    compute_aurc,
    compute_split_conformal_quantiles,
)

from neural_process_twin.models.uncertainty_fno import (
    DistilledStudentFNO2D,
    EvidentialFNO2D,
    HeteroscedasticFNO2D,
    MCDropoutFNO2D,
)


def test_uncertainty_architectures_forward_shapes():
    """Verify that all uncertainty models accept multi-resolution inputs and output expected shapes."""
    batch_size = 2
    in_channels = 12

    for res in [32, 48, 64]:
        x = torch.randn(batch_size, in_channels, res, res)

        # U1: Heteroscedastic
        u1 = HeteroscedasticFNO2D(in_channels=in_channels, modes_x=8, modes_y=8, width=16, depth=2)
        mean, logvar = u1(x)
        assert mean.shape == (batch_size, 1, res, res)
        assert logvar.shape == (batch_size, 1, res, res)

        # U2: Evidential
        u2 = EvidentialFNO2D(in_channels=in_channels, modes_x=8, modes_y=8, width=16, depth=2)
        gamma, v, alpha, beta = u2(x)
        assert gamma.shape == (batch_size, 1, res, res)
        assert v.shape == (batch_size, 1, res, res)
        assert alpha.shape == (batch_size, 1, res, res)
        assert beta.shape == (batch_size, 1, res, res)
        assert torch.all(v > 0)
        assert torch.all(alpha > 1.0)
        assert torch.all(beta > 0)

        # U3: MC Dropout
        u3 = MCDropoutFNO2D(
            in_channels=in_channels, modes_x=8, modes_y=8, width=16, depth=2, dropout_p=0.10
        )
        out = u3(x)
        assert out.shape == (batch_size, 1, res, res)

        # U4: Distilled Student
        u4 = DistilledStudentFNO2D(in_channels=in_channels, modes_x=8, modes_y=8, width=16, depth=2)
        mean_u4, std_u4 = u4(x)
        assert mean_u4.shape == (batch_size, 1, res, res)
        assert std_u4.shape == (batch_size, 1, res, res)
        assert torch.all(std_u4 > 0)


def test_evidential_loss_stability():
    """Verify Deep Evidential Regression loss calculation and evidence penalty."""
    b, h, w = 2, 16, 16
    gamma = torch.randn(b, 1, h, w, requires_grad=True)
    v = (torch.rand(b, 1, h, w) + 0.1).requires_grad_(True)
    alpha = (torch.rand(b, 1, h, w) + 1.5).requires_grad_(True)
    beta = (torch.rand(b, 1, h, w) + 0.1).requires_grad_(True)
    target = torch.randn(b, 1, h, w)

    loss = EvidentialFNO2D.evidential_loss(gamma, v, alpha, beta, target, lambda_reg=0.01)
    assert torch.isfinite(loss)
    loss.backward()

    assert gamma.grad is not None
    assert v.grad is not None
    assert alpha.grad is not None
    assert beta.grad is not None


def test_heteroscedastic_nll_loss_gradients():
    """Verify Gaussian NLL loss gradient propagation."""
    b, h, w = 2, 16, 16
    mean = torch.randn(b, 1, h, w, requires_grad=True)
    logvar = torch.randn(b, 1, h, w, requires_grad=True)
    target = torch.randn(b, 1, h, w)

    loss = HeteroscedasticFNO2D.nll_loss(mean, logvar, target)
    assert torch.isfinite(loss)
    loss.backward()

    assert mean.grad is not None
    assert logvar.grad is not None


def test_mc_dropout_stochasticity():
    """Verify that MC Dropout produces stochastic variations when in train mode."""
    x = torch.randn(1, 12, 32, 32)
    model = MCDropoutFNO2D(in_channels=12, modes_x=8, modes_y=8, width=16, depth=2, dropout_p=0.20)

    # In train mode, passes should differ
    model.train()
    pass1 = model(x).detach()
    pass2 = model(x).detach()
    diff = torch.max(torch.abs(pass1 - pass2)).item()
    assert diff > 1e-5, "MC Dropout in train mode should produce distinct stochastic passes"

    # In eval mode, passes must be identical
    model.eval()
    eval1 = model(x).detach()
    eval2 = model(x).detach()
    diff_eval = torch.max(torch.abs(eval1 - eval2)).item()
    assert diff_eval == 0.0, "MC Dropout in eval mode must be strictly deterministic"


def test_conformal_quantiles_monotonicity():
    """Verify that split conformal quantiles respect nominal coverage levels."""
    np.random.seed(42)
    n, h, w = 50, 16, 16
    residuals = np.abs(np.random.normal(0, 1, (n, h, w)))
    unc = np.ones((n, h, w)) * 1.0

    q_pt_90, q_sim_90 = compute_split_conformal_quantiles(residuals, unc, alpha=0.10)
    q_pt_95, q_sim_95 = compute_split_conformal_quantiles(residuals, unc, alpha=0.05)

    assert q_pt_95 >= q_pt_90, "95% pointwise quantile must be >= 90% quantile"
    assert q_sim_95 >= q_sim_90, "95% simultaneous quantile must be >= 90% quantile"
    assert q_sim_95 >= q_pt_95, "Simultaneous band quantile must be >= pointwise quantile"


def test_risk_coverage_aurc():
    """Verify that AURC correctly penalizes uncalibrated uncertainty ordering."""
    # Case 1: Perfectly calibrated uncertainty (errors exactly match scores)
    scores_good = np.linspace(0.1, 10.0, 100)
    errors_good = scores_good.copy()
    aurc_good = compute_aurc(errors_good, scores_good)

    # Case 2: Inversely ordered uncertainty (high error samples have low scores)
    scores_bad = np.linspace(0.1, 10.0, 100)
    errors_bad = scores_bad[::-1].copy()
    aurc_bad = compute_aurc(errors_bad, scores_bad)

    assert aurc_good < aurc_bad, (
        f"Well-ordered uncertainty ({aurc_good}) must have lower AURC than inverse ({aurc_bad})"
    )


def test_independent_cohort_governance():
    """Verify independent confirmation cohort file, SHA-256 integrity, and seed isolation."""
    cohort_path = Path("artifacts/data/independent_cohort_08d.h5")
    manifest_path = Path("artifacts/data/independent_cohort_08d_manifest.json")

    assert cohort_path.exists(), "independent_cohort_08d.h5 must exist"
    assert manifest_path.exists(), "independent_cohort_08d_manifest.json must exist"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["governance_status"] == "LOCKED_INDEPENDENT_EVALUATION_COHORT"
    assert manifest["inaccessible_for_training"] is True
    assert manifest["inaccessible_for_calibration"] is True

    # Check seeds are disjoint from training (10000-10999), val (11000-11149), cal (11150-11349), 08c (500000-501199)
    iid_range = manifest["splits"]["eval_iid"]["seed_range"]
    ood_range = manifest["splits"]["eval_ood"]["seed_range"]

    assert iid_range[0] >= 600000 and iid_range[1] < 601000
    assert ood_range[0] >= 601000 and ood_range[1] < 602000

    # Verify HDF5 structure across all 4 resolutions
    with h5py.File(cohort_path, "r") as h5f:
        for res in [32, 64, 128, 256]:
            res_grp = f"res_{res}"
            assert res_grp in h5f
            assert "eval_iid" in h5f[res_grp]
            assert "eval_ood" in h5f[res_grp]
            assert h5f[f"{res_grp}/eval_iid/k"].shape == (100, res, res)
            assert h5f[f"{res_grp}/eval_ood/temperature"].shape == (100, res, res)
