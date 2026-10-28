"""Scientific regression tests for Milestone 08E.1: Safety Statistics, Routing Economics, and Production Promotion Audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neural_process_twin.inference.adaptive_routing import calculate_wilson_interval


def test_wilson_score_interval_exact_boundaries() -> None:
    """Verify exact Wilson score interval calculations against closed-form mathematical expectations."""
    # 0 events out of 100 trials: Rule of Three upper bound ~ 3.7%
    p_hat, ci_l, ci_u = calculate_wilson_interval(0, 100)
    assert p_hat == 0.0
    assert ci_l == 0.0
    assert ci_u is not None
    assert 0.035 < ci_u < 0.038

    # 100 events out of 100 trials: Lower bound ~ 96.3%
    p_hat, ci_l, ci_u = calculate_wilson_interval(100, 100)
    assert p_hat == 1.0
    assert ci_l is not None
    assert 0.962 < ci_l < 0.965
    assert ci_u == 1.0

    # 11 events out of 100 trials (Unconditional R2 rate)
    p_hat, ci_l, ci_u = calculate_wilson_interval(11, 100)
    assert p_hat == 0.11
    assert ci_l is not None and ci_u is not None
    assert np.isclose(ci_l, 0.0625, atol=1e-3)
    assert np.isclose(ci_u, 0.1863, atol=1e-3)

    # 11 events out of 11 accepted trials (Conditional R2 rate)
    p_hat, ci_l, ci_u = calculate_wilson_interval(11, 11)
    assert p_hat == 1.0
    assert ci_l is not None
    assert np.isclose(ci_l, 0.7412, atol=1e-3)
    assert ci_u == 1.0


def test_empty_cohort_conditional_rate_is_unestimable() -> None:
    """Verify that an empty accepted cohort (N=0) yields unestimable conditional failure rate."""
    p_hat, ci_l, ci_u = calculate_wilson_interval(0, 0)
    assert p_hat is None
    assert ci_l is None
    assert ci_u is None


def test_artifact_0_793_root_cause() -> None:
    """Verify the exact mathematical origin of the 0.793 artifact from Milestone 08E."""
    # When denominator was forced to max(0, 1) = 1, Wilson upper bound evaluated:
    p_hat, ci_l, ci_u = calculate_wilson_interval(0, 1)
    assert p_hat == 0.0
    assert ci_l == 0.0
    assert ci_u is not None
    assert np.isclose(ci_u, 0.79345, atol=1e-4)


def test_unconditional_vs_conditional_distinct_denominators() -> None:
    """Verify that unconditional and conditional rates use distinct denominators and yield different CIs."""
    n_total = 100
    n_accepted = 11
    k_failures = 11

    # Unconditional: k / n_total
    p_uncond, l_uncond, u_uncond = calculate_wilson_interval(k_failures, n_total)
    # Conditional: k / n_accepted
    p_cond, l_cond, u_cond = calculate_wilson_interval(k_failures, n_accepted)

    assert p_uncond == 0.11
    assert p_cond == 1.0
    assert l_uncond is not None and u_uncond is not None
    assert l_cond is not None and u_cond is not None
    assert u_uncond < l_cond  # The two intervals do not overlap!


def test_physical_flux_imbalance_anomaly_classification() -> None:
    """Verify that the 115.16% threshold cannot represent a physical conservation guarantee."""
    audit_file = Path("artifacts/reports/milestone_08e1_audit_results.json")
    if not audit_file.exists():
        pytest.skip("Audit results file not yet generated")

    with open(audit_file) as f:
        data = json.load(f)

    p3 = data["phase3_physical_conservation_threshold"]["threshold_audit"]
    assert p3["threshold_value_pct"] > 100.0
    assert p3["is_physical_conservation_guarantee"] is False
    assert p3["scientific_classification"] == "EMPIRICAL_ANOMALY_CLASSIFIER_ONLY"


def test_r3_accepted_iid_exceeds_engineering_tolerance() -> None:
    """Verify that R3 violates the MaxAE <= 10 K engineering tolerance on accepted IID predictions."""
    audit_file = Path("artifacts/reports/milestone_08e1_audit_results.json")
    if not audit_file.exists():
        pytest.skip("Audit results file not yet generated")

    with open(audit_file) as f:
        data = json.load(f)

    r3_iid = data["phase4_accepted_iid_evaluation"]["R3_u3_t3_multi_pass"]
    assert r3_iid["engineering_tolerance_violated"] is True
    assert r3_iid["conditional_failure_rate_10k_pct"] > 50.0  # Measured > 75%
    assert r3_iid["maxae_percentiles_k"]["median"] > 10.0


def test_production_promotion_decision_is_rejection() -> None:
    """Verify that R3 is formally rejected for production promotion and R1 is retained."""
    audit_file = Path("artifacts/reports/milestone_08e1_audit_results.json")
    if not audit_file.exists():
        pytest.skip("Audit results file not yet generated")

    with open(audit_file) as f:
        data = json.load(f)

    decision = data["phase6_production_promotion_decision"]
    assert decision["verdict"] == "REJECT_PRODUCTION_PROMOTION"
    assert decision["approved_status"] == "RESEARCH_CANDIDATE"
    assert decision["production_baseline_action"] == "RETAIN_R1_PRODUCTION_HYBRID"
