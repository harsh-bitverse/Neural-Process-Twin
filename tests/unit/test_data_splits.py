"""Unit tests for dataset splits, seed derivation, and OOD verification."""

from __future__ import annotations

import numpy as np
import pytest

from neural_process_twin.data.splits import (
    OOD_PARTITION_RULE,
    DatasetPartitionPlan,
    SplitName,
    verify_ood_criteria,
    verify_partitions_disjoint,
)


@pytest.mark.unit
def test_partition_plan_counts_and_seeds() -> None:
    """Test sample counts, total samples, and disjoint seed sequences."""
    plan = DatasetPartitionPlan(
        base_seed=42,
        num_train=100,
        num_val=20,
        num_test_iid=20,
        num_test_ood=10,
    )
    assert plan.total_samples == 150
    assert plan.get_split_count(SplitName.TRAIN) == 100
    assert plan.get_split_count(SplitName.VALIDATION) == 20
    assert plan.get_split_count(SplitName.TEST_IID) == 20
    assert plan.get_split_count(SplitName.TEST_OOD) == 10

    # Verify seeds are unique and non-overlapping across partitions
    all_seeds = {
        split: plan.get_split_seeds(split)
        for split in (
            SplitName.TRAIN,
            SplitName.VALIDATION,
            SplitName.TEST_IID,
            SplitName.TEST_OOD,
        )
    }
    # Should pass without error
    verify_partitions_disjoint(all_seeds)


@pytest.mark.unit
def test_partitions_disjoint_leakage_detection() -> None:
    """Test that overlapping seeds trigger ValueError to prevent data leakage."""
    bad_seeds = {
        "train": [1, 2, 3, 4],
        "validation": [5, 6, 3],  # Seed 3 overlaps!
    }
    with pytest.raises(ValueError, match="Data leakage detected"):
        verify_partitions_disjoint(bad_seeds)


@pytest.mark.unit
def test_ood_contrast_verification() -> None:
    """Test verification of OOD contrast ratio separation."""
    # IID samples: low contrast ratio (around 3 to 5)
    iid_k = [np.full((8, 8), 1.0) for _ in range(5)]
    for arr in iid_k:
        arr[0, 0] = 4.0  # contrast 4:1

    # OOD samples: high contrast ratio (e.g. 30:1)
    ood_k = [np.full((8, 8), 1.0) for _ in range(5)]
    for arr in ood_k:
        arr[0, 0] = 30.0  # contrast 30:1

    summary = verify_ood_criteria(iid_k, ood_k, contrast_threshold=10.0)
    assert summary["mean_ood_contrast"] > summary["mean_iid_contrast"]
    assert np.isclose(summary["min_ood_contrast"], 30.0, atol=1e-5)

    # Reversed sets should raise ValueError
    with pytest.raises(ValueError, match="OOD contrast verification failed"):
        verify_ood_criteria(ood_k, iid_k)


@pytest.mark.unit
def test_partition_rule_is_ood_sample() -> None:
    """Test PartitionRule.is_ood_sample flag."""
    rule = OOD_PARTITION_RULE
    k_iid = np.full((16, 16), 2.0)
    k_iid[0, 0] = 4.0  # contrast 2:1
    q_iid = np.full((16, 16), 200.0)
    assert not rule.is_ood_sample(k_iid, q_iid)

    # High contrast conductivity
    k_ood = np.full((16, 16), 1.0)
    k_ood[0, 0] = 50.0  # contrast 50:1
    assert rule.is_ood_sample(k_ood, q_iid)

    # Intense source excursion
    q_ood = np.full((16, 16), 5000.0)  # > 2000.0 W/m^3
    assert rule.is_ood_sample(k_iid, q_ood)
