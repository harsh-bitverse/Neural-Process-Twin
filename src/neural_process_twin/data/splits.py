"""Dataset partitioning and Out-of-Distribution (OOD) protocols.

Defines disjoint dataset splits ('train', 'validation', 'test_iid', 'test_ood')
and explicit physical criteria separating in-distribution samples from out-of-distribution samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class SplitName(StrEnum):
    """Canonical dataset partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST_IID = "test_iid"
    TEST_OOD = "test_ood"
    CALIBRATION = "calibration"


@dataclass(frozen=True)
class PartitionRule:
    """Generation parameters defining the physical regime for a partition."""

    split: SplitName
    conductivity_regime: str  # e.g., 'grf_smooth' or 'composite_high_contrast'
    correlation_length_range: tuple[float, float]
    conductivity_range: tuple[float, float]
    source_amplitude_range: tuple[float, float]
    temperature_bc_range: tuple[float, float]

    def is_ood_sample(
        self,
        k_field: np.ndarray,
        q_field: np.ndarray,
        contrast_ratio_threshold: float = 15.0,
    ) -> bool:
        """Evaluate whether a sample meets the explicit physical OOD criteria."""
        k_min = float(np.min(k_field))
        k_max = float(np.max(k_field))
        contrast = k_max / (k_min + 1e-12)

        # OOD samples feature extreme contrast ratio or intense localized sources
        return (
            contrast >= contrast_ratio_threshold
            or float(np.max(q_field)) > self.source_amplitude_range[1]
        )


# Canonical IID Physical Regime
# Smooth GRF conductivity with moderate contrast, moderate localized heat source
IID_PARTITION_RULE = PartitionRule(
    split=SplitName.TRAIN,
    conductivity_regime="grf_smooth",
    correlation_length_range=(0.15, 0.35),
    conductivity_range=(0.5, 5.0),
    source_amplitude_range=(0.0, 500.0),
    temperature_bc_range=(290.0, 360.0),
)

# Canonical OOD Physical Regime
# Sharp piecewise-constant inclusions with extreme contrast ratio (> 20) or fine microstructures
OOD_PARTITION_RULE = PartitionRule(
    split=SplitName.TEST_OOD,
    conductivity_regime="composite_high_contrast",
    correlation_length_range=(0.03, 0.08),
    conductivity_range=(1.0, 50.0),  # contrast up to 50:1
    source_amplitude_range=(500.0, 2000.0),
    temperature_bc_range=(280.0, 420.0),
)


@dataclass(frozen=True)
class DatasetPartitionPlan:
    """Complete specification of sample counts, seeds, and regimes across splits."""

    base_seed: int
    num_train: int
    num_val: int
    num_test_iid: int
    num_test_ood: int
    num_cal: int = 0

    def __post_init__(self) -> None:
        if self.num_train < 1:
            raise ValueError(f"num_train must be >= 1, got {self.num_train}.")
        if self.num_val < 0 or self.num_test_iid < 0 or self.num_test_ood < 0 or self.num_cal < 0:
            raise ValueError("Partition sample counts must be non-negative.")

    @property
    def total_samples(self) -> int:
        """Total number of samples across all partitions."""
        return self.num_train + self.num_val + self.num_test_iid + self.num_test_ood + self.num_cal

    def get_split_count(self, split: SplitName | str) -> int:
        """Get sample count for a specific split."""
        s = SplitName(split)
        counts = {
            SplitName.TRAIN: self.num_train,
            SplitName.VALIDATION: self.num_val,
            SplitName.TEST_IID: self.num_test_iid,
            SplitName.TEST_OOD: self.num_test_ood,
            SplitName.CALIBRATION: self.num_cal,
        }
        return counts[s]

    def get_split_seeds(self, split: SplitName | str) -> list[int]:
        """Derive disjoint, deterministic integer seeds for each sample in a split."""
        s = SplitName(split)
        # Disjoint seed offsets per split
        offsets = {
            SplitName.TRAIN: 1_000_000,
            SplitName.VALIDATION: 2_000_000,
            SplitName.TEST_IID: 3_000_000,
            SplitName.TEST_OOD: 4_000_000,
            SplitName.CALIBRATION: 5_000_000,
        }
        base_offset = offsets[s] + self.base_seed
        count = self.get_split_count(s)
        return [base_offset + i for i in range(count)]


def verify_partitions_disjoint(split_seeds: dict[str, list[int]]) -> None:
    """Verify that no sample seed or index is shared between partitions.

    Args:
        split_seeds: Mapping from split name to list of sample seeds.

    Raises:
        ValueError: If overlapping seeds are detected between any pair of splits.
    """
    seen_seeds: dict[int, str] = {}
    for split_name, seeds in split_seeds.items():
        for s in seeds:
            if s in seen_seeds:
                raise ValueError(
                    f"Data leakage detected! Seed {s} is shared between partition "
                    f"'{seen_seeds[s]}' and '{split_name}'."
                )
            seen_seeds[s] = split_name


def verify_ood_criteria(
    k_samples_iid: list[np.ndarray],
    k_samples_ood: list[np.ndarray],
    contrast_threshold: float = 10.0,
) -> dict[str, Any]:
    """Verify that OOD conductivity fields exhibit strictly higher contrast than IID.

    Args:
        k_samples_iid: List of 2D conductivity fields from train or test_iid.
        k_samples_ood: List of 2D conductivity fields from test_ood.
        contrast_threshold: Expected minimum contrast ratio k_max / k_min for OOD.

    Returns:
        Summary dict containing contrast metrics.
    """
    iid_contrasts = [float(np.max(k) / (np.min(k) + 1e-12)) for k in k_samples_iid]
    ood_contrasts = [float(np.max(k) / (np.min(k) + 1e-12)) for k in k_samples_ood]

    avg_iid = float(np.mean(iid_contrasts)) if iid_contrasts else 0.0
    avg_ood = float(np.mean(ood_contrasts)) if ood_contrasts else 0.0

    if avg_ood <= avg_iid:
        raise ValueError(
            f"OOD contrast verification failed: mean OOD contrast ({avg_ood:.2f}) "
            f"must be strictly greater than mean IID contrast ({avg_iid:.2f})."
        )

    return {
        "mean_iid_contrast": avg_iid,
        "mean_ood_contrast": avg_ood,
        "min_ood_contrast": min(ood_contrasts) if ood_contrasts else 0.0,
        "max_iid_contrast": max(iid_contrasts) if iid_contrasts else 0.0,
        "contrast_threshold": contrast_threshold,
    }
