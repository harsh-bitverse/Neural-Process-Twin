"""Uncertainty quantification methods for Neural Process Twin.

This package provides uncertainty estimation techniques for spatial field
predictions, enabling safety margins in engineering digital twin workflows.

Capabilities:
    - Deep Ensembles for epistemic uncertainty estimation.
    - Split Conformal prediction intervals for distribution-free spatial field coverage.
    - Out-of-distribution (OOD) detection using ensemble disagreement and physical input domains.
    - Uncertainty quality evaluation: risk-coverage curves, error correlation, and worst-case detection.
"""

from __future__ import annotations

from neural_process_twin.uncertainty.calibration import (
    ConformalCalibrationResult,
    ConformalIntervalCalibrator,
)
from neural_process_twin.uncertainty.ensemble import (
    DeepEnsemble,
    EnsemblePrediction,
)
from neural_process_twin.uncertainty.metrics import (
    RiskCoverageResult,
    UncertaintyQualityMetrics,
    compute_risk_coverage,
    compute_uncertainty_quality,
)
from neural_process_twin.uncertainty.ood import (
    EnsembleDisagreementDetector,
    InputDistributionDetector,
    OODDetectionResult,
    compute_auroc_auprc,
    evaluate_ood_detection,
)

__all__: list[str] = [
    "ConformalCalibrationResult",
    "ConformalIntervalCalibrator",
    "DeepEnsemble",
    "EnsembleDisagreementDetector",
    "EnsemblePrediction",
    "InputDistributionDetector",
    "OODDetectionResult",
    "RiskCoverageResult",
    "UncertaintyQualityMetrics",
    "compute_auroc_auprc",
    "compute_risk_coverage",
    "compute_uncertainty_quality",
    "evaluate_ood_detection",
]
