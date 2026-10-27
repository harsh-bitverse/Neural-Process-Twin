"""Hybrid Digital Twin inference engine and safety orchestration contracts.

Provides unified inference requests, multi-tiered safety validation, surrogate
execution, conformal uncertainty estimation, discrete physical diagnostics,
and fail-closed numerical fallback.
"""

from __future__ import annotations

from neural_process_twin.inference.contracts import (
    ConformalIntervalOutput,
    ExecutionLatency,
    HybridInferenceResult,
    InferenceOptions,
    InferenceProvenance,
    InferenceRequest,
    InferenceStatus,
    PhysicalDiagnosticsOutput,
    RejectionCode,
    RejectionRecord,
)
from neural_process_twin.inference.engine import HybridInferenceEngine
from neural_process_twin.inference.screening import SafetyScreener, ScreeningResult
from neural_process_twin.inference.validator import InputValidator, ValidationResult

__all__ = [
    "ConformalIntervalOutput",
    "ExecutionLatency",
    "HybridInferenceEngine",
    "HybridInferenceResult",
    "InferenceOptions",
    "InferenceProvenance",
    "InferenceRequest",
    "InferenceStatus",
    "InputValidator",
    "PhysicalDiagnosticsOutput",
    "RejectionCode",
    "RejectionRecord",
    "SafetyScreener",
    "ScreeningResult",
    "ValidationResult",
]
