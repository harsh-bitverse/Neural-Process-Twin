"""Typed request and response contracts for Hybrid Digital Twin inference.

Defines unified data models for problem specification, surrogate predictions,
calibrated prediction intervals, physical diagnostics, execution provenance,
and machine-readable status/rejection audit trails.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from neural_process_twin.solvers.boundary import BoundaryConditions
from neural_process_twin.solvers.grid import Grid2D


class InferenceStatus(StrEnum):
    """Terminal and intermediate execution status for the Hybrid Inference Engine."""

    ACCEPTED = "accepted"  # Surrogate prediction passed all validation and screening gates
    FALLBACK_SUCCEEDED = "fallback_succeeded"  # Fallback to numerical reference solver succeeded
    INVALID_INPUT = "invalid_input"  # Input problem failed physical or dimensional validation
    SOLVER_FAILED = "solver_failed"  # Numerical reference solver diverged, timed out, or failed
    FALLBACK_REQUIRED = "fallback_required"  # Intermediate decision state triggering fallback


class RejectionCode(StrEnum):
    """Machine-readable reason for surrogate rejection, fallback, or pipeline failure."""

    NONE = "none"
    NON_POSITIVE_CONDUCTIVITY = "non_positive_conductivity"
    NON_FINITE_VALUES = "non_finite_values"
    INCOMPATIBLE_GRID_DIMENSIONS = "incompatible_grid_dimensions"
    UNSUPPORTED_BOUNDARY_CONDITION = "unsupported_boundary_condition"
    PURE_NEUMANN_WITHOUT_GAUGE = "pure_neumann_without_gauge"
    HIGH_ENSEMBLE_DISAGREEMENT = "high_ensemble_disagreement"
    HIGH_PEAK_DISAGREEMENT = "high_peak_disagreement"
    HIGH_INPUT_CONTRAST = "high_input_contrast"
    HIGH_HEAT_SOURCE = "high_heat_source"
    EXCESSIVE_PDE_RESIDUAL = "excessive_pde_residual"
    EXCESSIVE_BOUNDARY_FLUX_IMBALANCE = "excessive_boundary_flux_imbalance"
    SURROGATE_OUTPUT_NON_FINITE = "surrogate_output_non_finite"
    SOLVER_DIVERGENCE = "solver_divergence"
    SOLVER_TIMEOUT = "solver_timeout"
    SOLVER_EXCEPTION = "solver_exception"
    FALLBACK_DISABLED = "fallback_disabled"


@dataclass(frozen=True)
class RejectionRecord:
    """Structured audit record describing the cause of surrogate rejection or failure."""

    code: RejectionCode
    message: str
    triggering_metric: str | None = None
    observed_value: float | None = None
    threshold_value: float | None = None


@dataclass(frozen=True)
class ConformalIntervalOutput:
    """Pointwise and simultaneous prediction intervals in physical Kelvin [K].

    Attributes:
        pointwise_lower_kelvin: Cell-level lower bound (ny, nx) [K].
        pointwise_upper_kelvin: Cell-level upper bound (ny, nx) [K].
        simultaneous_lower_kelvin: Whole-field simultaneous lower bound (ny, nx) [K].
        simultaneous_upper_kelvin: Whole-field simultaneous upper bound (ny, nx) [K].
        pointwise_mean_width_kelvin: Average pointwise interval width [K].
        simultaneous_mean_width_kelvin: Average simultaneous interval width [K].
        nominal_confidence: Specified confidence level 1 - alpha (e.g. 0.95).
        pointwise_q_hat: Empirical nonconformity quantile multiplier for pointwise coverage.
        simultaneous_q_hat: Empirical nonconformity quantile multiplier for simultaneous coverage.
    """

    pointwise_lower_kelvin: np.ndarray
    pointwise_upper_kelvin: np.ndarray
    simultaneous_lower_kelvin: np.ndarray
    simultaneous_upper_kelvin: np.ndarray
    pointwise_mean_width_kelvin: float
    simultaneous_mean_width_kelvin: float
    nominal_confidence: float
    pointwise_q_hat: float
    simultaneous_q_hat: float


@dataclass(frozen=True)
class PhysicalDiagnosticsOutput:
    """Discrete physical conservation and algebraic residual diagnostics.

    Distinguishes local cell-wise PDE residual norms from global boundary heat balance,
    and explicitly tracks signed net energy imbalance separately from absolute imbalance.

    Attributes:
        pde_residual_l2_w: Euclidean L2 norm of discrete cell residual ||A * T - b||_2 [W].
        pde_residual_relative: Normalized algebraic residual ||A * T - b||_2 / ||b||_2.
        pde_residual_linf_w: Maximum absolute single-cell residual max |A * T - b| [W].
        q_generated_total_w: Total volumetric heat generation rate in domain [W].
        q_boundary_out_total_w: Total conductive heat flow rate leaving domain [W].
        signed_net_imbalance_w: Signed conservation error Q_boundary_out - Q_generated [W].
            Positive denotes net heat loss; negative denotes net heat accumulation.
        absolute_flux_imbalance_w: Absolute conservation error |Q_boundary_out - Q_generated| [W].
        relative_flux_imbalance: Dimensionless relative error |imbalance| / max(|Q_gen|, |Q_out|).
        boundary_heat_flow_out_w: Outward heat flow per boundary edge (left, right, bottom, top) [W].
        boundary_flux_densities_w_m2: Outward heat flux density arrays per edge [W/m^2].
    """

    pde_residual_l2_w: float
    pde_residual_relative: float
    pde_residual_linf_w: float
    q_generated_total_w: float
    q_boundary_out_total_w: float
    signed_net_imbalance_w: float
    absolute_flux_imbalance_w: float
    relative_flux_imbalance: float
    boundary_heat_flow_out_w: dict[str, float]
    boundary_flux_densities_w_m2: dict[str, np.ndarray]


@dataclass(frozen=True)
class ExecutionLatency:
    """Detailed wall-clock execution timings in milliseconds."""

    input_validation_ms: float
    surrogate_inference_ms: float
    diagnostics_evaluation_ms: float
    solver_fallback_ms: float
    total_pipeline_ms: float


@dataclass(frozen=True)
class InferenceProvenance:
    """Traceability metadata for models, calibration datasets, and solvers."""

    ensemble_member_checkpoints: list[str]
    ensemble_member_sha256: list[str]
    normalizer_method: str
    temperature_mean_kelvin: float
    temperature_std_kelvin: float
    calibration_dataset_id: str
    calibration_confidence_level: float
    solver_name: str
    solver_tolerance: float


@dataclass(frozen=True)
class InferenceOptions:
    """Configuration options governing screening thresholds, timeouts, and fallback.

    Provisional screening thresholds default to Milestone 05B independent calibration values.
    PDE residual and boundary flux imbalance acceptance thresholds are disabled (None) by
    default, awaiting empirical calibration against maximum temperature error tolerances.
    """

    enable_fallback: bool = True
    provisional_mean_disagreement_threshold_k: float = 0.9978  # Calibrated 95th pct
    provisional_max_disagreement_threshold_k: float = 4.4526  # Calibrated 95th pct
    input_contrast_threshold: float = 7.16  # Max training contrast
    input_q_max_threshold: float = 912.8  # Max training heat source [W/m^3]
    pde_residual_threshold_w: float | None = None  # Configurable, disabled by default
    relative_pde_residual_threshold: float | None = None  # Configurable, disabled by default
    boundary_flux_imbalance_threshold_w: float | None = None  # Configurable, disabled by default
    relative_flux_imbalance_threshold: float | None = None  # Configurable, disabled by default
    solver_timeout_seconds: float = 10.0
    solver_tolerance: float = 1e-10
    fallback_isolation: str = "thread"  # "thread" or "process"
    conformal_confidence_level: float = 0.95
    compute_diagnostics_on_fallback: bool = True


@dataclass(frozen=True)
class InferenceRequest:
    """Specification of a physical steady-state heat conduction problem for inference."""

    grid: Grid2D
    conductivity: float | int | np.ndarray | Callable[[np.ndarray, np.ndarray], Any]
    source: float | int | np.ndarray | Callable[[np.ndarray, np.ndarray], Any]
    boundary_conditions: BoundaryConditions
    gauge_pin: tuple[int, int] | None = None
    gauge_temperature: float | None = None
    options: InferenceOptions = field(default_factory=InferenceOptions)


@dataclass(frozen=True)
class HybridInferenceResult:
    """Unified result container returned by the Hybrid Digital Twin.

    Adheres strictly to the fail-closed contract:
    - If status == ACCEPTED: temperature contains verified surrogate prediction.
    - If status == FALLBACK_SUCCEEDED: temperature contains numerical reference solution.
    - If status in (INVALID_INPUT, SOLVER_FAILED): temperature is strictly None.
    """

    temperature: np.ndarray | None  # 2D temperature field (ny, nx) [K], or None if failed
    status: InferenceStatus
    execution_path: str  # 'surrogate_fast_path', 'solver_fallback_path', 'aborted'
    rejection: RejectionRecord
    ensemble_disagreement_kelvin: np.ndarray | None  # Per-cell std dev s_T (ny, nx) [K]
    mean_disagreement_kelvin: float | None
    max_disagreement_kelvin: float | None
    conformal_intervals: ConformalIntervalOutput | None
    diagnostics: PhysicalDiagnosticsOutput | None
    latency: ExecutionLatency
    provenance: InferenceProvenance

    @property
    def is_accepted(self) -> bool:
        """True if the surrogate prediction passed all screening checks."""
        return self.status == InferenceStatus.ACCEPTED

    @property
    def is_fallback(self) -> bool:
        """True if the numerical reference solver was invoked successfully."""
        return self.status == InferenceStatus.FALLBACK_SUCCEEDED

    @property
    def is_failure(self) -> bool:
        """True if the request aborted due to invalid input or solver failure."""
        return self.status in (InferenceStatus.INVALID_INPUT, InferenceStatus.SOLVER_FAILED)
