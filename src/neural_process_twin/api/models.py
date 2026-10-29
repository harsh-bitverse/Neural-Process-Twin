"""Pydantic data models for the Neural Process Twin scientific API."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


class GridModel(BaseModel):
    """Specification of the computational 2D domain and cell resolution."""

    x_min: float = Field(0.0, description="Minimum x-coordinate in meters [m].")
    x_max: float = Field(1.0, description="Maximum x-coordinate in meters [m].")
    y_min: float = Field(0.0, description="Minimum y-coordinate in meters [m].")
    y_max: float = Field(1.0, description="Maximum y-coordinate in meters [m].")
    nx: int = Field(32, ge=4, le=256, description="Number of grid cells along x.")
    ny: int = Field(32, ge=4, le=256, description="Number of grid cells along y.")
    thickness: float = Field(1.0, gt=0.0, description="Slab thickness in meters [m].")

    @model_validator(mode="after")
    def validate_bounds(self) -> GridModel:
        if self.x_max <= self.x_min:
            raise ValueError(
                f"x_max ({self.x_max}) must be strictly greater than x_min ({self.x_min})."
            )
        if self.y_max <= self.y_min:
            raise ValueError(
                f"y_max ({self.y_max}) must be strictly greater than y_min ({self.y_min})."
            )
        return self


class BoundaryConditionItem(BaseModel):
    """Specification of boundary condition on a single domain side."""

    side: Literal["left", "right", "bottom", "top"]
    type: Literal["dirichlet", "neumann"]
    value: float = Field(
        ..., description="Prescribed temperature [K] (Dirichlet) or outward flux [W/m^2] (Neumann)."
    )

    @field_validator("value")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        if not np.isfinite(v):
            raise ValueError("Boundary condition value must be a finite numerical scalar.")
        return v


class BoundaryConditionsModel(BaseModel):
    """Boundary conditions on all four physical edges of the 2D slab."""

    left: BoundaryConditionItem
    right: BoundaryConditionItem
    bottom: BoundaryConditionItem
    top: BoundaryConditionItem

    @model_validator(mode="after")
    def validate_well_posedness(self) -> BoundaryConditionsModel:
        # Check Dirichlet temperatures are positive (> 0 K)
        for side_name in ("left", "right", "bottom", "top"):
            bc: BoundaryConditionItem = getattr(self, side_name)
            if bc.type == "dirichlet" and bc.value <= 0.0:
                raise ValueError(
                    f"Dirichlet temperature on {side_name} boundary must be strictly positive Kelvin "
                    f"(observed {bc.value} K)."
                )

        # Check pure-Neumann solvability
        all_neumann = all(
            getattr(self, s).type == "neumann" for s in ("left", "right", "bottom", "top")
        )
        if all_neumann:
            # We reject pure Neumann without explicit temperature pin
            raise ValueError(
                "Pure-Neumann boundary conditions on all four boundaries are singular without a gauge reference."
            )
        return self


class FieldPatternModel(BaseModel):
    """Specification for generating or inputting thermal conductivity fields k(x, y)."""

    pattern_type: Literal["uniform", "inclusion", "gradient", "custom"] = "uniform"
    uniform_value: float = Field(2.5, gt=0.0, description="Uniform conductivity [W/(m K)].")
    inclusion_k_background: float = Field(1.0, gt=0.0, description="Matrix conductivity [W/(m K)].")
    inclusion_k_feature: float = Field(5.0, gt=0.0, description="Inclusion conductivity [W/(m K)].")
    inclusion_center_x: float = Field(0.5, description="Inclusion center x [m].")
    inclusion_center_y: float = Field(0.5, description="Inclusion center y [m].")
    inclusion_radius: float = Field(0.2, gt=0.0, description="Inclusion radius [m].")
    custom_values: list[list[float]] | None = Field(
        None, description="2D matrix (ny, nx) of conductivity values."
    )


class HeatSourcePatternModel(BaseModel):
    """Specification for generating or inputting volumetric heat source fields q(x, y)."""

    pattern_type: Literal["uniform", "gaussian_spot", "custom"] = "uniform"
    uniform_value: float = Field(50.0, description="Uniform volumetric heat source [W/m^3].")
    spot_peak_value: float = Field(200.0, description="Peak heat source value [W/m^3].")
    spot_center_x: float = Field(0.5, description="Spot center x [m].")
    spot_center_y: float = Field(0.5, description="Spot center y [m].")
    spot_sigma: float = Field(0.15, gt=0.0, description="Spot Gaussian standard deviation [m].")
    custom_values: list[list[float]] | None = Field(
        None, description="2D matrix (ny, nx) of source values."
    )


class OptionsModel(BaseModel):
    """Execution and screening options for the inference engine."""

    enable_fallback: bool = Field(
        True, description="Enable automatic conservative FVM fallback when rejected."
    )
    fallback_isolation: Literal["thread", "process"] = Field(
        "thread", description="Concurrency isolation for solver fallback."
    )
    provisional_mean_disagreement_threshold_k: float = Field(
        0.9978, gt=0.0, description="Mean ensemble disagreement threshold [K]."
    )
    provisional_max_disagreement_threshold_k: float = Field(
        4.4526, gt=0.0, description="Peak cell disagreement threshold [K]."
    )
    input_contrast_threshold: float = Field(
        7.16, gt=1.0, description="Maximum training domain contrast ratio."
    )
    input_q_max_threshold: float = Field(
        912.8, gt=0.0, description="Maximum training volumetric heat source [W/m^3]."
    )
    solver_timeout_seconds: float = Field(
        10.0, gt=0.1, le=60.0, description="Fallback solver timeout limit in seconds."
    )


class InferenceRequestModel(BaseModel):
    """Complete problem specification request for hybrid or FVM inference."""

    grid: GridModel = Field(default_factory=GridModel)
    conductivity: FieldPatternModel = Field(default_factory=FieldPatternModel)
    source: HeatSourcePatternModel = Field(default_factory=HeatSourcePatternModel)
    boundary_conditions: BoundaryConditionsModel
    options: OptionsModel = Field(default_factory=OptionsModel)


# =============================================================================
# API Response Models
# =============================================================================


class HealthResponse(BaseModel):
    """API health and environment status."""

    status: str
    version: str
    device: str
    torch_version: str
    ensemble_loaded: bool
    ensemble_member_count: int


class CapabilitiesResponse(BaseModel):
    """Disclosed model architectures, training envelopes, and performance expectations."""

    surrogate_supported_resolution: tuple[int, int]
    fvm_supported_resolution_range: str
    ensemble_members: list[dict[str, str]]
    training_domain_limits: dict[str, Any]
    disagreement_screening_thresholds: dict[str, float]
    physical_units: dict[str, str]
    performance_disclosure: str


class ValidationResponse(BaseModel):
    """Pre-inference validation assessment of a physical problem specification."""

    is_valid: bool
    surrogate_compatible: bool
    k_min: float
    k_max: float
    k_contrast: float
    q_max: float
    within_contrast_envelope: bool
    within_source_envelope: bool
    errors: list[str]
    warnings: list[str]


class ConformalIntervalsResponse(BaseModel):
    """Pointwise and simultaneous split-conformal prediction interval summaries."""

    nominal_confidence: float
    pointwise_q_hat: float
    simultaneous_q_hat: float
    pointwise_mean_width_k: float
    simultaneous_mean_width_k: float
    calibration_dataset_id: str
    disclaimer: str


class PhysicalDiagnosticsResponse(BaseModel):
    """Independent discrete PDE residual and global energy conservation metrics."""

    pde_residual_l2_w: float
    pde_residual_relative: float
    pde_residual_linf_w: float
    q_generated_total_w: float
    q_boundary_out_total_w: float
    signed_net_imbalance_w: float
    absolute_flux_imbalance_w: float
    relative_flux_imbalance: float
    boundary_heat_flow_out_w: dict[str, float]


class ExecutionLatencyResponse(BaseModel):
    """Detailed wall-clock execution timings in milliseconds."""

    input_validation_ms: float
    surrogate_inference_ms: float
    diagnostics_evaluation_ms: float
    solver_fallback_ms: float
    total_pipeline_ms: float


class RejectionRecordResponse(BaseModel):
    """Audit record explaining surrogate rejection and fallback routing."""

    code: str
    message: str
    triggering_metric: str | None
    observed_value: float | None
    threshold_value: float | None


class InferenceResponse(BaseModel):
    """Unified result schema for surrogate, fallback, and direct FVM solutions."""

    status: str  # 'accepted', 'fallback_succeeded', 'fallback_required', 'invalid_input', 'solver_failed'
    execution_path: (
        str  # 'surrogate_fast_path', 'solver_fallback_path', 'direct_fvm_solve', 'aborted'
    )
    temperature: list[list[float]] | None
    temperature_min_k: float | None
    temperature_max_k: float | None
    rejection: RejectionRecordResponse
    ensemble_disagreement_k: list[list[float]] | None
    mean_disagreement_k: float | None
    max_disagreement_k: float | None
    conformal_intervals: ConformalIntervalsResponse | None
    diagnostics: PhysicalDiagnosticsResponse | None
    latency: ExecutionLatencyResponse
    provenance: dict[str, Any]
    conductivity_field: list[list[float]]
    source_field: list[list[float]]
    x_coords: list[float]
    y_coords: list[float]


class ComparisonResponse(BaseModel):
    """Side-by-side research comparison between the neural surrogate and reference FVM."""

    problem_summary: dict[str, Any]
    hybrid_inference: InferenceResponse
    fvm_reference: InferenceResponse
    temperature_difference_field: list[list[float]] | None
    rmse_k: float | None
    relative_l2_error_pct: float | None
    max_absolute_error_k: float | None
    surrogate_latency_ms: float
    fvm_latency_ms: float
    fvm_speedup_factor: float
    surrogate_acceptance_status: str
    evaluation_notice: str


# =============================================================================
# Milestone 09: Dual-Mode Platform Integration Models
# =============================================================================


class SimulationRequestModel(BaseModel):
    """Unified physical simulation request supporting both Verified Numerical and Experimental Neural modes."""

    mode: Literal[
        "numerical_fvm",
        "experimental_neural",
        "experimental_neural_corrected",
        "legacy_hybrid",
    ] = Field(
        "numerical_fvm",
        description=(
            "Operating mode: 'numerical_fvm' (default trusted numerical reference), "
            "'experimental_neural' (fast A3 RefinedFNO surrogate), "
            "'experimental_neural_corrected' (A3 + Selective Boundary Jacobi), or "
            "'legacy_hybrid' (32x32 ensemble with fallback)."
        ),
    )
    grid: GridModel = Field(default_factory=GridModel)
    conductivity: FieldPatternModel = Field(default_factory=FieldPatternModel)
    source: HeatSourcePatternModel = Field(default_factory=HeatSourcePatternModel)
    boundary_conditions: BoundaryConditionsModel
    options: OptionsModel = Field(default_factory=OptionsModel)


class NumericalSolverInfo(BaseModel):
    """Detailed numerical solver provenance, convergence, and discretization metadata."""

    method: str = "Cell-Centered Finite Volume Method (FVM) with Harmonic Interface Conductivities"
    linear_solver: str = "Direct Sparse LU Factorization (SuperLU via SciPy)"
    is_converged: bool = True
    algebraic_residual_norm: float
    discretization_order: str = "Second-Order Spatial Truncation Error O(h^2)"
    notice: str = (
        "Numerical convergence denotes exact solution of the discrete linear system A*T=b. "
        "It is NOT mathematically exact in the continuous sense due to finite grid cell discretization."
    )


class ExperimentalDiagnostics(BaseModel):
    """Physical diagnostics and empirical quality indicators for experimental neural surrogates."""

    pde_residual_l2_w: float
    pde_residual_linf_w: float
    pde_residual_relative: float
    pointwise_jacobi_residual_max_k: float
    indicator_taxonomy: str = (
        "Pointwise Jacobi residual is a localized numerical indicator in Kelvin. "
        "It is NOT a mathematically certified upper bound on true MaxAE."
    )
    safety_disclosure: str = (
        "EXPERIMENTAL SURROGATE: Not certified to satisfy MaxAE <= 10 K. "
        "Audited failure rate is 36% to 80% on fine grids. Never deploy for safety-critical heat balance."
    )


class SimulationResponse(BaseModel):
    """Unified dual-mode simulation response cleanly separating verified numerical from experimental outputs."""

    mode: str
    status: str
    is_verified_numerical: bool
    temperature: list[list[float]] | None
    temperature_min_k: float | None
    temperature_max_k: float | None
    grid_resolution: tuple[int, int]
    boundary_condition_summary: dict[str, str]
    diagnostics: PhysicalDiagnosticsResponse | None
    numerical_solver_info: NumericalSolverInfo | None
    experimental_diagnostics: ExperimentalDiagnostics | None
    warnings: list[str]
    disclaimers: list[str]
    latency_ms: dict[str, float]
    provenance: dict[str, Any]
    conductivity_field: list[list[float]]
    source_field: list[list[float]]
    x_coords: list[float]
    y_coords: list[float]


class DualModeComparisonResponse(BaseModel):
    """Side-by-side scientific comparison between Verified Numerical FVM and Experimental Neural Surrogate."""

    problem_summary: dict[str, Any]
    numerical_reference: SimulationResponse
    experimental_surrogate: SimulationResponse
    temperature_difference_field: list[list[float]] | None
    rmse_k: float | None
    relative_l2_error_pct: float | None
    max_absolute_error_k: float | None
    fvm_latency_ms: float
    surrogate_latency_ms: float
    speedup_factor: float
    evaluation_notice: str
