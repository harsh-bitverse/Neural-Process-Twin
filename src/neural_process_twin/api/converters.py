"""Conversion utilities between Pydantic API schemas and internal physics classes."""

from __future__ import annotations

from typing import Any

import numpy as np

from neural_process_twin.api.models import (
    BoundaryConditionsModel,
    ConformalIntervalsResponse,
    ExecutionLatencyResponse,
    FieldPatternModel,
    GridModel,
    HeatSourcePatternModel,
    InferenceRequestModel,
    InferenceResponse,
    OptionsModel,
    PhysicalDiagnosticsResponse,
    RejectionRecordResponse,
)
from neural_process_twin.inference.contracts import (
    HybridInferenceResult,
    InferenceOptions,
    InferenceRequest,
    InferenceStatus,
    RejectionCode,
)
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import SolverResult


def build_domain_grid(model: GridModel) -> Grid2D:
    """Instantiate domain Grid2D from request model."""
    return Grid2D(
        x_min=model.x_min,
        x_max=model.x_max,
        y_min=model.y_min,
        y_max=model.y_max,
        nx=model.nx,
        ny=model.ny,
        thickness=model.thickness,
    )


def build_conductivity_array(grid: Grid2D, model: FieldPatternModel) -> np.ndarray:
    """Synthesize or unpack 2D thermal conductivity field k(x, y) [W/(m K)]."""
    if model.pattern_type == "uniform":
        return np.full((grid.ny, grid.nx), model.uniform_value, dtype=np.float64)

    if model.pattern_type == "inclusion":
        x_mesh, y_mesh = np.meshgrid(grid.x_centers, grid.y_centers)
        dist2 = (x_mesh - model.inclusion_center_x) ** 2 + (y_mesh - model.inclusion_center_y) ** 2
        mask = dist2 <= (model.inclusion_radius**2)
        return np.where(mask, model.inclusion_k_feature, model.inclusion_k_background).astype(
            np.float64
        )

    if model.pattern_type == "gradient":
        x_mesh, _ = np.meshgrid(grid.x_centers, grid.y_centers)
        span = max(grid.x_max - grid.x_min, 1e-12)
        norm_x = (x_mesh - grid.x_min) / span
        return (
            model.inclusion_k_background
            + norm_x * (model.inclusion_k_feature - model.inclusion_k_background)
        ).astype(np.float64)

    if model.pattern_type == "custom":
        if model.custom_values is None:
            raise ValueError("custom_values cannot be None when pattern_type is 'custom'.")
        arr = np.array(model.custom_values, dtype=np.float64)
        if arr.shape != (grid.ny, grid.nx):
            raise ValueError(
                f"Custom conductivity array shape {arr.shape} does not match grid ({grid.ny}, {grid.nx})."
            )
        return arr

    raise ValueError(f"Unsupported conductivity pattern type: '{model.pattern_type}'.")


def build_source_array(grid: Grid2D, model: HeatSourcePatternModel) -> np.ndarray:
    """Synthesize or unpack 2D volumetric heat source field q(x, y) [W/m^3]."""
    if model.pattern_type == "uniform":
        return np.full((grid.ny, grid.nx), model.uniform_value, dtype=np.float64)

    if model.pattern_type == "gaussian_spot":
        x_mesh, y_mesh = np.meshgrid(grid.x_centers, grid.y_centers)
        dist2 = (x_mesh - model.spot_center_x) ** 2 + (y_mesh - model.spot_center_y) ** 2
        sigma2 = max(2.0 * (model.spot_sigma**2), 1e-12)
        return (model.spot_peak_value * np.exp(-dist2 / sigma2)).astype(np.float64)

    if model.pattern_type == "custom":
        if model.custom_values is None:
            raise ValueError("custom_values cannot be None when pattern_type is 'custom'.")
        arr = np.array(model.custom_values, dtype=np.float64)
        if arr.shape != (grid.ny, grid.nx):
            raise ValueError(
                f"Custom heat source array shape {arr.shape} does not match grid ({grid.ny}, {grid.nx})."
            )
        return arr

    raise ValueError(f"Unsupported heat source pattern type: '{model.pattern_type}'.")


def build_boundary_conditions(model: BoundaryConditionsModel) -> BoundaryConditions:
    """Construct physical BoundaryConditions domain object."""
    return BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType(model.left.type), model.left.value),
        right=BoundaryCondition(
            BoundarySide.RIGHT, BoundaryType(model.right.type), model.right.value
        ),
        bottom=BoundaryCondition(
            BoundarySide.BOTTOM, BoundaryType(model.bottom.type), model.bottom.value
        ),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType(model.top.type), model.top.value),
    )


def build_inference_options(model: OptionsModel) -> InferenceOptions:
    """Construct InferenceOptions configuration."""
    return InferenceOptions(
        enable_fallback=model.enable_fallback,
        fallback_isolation=model.fallback_isolation,
        provisional_mean_disagreement_threshold_k=model.provisional_mean_disagreement_threshold_k,
        provisional_max_disagreement_threshold_k=model.provisional_max_disagreement_threshold_k,
        input_contrast_threshold=model.input_contrast_threshold,
        input_q_max_threshold=model.input_q_max_threshold,
        solver_timeout_seconds=model.solver_timeout_seconds,
    )


def build_inference_request(
    req_model: InferenceRequestModel,
) -> tuple[InferenceRequest, np.ndarray, np.ndarray]:
    """Compile InferenceRequestModel into internal domain request and evaluated fields."""
    grid = build_domain_grid(req_model.grid)
    k_field = build_conductivity_array(grid, req_model.conductivity)
    q_field = build_source_array(grid, req_model.source)
    bcs = build_boundary_conditions(req_model.boundary_conditions)
    options = build_inference_options(req_model.options)

    request = InferenceRequest(
        grid=grid,
        conductivity=k_field,
        source=q_field,
        boundary_conditions=bcs,
        options=options,
    )
    return request, k_field, q_field


def format_hybrid_result_to_response(
    result: HybridInferenceResult,
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
) -> InferenceResponse:
    """Format HybridInferenceResult into public API InferenceResponse."""
    t_min = float(np.min(result.temperature)) if result.temperature is not None else None
    t_max = float(np.max(result.temperature)) if result.temperature is not None else None

    conformal_resp = None
    if result.conformal_intervals is not None:
        c = result.conformal_intervals
        conformal_resp = ConformalIntervalsResponse(
            nominal_confidence=c.nominal_confidence,
            pointwise_q_hat=c.pointwise_q_hat,
            simultaneous_q_hat=c.simultaneous_q_hat,
            pointwise_mean_width_k=c.pointwise_mean_width_kelvin,
            simultaneous_mean_width_k=c.simultaneous_mean_width_kelvin,
            calibration_dataset_id=result.provenance.calibration_dataset_id,
            disclaimer=(
                "Conformal intervals guarantee 95% marginal coverage under the independent IID "
                "calibration distribution. Coverage guarantees are not valid under out-of-distribution shifts."
            ),
        )

    diag_resp = None
    if result.diagnostics is not None:
        d = result.diagnostics
        diag_resp = PhysicalDiagnosticsResponse(
            pde_residual_l2_w=d.pde_residual_l2_w,
            pde_residual_relative=d.pde_residual_relative,
            pde_residual_linf_w=d.pde_residual_linf_w,
            q_generated_total_w=d.q_generated_total_w,
            q_boundary_out_total_w=d.q_boundary_out_total_w,
            signed_net_imbalance_w=d.signed_net_imbalance_w,
            absolute_flux_imbalance_w=d.absolute_flux_imbalance_w,
            relative_flux_imbalance=d.relative_flux_imbalance,
            boundary_heat_flow_out_w=d.boundary_heat_flow_out_w,
        )

    latency_resp = ExecutionLatencyResponse(
        input_validation_ms=result.latency.input_validation_ms,
        surrogate_inference_ms=result.latency.surrogate_inference_ms,
        diagnostics_evaluation_ms=result.latency.diagnostics_evaluation_ms,
        solver_fallback_ms=result.latency.solver_fallback_ms,
        total_pipeline_ms=result.latency.total_pipeline_ms,
    )

    rejection_resp = RejectionRecordResponse(
        code=result.rejection.code.value,
        message=result.rejection.message,
        triggering_metric=result.rejection.triggering_metric,
        observed_value=result.rejection.observed_value,
        threshold_value=result.rejection.threshold_value,
    )

    provenance_dict: dict[str, Any] = {
        "ensemble_checkpoints": result.provenance.ensemble_member_checkpoints,
        "ensemble_sha256": result.provenance.ensemble_member_sha256,
        "calibration_dataset_id": result.provenance.calibration_dataset_id,
        "solver_name": result.provenance.solver_name,
        "solver_tolerance": result.provenance.solver_tolerance,
    }

    return InferenceResponse(
        status=result.status.value,
        execution_path=result.execution_path,
        temperature=result.temperature.tolist() if result.temperature is not None else None,
        temperature_min_k=t_min,
        temperature_max_k=t_max,
        rejection=rejection_resp,
        ensemble_disagreement_k=result.ensemble_disagreement_kelvin.tolist()
        if result.ensemble_disagreement_kelvin is not None
        else None,
        mean_disagreement_k=result.mean_disagreement_kelvin,
        max_disagreement_k=result.max_disagreement_kelvin,
        conformal_intervals=conformal_resp,
        diagnostics=diag_resp,
        latency=latency_resp,
        provenance=provenance_dict,
        conductivity_field=k_field.tolist(),
        source_field=q_field.tolist(),
        x_coords=grid.x_centers.tolist(),
        y_coords=grid.y_centers.tolist(),
    )


def format_solver_result_to_response(
    solver_res: SolverResult,
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    latency_ms: float,
) -> InferenceResponse:
    """Format direct HeatConductionSolver2D result into public API InferenceResponse."""
    t_min = float(np.min(solver_res.temperature))
    t_max = float(np.max(solver_res.temperature))
    linf = float(
        np.max(
            np.abs(
                solver_res.linear_system.A.dot(solver_res.temperature.ravel())
                - solver_res.linear_system.b
            )
        )
    )

    signed_imb = (
        solver_res.diagnostics.energy_balance.q_boundary_out_total
        - solver_res.diagnostics.energy_balance.q_generated_total
    )
    diag_resp = PhysicalDiagnosticsResponse(
        pde_residual_l2_w=solver_res.diagnostics.linear_residual_l2,
        pde_residual_relative=solver_res.diagnostics.relative_residual,
        pde_residual_linf_w=linf,
        q_generated_total_w=solver_res.diagnostics.energy_balance.q_generated_total,
        q_boundary_out_total_w=solver_res.diagnostics.energy_balance.q_boundary_out_total,
        signed_net_imbalance_w=signed_imb,
        absolute_flux_imbalance_w=solver_res.diagnostics.energy_balance.flux_imbalance,
        relative_flux_imbalance=solver_res.diagnostics.energy_balance.relative_imbalance,
        boundary_heat_flow_out_w=solver_res.diagnostics.energy_balance.boundary_heat_flow_out,
    )

    latency_resp = ExecutionLatencyResponse(
        input_validation_ms=0.0,
        surrogate_inference_ms=0.0,
        diagnostics_evaluation_ms=0.0,
        solver_fallback_ms=latency_ms,
        total_pipeline_ms=latency_ms,
    )

    rejection_resp = RejectionRecordResponse(
        code=RejectionCode.NONE.value,
        message="Direct numerical finite-volume reference solve succeeded.",
        triggering_metric=None,
        observed_value=None,
        threshold_value=None,
    )

    return InferenceResponse(
        status=InferenceStatus.FALLBACK_SUCCEEDED.value,
        execution_path="direct_fvm_solve",
        temperature=solver_res.temperature.tolist(),
        temperature_min_k=t_min,
        temperature_max_k=t_max,
        rejection=rejection_resp,
        ensemble_disagreement_k=None,  # Never fabricate uncertainty for deterministic FVM
        mean_disagreement_k=None,
        max_disagreement_k=None,
        conformal_intervals=None,
        diagnostics=diag_resp,
        latency=latency_resp,
        provenance={
            "solver_name": "HeatConductionSolver2D",
            "solver_method": "Finite Volume Method (FVM)",
            "sparse_linear_solver": "scipy.sparse.linalg.spsolve (SuperLU)",
            "matrix_condition_converged": solver_res.diagnostics.is_converged,
        },
        conductivity_field=k_field.tolist(),
        source_field=q_field.tolist(),
        x_coords=grid.x_centers.tolist(),
        y_coords=grid.y_centers.tolist(),
    )
