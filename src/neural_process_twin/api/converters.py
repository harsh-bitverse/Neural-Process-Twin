"""Conversion utilities between Pydantic API schemas and internal physics classes."""

from __future__ import annotations

from typing import Any

import numpy as np

from neural_process_twin.api.models import (
    BoundaryConditionsModel,
    ConformalIntervalsResponse,
    ExecutionLatencyResponse,
    ExperimentalDiagnostics,
    FieldPatternModel,
    GridModel,
    HeatSourcePatternModel,
    InferenceRequestModel,
    InferenceResponse,
    NumericalSolverInfo,
    OptionsModel,
    PhysicalDiagnosticsResponse,
    RejectionRecordResponse,
    SimulationResponse,
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


def format_fvm_simulation_response(
    solver_res: SolverResult,
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
    latency_ms: float,
) -> SimulationResponse:
    """Format reference FVM solver result into public SimulationResponse."""
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

    solver_info = NumericalSolverInfo(
        method="Cell-Centered Finite Volume Method (FVM) with Harmonic Interface Conductivities",
        linear_solver="Direct Sparse LU Factorization (SuperLU via SciPy)",
        is_converged=solver_res.diagnostics.is_converged,
        algebraic_residual_norm=solver_res.diagnostics.linear_residual_l2,
        discretization_order="Second-Order Spatial Truncation Error O(h^2)",
        notice=(
            "Numerical convergence denotes exact solution of the discrete linear system A*T=b. "
            "It is NOT mathematically exact in the continuous sense due to finite grid cell discretization."
        ),
    )

    bcs_summary = {
        "left": f"{bcs.left.bc_type.value}: {bcs.left.value}",
        "right": f"{bcs.right.bc_type.value}: {bcs.right.value}",
        "bottom": f"{bcs.bottom.bc_type.value}: {bcs.bottom.value}",
        "top": f"{bcs.top.bc_type.value}: {bcs.top.value}",
    }

    return SimulationResponse(
        mode="numerical_fvm",
        status="numerical_reference_computed",
        is_verified_numerical=True,
        temperature=solver_res.temperature.tolist(),
        temperature_min_k=t_min,
        temperature_max_k=t_max,
        grid_resolution=(grid.nx, grid.ny),
        boundary_condition_summary=bcs_summary,
        diagnostics=diag_resp,
        numerical_solver_info=solver_info,
        experimental_diagnostics=None,
        warnings=[],
        disclaimers=[
            "Verified Numerical Mode: Conservative reference solution. Governed by O(h^2) discrete FVM truncation error."
        ],
        latency_ms={"solver_ms": latency_ms, "total_ms": latency_ms},
        provenance={
            "mode": "numerical_fvm",
            "solver": "HeatConductionSolver2D",
            "discretization": "FVM (Harmonic Averaging)",
            "tolerance": 1e-10,
        },
        conductivity_field=k_field.tolist(),
        source_field=q_field.tolist(),
        x_coords=grid.x_centers.tolist(),
        y_coords=grid.y_centers.tolist(),
    )


def format_experimental_simulation_response(
    surr_res: Any,
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
    mode_name: str = "experimental_neural",
) -> SimulationResponse:
    """Format ExperimentalSurrogateResult into public SimulationResponse."""
    t_min = float(np.min(surr_res.temperature))
    t_max = float(np.max(surr_res.temperature))

    diag_resp = None
    if surr_res.heat_balance is not None:
        hb = surr_res.heat_balance
        signed_imb = hb.q_boundary_out_total - hb.q_generated_total
        diag_resp = PhysicalDiagnosticsResponse(
            pde_residual_l2_w=surr_res.pde_residual_l2_w,
            pde_residual_relative=surr_res.pde_residual_relative,
            pde_residual_linf_w=surr_res.pde_residual_linf_w,
            q_generated_total_w=hb.q_generated_total,
            q_boundary_out_total_w=hb.q_boundary_out_total,
            signed_net_imbalance_w=signed_imb,
            absolute_flux_imbalance_w=hb.flux_imbalance,
            relative_flux_imbalance=hb.relative_imbalance,
            boundary_heat_flow_out_w=hb.boundary_heat_flow_out,
        )

    exp_diag = None
    if surr_res.pde_residual_l2_w is not None and surr_res.jacobi_residual_max_k is not None:
        exp_diag = ExperimentalDiagnostics(
            pde_residual_l2_w=surr_res.pde_residual_l2_w,
            pde_residual_linf_w=surr_res.pde_residual_linf_w,
            pde_residual_relative=surr_res.pde_residual_relative,
            pointwise_jacobi_residual_max_k=surr_res.jacobi_residual_max_k,
            indicator_taxonomy=(
                "Pointwise Jacobi residual is a localized numerical indicator in Kelvin. "
                "It is NOT a mathematically certified upper bound on true MaxAE."
            ),
            safety_disclosure=(
                "EXPERIMENTAL SURROGATE: Not certified to satisfy MaxAE <= 10 K. "
                "Audited failure rate is 36% to 80% on fine grids. Never deploy for safety-critical heat balance."
            ),
        )

    bcs_summary = {
        "left": f"{bcs.left.bc_type.value}: {bcs.left.value}",
        "right": f"{bcs.right.bc_type.value}: {bcs.right.value}",
        "bottom": f"{bcs.bottom.bc_type.value}: {bcs.bottom.value}",
        "top": f"{bcs.top.bc_type.value}: {bcs.top.value}",
    }

    return SimulationResponse(
        mode=mode_name,
        status="experimental_prediction_computed",
        is_verified_numerical=False,
        temperature=surr_res.temperature.tolist(),
        temperature_min_k=t_min,
        temperature_max_k=t_max,
        grid_resolution=surr_res.resolution,
        boundary_condition_summary=bcs_summary,
        diagnostics=diag_resp,
        numerical_solver_info=None,
        experimental_diagnostics=exp_diag,
        warnings=surr_res.warnings,
        disclaimers=surr_res.disclaimers,
        latency_ms={
            "inference_ms": surr_res.inference_time_ms,
            "correction_ms": surr_res.correction_time_ms,
            "total_ms": surr_res.total_time_ms,
        },
        provenance=surr_res.provenance,
        conductivity_field=k_field.tolist(),
        source_field=q_field.tolist(),
        x_coords=grid.x_centers.tolist(),
        y_coords=grid.y_centers.tolist(),
    )


def format_legacy_hybrid_simulation_response(
    hybrid_res: HybridInferenceResult,
    grid: Grid2D,
    k_field: np.ndarray,
    q_field: np.ndarray,
    bcs: BoundaryConditions,
) -> SimulationResponse:
    """Format legacy HybridInferenceResult into public SimulationResponse."""
    t_min = float(np.min(hybrid_res.temperature)) if hybrid_res.temperature is not None else None
    t_max = float(np.max(hybrid_res.temperature)) if hybrid_res.temperature is not None else None

    is_verified = (
        hybrid_res.execution_path == "solver_fallback_path"
        or hybrid_res.status == InferenceStatus.FALLBACK_SUCCEEDED
    )

    diag_resp = None
    if hybrid_res.diagnostics is not None:
        d = hybrid_res.diagnostics
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

    bcs_summary = {
        "left": f"{bcs.left.bc_type.value}: {bcs.left.value}",
        "right": f"{bcs.right.bc_type.value}: {bcs.right.value}",
        "bottom": f"{bcs.bottom.bc_type.value}: {bcs.bottom.value}",
        "top": f"{bcs.top.bc_type.value}: {bcs.top.value}",
    }

    warnings = []
    if not is_verified:
        warnings.append(
            "LEGACY HYBRID SURROGATE: Evaluated via 32x32 deep ensemble with disagreement screening."
        )

    return SimulationResponse(
        mode="legacy_hybrid",
        status=hybrid_res.status.value,
        is_verified_numerical=is_verified,
        temperature=hybrid_res.temperature.tolist() if hybrid_res.temperature is not None else None,
        temperature_min_k=t_min,
        temperature_max_k=t_max,
        grid_resolution=(grid.nx, grid.ny),
        boundary_condition_summary=bcs_summary,
        diagnostics=diag_resp,
        numerical_solver_info=None,
        experimental_diagnostics=None,
        warnings=warnings,
        disclaimers=["Legacy hybrid screening route from Milestone 06."],
        latency_ms={
            "surrogate_ms": hybrid_res.latency.surrogate_inference_ms,
            "fallback_ms": hybrid_res.latency.solver_fallback_ms,
            "total_ms": hybrid_res.latency.total_pipeline_ms,
        },
        provenance={"execution_path": hybrid_res.execution_path},
        conductivity_field=k_field.tolist(),
        source_field=q_field.tolist(),
        x_coords=grid.x_centers.tolist(),
        y_coords=grid.y_centers.tolist(),
    )
