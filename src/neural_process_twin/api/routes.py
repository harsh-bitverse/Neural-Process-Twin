"""FastAPI router endpoints for Neural Process Twin hybrid inference and validation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, status

from neural_process_twin.api.converters import (
    build_boundary_conditions,
    build_conductivity_array,
    build_domain_grid,
    build_inference_options,
    build_inference_request,
    build_source_array,
    format_experimental_simulation_response,
    format_fvm_simulation_response,
    format_hybrid_result_to_response,
    format_legacy_hybrid_simulation_response,
    format_solver_result_to_response,
)
from neural_process_twin.api.deployment_config import GOVERNOR, METRICS
from neural_process_twin.api.models import (
    CapabilitiesResponse,
    ComparisonResponse,
    DualModeComparisonResponse,
    HealthResponse,
    InferenceRequestModel,
    InferenceResponse,
    SimulationRequestModel,
    SimulationResponse,
    ValidationResponse,
)
from neural_process_twin.inference.contracts import (
    InferenceRequest,
    InferenceStatus,
)
from neural_process_twin.inference.engine import (
    FROZEN_ENSEMBLE_SHA256,
    HybridInferenceEngine,
)
from neural_process_twin.inference.experimental_surrogate import ExperimentalSurrogateService
from neural_process_twin.inference.validator import InputValidator
from neural_process_twin.solvers.solver import HeatConductionSolver2D

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Inference & Diagnostics"])

# Singleton engine instance, initialized lazily with thread lock protection
_ENGINE_INSTANCE: HybridInferenceEngine | None = None
_VALIDATOR_INSTANCE: InputValidator | None = None
_SOLVER_INSTANCE: HeatConductionSolver2D | None = None
_EXPERIMENTAL_SURROGATE: ExperimentalSurrogateService | None = None
_SINGLETON_LOCK = threading.Lock()


def get_experimental_surrogate() -> ExperimentalSurrogateService:
    """Retrieve or lazily initialize the singleton ExperimentalSurrogateService with thread safety."""
    global _EXPERIMENTAL_SURROGATE
    if _EXPERIMENTAL_SURROGATE is None:
        with _SINGLETON_LOCK:
            if _EXPERIMENTAL_SURROGATE is None:
                logger.info("Initializing ExperimentalSurrogateService (A3 RefinedFNO2D)...")
                _EXPERIMENTAL_SURROGATE = ExperimentalSurrogateService(device="cpu")
    return _EXPERIMENTAL_SURROGATE


def get_engine() -> HybridInferenceEngine:
    """Retrieve or lazily initialize the singleton HybridInferenceEngine with thread safety."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        with _SINGLETON_LOCK:
            if _ENGINE_INSTANCE is None:
                logger.info(
                    "Initializing HybridInferenceEngine from authoritative repository artifacts..."
                )
                _ENGINE_INSTANCE = HybridInferenceEngine.from_artifacts(device="cpu")
    return _ENGINE_INSTANCE


def get_validator() -> InputValidator:
    """Retrieve or lazily initialize the singleton InputValidator with thread safety."""
    global _VALIDATOR_INSTANCE
    if _VALIDATOR_INSTANCE is None:
        with _SINGLETON_LOCK:
            if _VALIDATOR_INSTANCE is None:
                _VALIDATOR_INSTANCE = InputValidator()
    return _VALIDATOR_INSTANCE


def get_solver() -> HeatConductionSolver2D:
    """Retrieve or lazily initialize the reference HeatConductionSolver2D with thread safety."""
    global _SOLVER_INSTANCE
    if _SOLVER_INSTANCE is None:
        with _SINGLETON_LOCK:
            if _SOLVER_INSTANCE is None:
                _SOLVER_INSTANCE = HeatConductionSolver2D(tolerance=1e-10)
    return _SOLVER_INSTANCE


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """System health, runtime environment, and ensemble checkpoint status."""
    engine = get_engine()
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        device="cpu",
        torch_version=torch.__version__,
        ensemble_loaded=engine.ensemble is not None,
        ensemble_member_count=len(engine.ensemble.members) if engine.ensemble else 0,
    )


@router.get("/ready")
def readiness_check() -> dict[str, Any]:
    """Kubernetes / container readiness probe checking model availability and solver initialization."""
    checks: dict[str, str] = {}
    is_ready = True

    try:
        solver = get_solver()
        checks["fvm_solver"] = "ok" if solver is not None else "unavailable"
    except Exception as exc:
        checks["fvm_solver"] = f"error: {exc}"
        is_ready = False

    try:
        surr = get_experimental_surrogate()
        checks["experimental_surrogate"] = (
            "ok" if surr.model is not None and surr.normalizer is not None else "unavailable"
        )
    except Exception as exc:
        checks["experimental_surrogate"] = f"error: {exc}"
        is_ready = False

    try:
        engine = get_engine()
        checks["hybrid_engine"] = "ok" if engine is not None else "unavailable"
    except Exception as exc:
        checks["hybrid_engine"] = f"error: {exc}"
        is_ready = False

    if not is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "not_ready", "checks": checks},
        )
    return {"status": "ready", "checks": checks}


@router.get("/metrics")
def get_operational_metrics() -> dict[str, Any]:
    """Retrieve operational telemetry, queue load, and latency percentiles."""
    return METRICS.get_summary()


@router.get("/capabilities", response_model=CapabilitiesResponse)
def get_capabilities() -> CapabilitiesResponse:
    """Expose supported model architectures, training limits, and performance expectations."""
    members = [{"member_name": name, "sha256": sha} for name, sha in FROZEN_ENSEMBLE_SHA256.items()]
    return CapabilitiesResponse(
        surrogate_supported_resolution=(32, 32),
        fvm_supported_resolution_range="4x4 to 256x256 (arbitrary cell counts and aspect ratios)",
        ensemble_members=members,
        training_domain_limits={
            "max_conductivity_contrast_ratio": 7.16,
            "max_volumetric_heat_source_w_m3": 912.8,
            "training_conductivity_range_w_m_k": [0.63, 4.85],
        },
        disagreement_screening_thresholds={
            "spatial_mean_disagreement_tau_mean_k": 0.9978,
            "peak_cell_disagreement_tau_max_k": 4.4526,
        },
        physical_units={
            "temperature": "Kelvin [K]",
            "thermal_conductivity": "Watts per meter Kelvin [W/(m K)]",
            "volumetric_heat_source": "Watts per cubic meter [W/m^3]",
            "heat_flux_density": "Watts per square meter [W/m^2]",
            "total_heat_flow": "Watts [W]",
            "spatial_coordinates": "Meters [m]",
        },
        performance_disclosure=(
            "SCIENTIFIC PERFORMANCE REALITY & DUAL-MODE PLATFORM ARCHITECTURE: "
            "The single-pass neural surrogate is 11-13x faster than the reference FVM solver on CPU at 32x32. "
            "1. Verified Numerical Mode (FVM) serves as the trusted, "
            "conservative computational reference across arbitrary grid resolutions (4x4 to 256x256). "
            "2. Experimental Neural Mode (A3 RefinedFNO2D / A4 Selective Jacobi) provides fast multi-resolution "
            "predictions across validated grids (32x32, 64x64, 128x128, 256x256), achieving ~1.1x to 7.0x speedups "
            "at high resolutions. Predictions carry explicit warnings as they are NOT certified to satisfy MaxAE <= 10 K."
        ),
    )


@router.post("/validate", response_model=ValidationResponse)
def validate_problem(req: InferenceRequestModel) -> ValidationResponse:
    """Pre-inference validation of a physical problem configuration."""
    try:
        grid = build_domain_grid(req.grid)
        k_field = build_conductivity_array(grid, req.conductivity)
        q_field = build_source_array(grid, req.source)
        bcs = build_boundary_conditions(req.boundary_conditions)
    except Exception as exc:
        return ValidationResponse(
            is_valid=False,
            surrogate_compatible=False,
            k_min=0.0,
            k_max=0.0,
            k_contrast=0.0,
            q_max=0.0,
            within_contrast_envelope=False,
            within_source_envelope=False,
            errors=[str(exc)],
            warnings=[],
        )

    options = build_inference_options(req.options)
    inf_req = InferenceRequest(
        grid=grid, conductivity=k_field, source=q_field, boundary_conditions=bcs, options=options
    )

    validator = get_validator()
    val_res = validator.validate(inf_req)

    k_min = float(np.min(k_field))
    k_max = float(np.max(k_field))
    contrast = float(k_max / max(k_min, 1e-12))
    q_max = float(np.max(np.abs(q_field)))

    errors: list[str] = []
    warnings: list[str] = []

    if not val_res.is_valid:
        errors.append(val_res.rejection.message)

    within_contrast = contrast <= req.options.input_contrast_threshold
    within_source = q_max <= req.options.input_q_max_threshold

    if not within_contrast:
        warnings.append(
            f"Conductivity contrast ratio ({contrast:.2f}:1) exceeds training envelope "
            f"({req.options.input_contrast_threshold:.2f}:1). Surrogate will trigger solver fallback."
        )

    if not within_source:
        warnings.append(
            f"Heat source ({q_max:.2f} W/m^3) exceeds training envelope "
            f"({req.options.input_q_max_threshold:.2f} W/m^3). Surrogate will trigger solver fallback."
        )

    if grid.nx != 32 or grid.ny != 32:
        warnings.append(
            f"Grid resolution ({grid.nx}x{grid.ny}) differs from the surrogate's trained resolution (32x32). "
            "Hybrid engine will route directly to the reference FVM solver."
        )

    return ValidationResponse(
        is_valid=val_res.is_valid,
        surrogate_compatible=val_res.surrogate_compatible and within_contrast and within_source,
        k_min=k_min,
        k_max=k_max,
        k_contrast=contrast,
        q_max=q_max,
        within_contrast_envelope=within_contrast,
        within_source_envelope=within_source,
        errors=errors,
        warnings=warnings,
    )


@router.post("/infer/hybrid", response_model=InferenceResponse)
def run_hybrid_inference(req: InferenceRequestModel) -> InferenceResponse:
    """Execute the Hybrid Digital Twin inference pipeline.

    Coordinates input validation, surrogate execution, safety screening,
    and automatic conservative fallback to the reference FVM solver.
    """
    try:
        inf_req, k_field, q_field = build_inference_request(req)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid physical problem specification: {exc}",
        ) from exc

    GOVERNOR.acquire_neural()
    try:
        engine = get_engine()
        result = engine.predict(inf_req)
    finally:
        GOVERNOR.release_neural()
    return format_hybrid_result_to_response(result, inf_req.grid, k_field, q_field)


@router.post("/infer/fvm", response_model=InferenceResponse)
def run_direct_fvm_solve(req: InferenceRequestModel) -> InferenceResponse:
    """Execute the deterministic reference finite-volume solver directly.

    Solves the conservative discrete linear system A * T = b. Never fabricates uncertainty.
    """
    try:
        inf_req, k_field, q_field = build_inference_request(req)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid physical problem specification: {exc}",
        ) from exc

    validator = get_validator()
    val_res = validator.validate(inf_req)
    if not val_res.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Physical problem failed validation: {val_res.rejection.message}",
        )

    GOVERNOR.acquire_numerical()
    t0 = time.perf_counter()
    try:
        solver = get_solver()
        solver_res = solver.solve(
            grid=inf_req.grid,
            conductivity=k_field,
            source=q_field,
            boundary_conditions=inf_req.boundary_conditions,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Finite-volume reference solver failed: {exc}",
        ) from exc
    finally:
        GOVERNOR.release_numerical()

    latency_ms = (time.perf_counter() - t0) * 1000.0
    METRICS.record_stage_latency("fvm_solve", latency_ms)
    return format_solver_result_to_response(solver_res, inf_req.grid, k_field, q_field, latency_ms)


@router.post("/infer/compare", response_model=ComparisonResponse)
def run_research_comparison(req: InferenceRequestModel) -> ComparisonResponse:
    """Execute both the neural surrogate and reference FVM solver on the identical problem.

    Provides transparent research comparison metrics (RMSE, Rel L2, MaxAE, latency ratio).
    Clearly demarcates whether the surrogate was accepted or evaluated for comparison only.
    """
    try:
        inf_req, k_field, q_field = build_inference_request(req)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid physical problem specification: {exc}",
        ) from exc

    validator = get_validator()
    val_res = validator.validate(inf_req)
    if not val_res.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Physical problem failed validation: {val_res.rejection.message}",
        )

    # 1. Run direct reference FVM solve
    solver = get_solver()
    t0 = time.perf_counter()
    solver_res = solver.solve(
        grid=inf_req.grid,
        conductivity=k_field,
        source=q_field,
        boundary_conditions=inf_req.boundary_conditions,
    )
    t_fvm_ms = (time.perf_counter() - t0) * 1000.0
    fvm_response = format_solver_result_to_response(
        solver_res, inf_req.grid, k_field, q_field, t_fvm_ms
    )

    # 2. Run hybrid inference engine
    engine = get_engine()
    hybrid_res = engine.predict(inf_req)
    hybrid_response = format_hybrid_result_to_response(hybrid_res, inf_req.grid, k_field, q_field)

    # 3. Always compute raw surrogate prediction for research comparison (if 32x32)
    surrogate_field: np.ndarray | None = None
    if inf_req.grid.nx == 32 and inf_req.grid.ny == 32:
        try:
            x_tensor = engine._construct_surrogate_input(
                inf_req.grid, k_field, q_field, inf_req.boundary_conditions
            )
            pred = engine.ensemble.predict(
                x_tensor,
                t_mean=engine.provenance.temperature_mean_kelvin,
                t_std=engine.provenance.temperature_std_kelvin,
            )
            surrogate_field = pred.mean_kelvin.squeeze().cpu().numpy().astype(np.float64)
        except Exception as exc:
            logger.warning("Could not compute raw surrogate comparison: %s", exc)

    # Compute comparison metrics between surrogate and reference FVM
    diff_field: list[list[float]] | None = None
    rmse_k: float | None = None
    rel_l2_pct: float | None = None
    max_ae_k: float | None = None

    if surrogate_field is not None:
        diff = surrogate_field - solver_res.temperature
        diff_field = diff.tolist()
        rmse_k = float(np.sqrt(np.mean(diff**2)))
        rel_l2_pct = float((np.linalg.norm(diff) / np.linalg.norm(solver_res.temperature)) * 100.0)
        max_ae_k = float(np.max(np.abs(diff)))

    t_surr_ms = hybrid_res.latency.surrogate_inference_ms or hybrid_res.latency.total_pipeline_ms
    speedup_ratio = float(t_surr_ms / max(t_fvm_ms, 1e-6))

    status_tag = hybrid_res.status.value
    if hybrid_res.status == InferenceStatus.ACCEPTED:
        notice = "Surrogate PASSED all safety gates and was accepted as the verified output."
    else:
        notice = (
            f"Surrogate was REJECTED ({hybrid_res.rejection.message}) and replaced by FVM fallback in "
            "the hybrid result. The raw surrogate field is displayed here strictly for research comparison."
        )

    return ComparisonResponse(
        problem_summary={
            "grid": f"{inf_req.grid.nx}x{inf_req.grid.ny}",
            "conductivity_contrast": float(np.max(k_field) / max(np.min(k_field), 1e-12)),
            "source_q_max": float(np.max(np.abs(q_field))),
        },
        hybrid_inference=hybrid_response,
        fvm_reference=fvm_response,
        temperature_difference_field=diff_field,
        rmse_k=rmse_k,
        relative_l2_error_pct=rel_l2_pct,
        max_absolute_error_k=max_ae_k,
        surrogate_latency_ms=t_surr_ms,
        fvm_latency_ms=t_fvm_ms,
        fvm_speedup_factor=speedup_ratio,
        surrogate_acceptance_status=status_tag,
        evaluation_notice=notice,
    )


# =============================================================================
# Milestone 09: Dual-Mode Platform Simulation Endpoints
# =============================================================================


@router.post("/simulate", response_model=SimulationResponse)
def run_simulation(req: SimulationRequestModel) -> SimulationResponse:
    """Execute dual-mode simulation: Verified Numerical Reference or Experimental Neural Surrogate.

    Strictly separates trusted numerical FVM calculations from experimental neural approximations.
    """
    try:
        grid = build_domain_grid(req.grid)
        k_field = build_conductivity_array(grid, req.conductivity)
        q_field = build_source_array(grid, req.source)
        bcs = build_boundary_conditions(req.boundary_conditions)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid physical problem specification: {exc}",
        ) from exc

    # 1. Mode: Verified Numerical FVM Reference (Default Trusted Path)
    if req.mode == "numerical_fvm":
        GOVERNOR.acquire_numerical()
        t0 = time.perf_counter()
        try:
            solver = get_solver()
            solver_res = solver.solve(
                grid=grid,
                conductivity=k_field,
                source=q_field,
                boundary_conditions=bcs,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Numerical reference solver execution failed: {exc}",
            ) from exc
        finally:
            GOVERNOR.release_numerical()

        latency_ms = (time.perf_counter() - t0) * 1000.0
        METRICS.record_stage_latency("fvm_solve", latency_ms)
        return format_fvm_simulation_response(solver_res, grid, k_field, q_field, bcs, latency_ms)

    # 2. Mode: Experimental Neural Surrogate (A3 or A4)
    if req.mode in ("experimental_neural", "experimental_neural_corrected"):
        apply_correction = req.mode == "experimental_neural_corrected"
        GOVERNOR.acquire_neural()
        try:
            surrogate_service = get_experimental_surrogate()
            surr_res = surrogate_service.predict(
                grid=grid,
                k_field=k_field,
                q_field=q_field,
                bcs=bcs,
                apply_physics_correction=apply_correction,
                compute_diagnostics=req.compute_diagnostics,
            )
            METRICS.record_stage_latency("neural_forward", surr_res.inference_time_ms)
            if apply_correction:
                METRICS.record_stage_latency("physics_correction", surr_res.correction_time_ms)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Experimental neural surrogate execution failed: {exc}",
            ) from exc
        finally:
            GOVERNOR.release_neural()

        return format_experimental_simulation_response(
            surr_res, grid, k_field, q_field, bcs, mode_name=req.mode
        )

    # 3. Mode: Legacy Hybrid Engine (Milestone 06)
    if req.mode == "legacy_hybrid":
        options = build_inference_options(req.options)
        inf_req = InferenceRequest(
            grid=grid,
            conductivity=k_field,
            source=q_field,
            boundary_conditions=bcs,
            options=options,
        )
        GOVERNOR.acquire_neural()
        try:
            engine = get_engine()
            hybrid_res = engine.predict(inf_req)
        finally:
            GOVERNOR.release_neural()
        return format_legacy_hybrid_simulation_response(hybrid_res, grid, k_field, q_field, bcs)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Unsupported simulation mode: '{req.mode}'.",
    )


@router.post("/compare", response_model=DualModeComparisonResponse)
def run_dual_mode_comparison(req: SimulationRequestModel) -> DualModeComparisonResponse:
    """Execute both Verified Numerical Reference and Experimental Neural Surrogate on the identical problem.

    Computes ground-truth error metrics strictly from the actual numerical reference solve.
    Never manufactures reference errors.
    """
    try:
        grid = build_domain_grid(req.grid)
        k_field = build_conductivity_array(grid, req.conductivity)
        q_field = build_source_array(grid, req.source)
        bcs = build_boundary_conditions(req.boundary_conditions)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid physical problem specification: {exc}",
        ) from exc

    # 1. Execute Verified Numerical Reference solve
    GOVERNOR.acquire_numerical()
    t0 = time.perf_counter()
    try:
        solver = get_solver()
        solver_res = solver.solve(
            grid=grid,
            conductivity=k_field,
            source=q_field,
            boundary_conditions=bcs,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Numerical reference solver execution failed: {exc}",
        ) from exc
    finally:
        GOVERNOR.release_numerical()

    fvm_lat_ms = (time.perf_counter() - t0) * 1000.0
    METRICS.record_stage_latency("fvm_solve", fvm_lat_ms)
    fvm_response = format_fvm_simulation_response(
        solver_res, grid, k_field, q_field, bcs, fvm_lat_ms
    )

    # 2. Execute Experimental Neural Surrogate
    apply_corr = req.mode == "experimental_neural_corrected"
    GOVERNOR.acquire_neural()
    try:
        surrogate_service = get_experimental_surrogate()
        surr_res = surrogate_service.predict(
            grid=grid,
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            apply_physics_correction=apply_corr,
        )
        METRICS.record_stage_latency("neural_forward", surr_res.inference_time_ms)
        if apply_corr:
            METRICS.record_stage_latency("physics_correction", surr_res.correction_time_ms)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experimental neural surrogate execution failed: {exc}",
        ) from exc
    finally:
        GOVERNOR.release_neural()

    surr_response = format_experimental_simulation_response(
        surr_res,
        grid,
        k_field,
        q_field,
        bcs,
        mode_name="experimental_neural_corrected" if apply_corr else "experimental_neural",
    )

    # 3. Compute Ground-Truth Error Metrics
    diff = surr_res.temperature - solver_res.temperature
    rmse_k = float(np.sqrt(np.mean(diff**2)))
    max_ae_k = float(np.max(np.abs(diff)))
    t_ref_norm = float(np.linalg.norm(solver_res.temperature))
    rel_l2_pct = float((np.linalg.norm(diff) / max(t_ref_norm, 1e-12)) * 100.0)

    speedup = float(fvm_lat_ms / max(surr_res.total_time_ms, 1e-6))

    eval_notice = (
        f"Ground-truth comparison against verified numerical FVM reference ({grid.nx}x{grid.ny}). "
        f"MaxAE: {max_ae_k:.2f} K (Tolerance: <= 10.0 K -> {'PASS' if max_ae_k <= 10.0 else 'VIOLATION'}). "
        f"Speedup: {speedup:.2f}x."
    )

    return DualModeComparisonResponse(
        problem_summary={
            "grid": f"{grid.nx}x{grid.ny}",
            "conductivity_contrast": float(np.max(k_field) / max(np.min(k_field), 1e-12)),
            "source_q_max": float(np.max(np.abs(q_field))),
            "boundary_conditions": {
                "left": f"{bcs.left.bc_type.value}: {bcs.left.value}",
                "right": f"{bcs.right.bc_type.value}: {bcs.right.value}",
                "bottom": f"{bcs.bottom.bc_type.value}: {bcs.bottom.value}",
                "top": f"{bcs.top.bc_type.value}: {bcs.top.value}",
            },
        },
        numerical_reference=fvm_response,
        experimental_surrogate=surr_response,
        temperature_difference_field=diff.tolist(),
        rmse_k=rmse_k,
        relative_l2_error_pct=rel_l2_pct,
        max_absolute_error_k=max_ae_k,
        fvm_latency_ms=fvm_lat_ms,
        surrogate_latency_ms=surr_res.total_time_ms,
        speedup_factor=speedup,
        evaluation_notice=eval_notice,
    )
