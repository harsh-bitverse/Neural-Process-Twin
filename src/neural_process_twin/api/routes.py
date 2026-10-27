"""FastAPI router endpoints for Neural Process Twin hybrid inference and validation."""

from __future__ import annotations

import logging
import time

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
    format_hybrid_result_to_response,
    format_solver_result_to_response,
)
from neural_process_twin.api.models import (
    CapabilitiesResponse,
    ComparisonResponse,
    HealthResponse,
    InferenceRequestModel,
    InferenceResponse,
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
from neural_process_twin.inference.validator import InputValidator
from neural_process_twin.solvers.solver import HeatConductionSolver2D

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Inference & Diagnostics"])

# Singleton engine instance, initialized lazily or via startup event
_ENGINE_INSTANCE: HybridInferenceEngine | None = None
_VALIDATOR_INSTANCE: InputValidator | None = None
_SOLVER_INSTANCE: HeatConductionSolver2D | None = None


def get_engine() -> HybridInferenceEngine:
    """Retrieve or lazily initialize the singleton HybridInferenceEngine."""
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        logger.info("Initializing HybridInferenceEngine from authoritative repository artifacts...")
        _ENGINE_INSTANCE = HybridInferenceEngine.from_artifacts(device="cpu")
    return _ENGINE_INSTANCE


def get_validator() -> InputValidator:
    """Retrieve or lazily initialize the singleton InputValidator."""
    global _VALIDATOR_INSTANCE
    if _VALIDATOR_INSTANCE is None:
        _VALIDATOR_INSTANCE = InputValidator()
    return _VALIDATOR_INSTANCE


def get_solver() -> HeatConductionSolver2D:
    """Retrieve or lazily initialize the reference HeatConductionSolver2D."""
    global _SOLVER_INSTANCE
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


@router.get("/capabilities", response_model=CapabilitiesResponse)
def get_capabilities() -> CapabilitiesResponse:
    """Expose supported model architectures, training limits, and performance expectations."""
    members = [{"member_name": name, "sha256": sha} for name, sha in FROZEN_ENSEMBLE_SHA256.items()]
    return CapabilitiesResponse(
        surrogate_supported_resolution=(32, 32),
        fvm_supported_resolution_range="4x4 to 128x128 (arbitrary cell counts and aspect ratios)",
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
            "SCIENTIFIC PERFORMANCE REALITY: On a single 32x32 mesh on CPU, direct sparse numerical "
            "factorization (SuperLU) executes in ~4-6 ms (~180 samples/s). The 5-member deep neural operator "
            "ensemble executes sequentially on CPU in ~55-65 ms (~16 samples/s). The direct FVM solver is "
            "11-13x faster for single CPU queries. The neural surrogate delivers speedups for massive parallel "
            "batch evaluation on GPU and for high-resolution 3D domains where sparse matrix factorizations scale "
            "superlinearly."
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

    engine = get_engine()
    result = engine.predict(inf_req)
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

    solver = get_solver()
    t0 = time.perf_counter()
    try:
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

    latency_ms = (time.perf_counter() - t0) * 1000.0
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
