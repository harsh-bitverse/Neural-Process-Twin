"""Hybrid Digital Twin inference engine orchestrating surrogate execution and solver fallback.

Integrates physical input validation, 12-channel tensor encoding, frozen deep ensemble
inference, split-conformal prediction intervals, discrete physical diagnostics,
multi-tiered safety screening, and fail-closed numerical reference solver fallback.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from neural_process_twin.data.boundary_encoding import encode_boundary_conditions
from neural_process_twin.data.normalization import FieldNormalizer, FieldStatistics
from neural_process_twin.evaluation.residual import compute_discrete_pde_residual
from neural_process_twin.inference.contracts import (
    ConformalIntervalOutput,
    ExecutionLatency,
    HybridInferenceResult,
    InferenceProvenance,
    InferenceRequest,
    InferenceStatus,
    PhysicalDiagnosticsOutput,
    RejectionCode,
    RejectionRecord,
)
from neural_process_twin.inference.screening import SafetyScreener
from neural_process_twin.inference.validator import InputValidator
from neural_process_twin.solvers.diagnostics import compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D, SolverResult
from neural_process_twin.uncertainty.calibration import ConformalIntervalCalibrator
from neural_process_twin.uncertainty.ensemble import DeepEnsemble

logger = logging.getLogger(__name__)

# Authoritative Milestone 04B / 05B frozen checkpoint digests
FROZEN_ENSEMBLE_SHA256: dict[str, str] = {
    "scaled_fno_supervised_seed42": "37b0b1a4b717a9c6be2558420fcde6aaff498c79d76aec7e1712b16eb5f82b02",
    "scaled_fno_supervised_seed43": "b7ef5e3c9e59b467be8c9fb783f0ea4a2f1c5316661ed1fc38f36a7e7b34e90a",
    "scaled_fno_supervised_seed44": "379bbdf16b28553e15431844bc94e71ab6785e45fdc0093117b9350966261434",
    "scaled_fno_supervised_seed45": "b407a34628f7953122a18e003a44b2013b7d387b6d9e95b039b6eeef96777bf8",
    "scaled_fno_supervised_seed46": "722123f37783b48eaf58c537df21af64af0a3f434dae037b9246a43f751a3905",
}


def _process_solve_worker(
    grid: Grid2D,
    conductivity: np.ndarray,
    source: np.ndarray,
    bcs: Any,
    gauge_pin: tuple[int, int] | None,
    gauge_temperature: float | None,
    solver_tolerance: float,
    result_queue: Any,
) -> None:
    """Worker function executed in an isolated process for hard-cancellable fallback solves."""
    try:
        solver = HeatConductionSolver2D(tolerance=solver_tolerance)
        res = solver.solve(
            grid=grid,
            conductivity=conductivity,
            source=source,
            boundary_conditions=bcs,
            gauge_pin=gauge_pin,
            gauge_temperature=gauge_temperature,
        )
        result_queue.put((True, res))
    except Exception as exc:
        result_queue.put((False, exc))


def _compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _load_normalizer_from_h5(h5_path: Path) -> FieldNormalizer:
    """Extract fitted normalizer from existing scaled dataset archive."""
    with h5py.File(h5_path, "r") as f:
        if "normalization" not in f:
            raise ValueError(f"No normalization group found in {h5_path}")
        norm_grp = f["normalization"]
        method = str(norm_grp.attrs.get("method", "standard"))
        eps = float(norm_grp.attrs.get("eps", 1e-8))
        norm = FieldNormalizer(method=method, eps=eps)

        def read_stats(grp_name: str) -> FieldStatistics | None:
            if grp_name in norm_grp:
                sg = norm_grp[grp_name]
                return FieldStatistics(
                    mean=float(sg.attrs["mean"]),
                    std=float(sg.attrs["std"]),
                    min=float(sg.attrs["min"]),
                    max=float(sg.attrs["max"]),
                )
            return None

        norm.k_stats = read_stats("k_stats") or read_stats("k")
        norm.q_stats = read_stats("q_stats") or read_stats("q")
        norm.t_stats = read_stats("t_stats") or read_stats("temperature")
        if not norm.is_fitted:
            raise ValueError(f"Incomplete normalization statistics in {h5_path}")
        return norm


class HybridInferenceEngine:
    """Production-grade Hybrid Digital Twin inference engine."""

    def __init__(
        self,
        ensemble: DeepEnsemble,
        normalizer: FieldNormalizer,
        solver: HeatConductionSolver2D | None = None,
        validator: InputValidator | None = None,
        screener: SafetyScreener | None = None,
        pointwise_calibrator: ConformalIntervalCalibrator | None = None,
        simultaneous_calibrator: ConformalIntervalCalibrator | None = None,
        provenance: InferenceProvenance | None = None,
    ) -> None:
        """Initialize hybrid engine with required components.

        Args:
            ensemble: Initialized DeepEnsemble surrogate model.
            normalizer: Fitted FieldNormalizer for input/output scaling.
            solver: Numerical reference solver (HeatConductionSolver2D).
            validator: InputValidator instance.
            screener: SafetyScreener instance.
            pointwise_calibrator: Calibrated ConformalIntervalCalibrator for cell coverage.
            simultaneous_calibrator: Calibrated ConformalIntervalCalibrator for field coverage.
            provenance: InferenceProvenance tracking model and calibration metadata.
        """
        self.ensemble = ensemble
        self.normalizer = normalizer
        self.solver = solver or HeatConductionSolver2D(tolerance=1e-10)
        self.validator = validator or InputValidator()
        self.screener = screener or SafetyScreener()
        self.pointwise_calibrator = pointwise_calibrator
        self.simultaneous_calibrator = simultaneous_calibrator
        self.provenance = provenance or self._default_provenance()

    def _default_provenance(self) -> InferenceProvenance:
        """Generate default provenance when not explicitly provided."""
        t_mean = 0.0
        t_std = 1.0
        if self.normalizer.t_stats:
            t_mean = self.normalizer.t_stats.mean
            t_std = self.normalizer.t_stats.std

        return InferenceProvenance(
            ensemble_member_checkpoints=self.ensemble.member_names,
            ensemble_member_sha256=["unspecified"] * len(self.ensemble.members),
            normalizer_method=self.normalizer.method,
            temperature_mean_kelvin=t_mean,
            temperature_std_kelvin=t_std,
            calibration_dataset_id="provisional_default",
            calibration_confidence_level=0.95,
            solver_name="HeatConductionSolver2D",
            solver_tolerance=self.solver.tolerance,
        )

    @classmethod
    def from_artifacts(
        cls,
        checkpoint_paths: list[str | Path] | None = None,
        calibration_results_path: str | Path = Path(
            "artifacts/reports/milestone_05b_calibration_results.json"
        ),
        reference_dataset_path: str | Path = Path("artifacts/data/scaled_dataset.h5"),
        device: str | torch.device = "cpu",
    ) -> HybridInferenceEngine:
        """Instantiate HybridInferenceEngine from repository checkpoints and calibration artifacts."""
        if checkpoint_paths is None:
            checkpoint_paths = [
                Path(f"artifacts/checkpoints/scaled_fno_supervised_seed{s}/best_model.pt")
                for s in (42, 43, 44, 45, 46)
            ]

        resolved_paths = [Path(p).resolve() for p in checkpoint_paths]
        sha256_digests = [_compute_file_sha256(p) for p in resolved_paths]

        # Verify checkpoint integrity against authoritative frozen digests
        for p, sha in zip(resolved_paths, sha256_digests, strict=True):
            member_name = p.parent.name
            if member_name in FROZEN_ENSEMBLE_SHA256:
                expected_sha = FROZEN_ENSEMBLE_SHA256[member_name]
                if sha != expected_sha:
                    raise ValueError(
                        f"Checkpoint integrity violation for {member_name}: "
                        f"expected SHA-256 {expected_sha}, observed {sha}. "
                        "The frozen model artifact has been modified or corrupted."
                    )

        # Load DeepEnsemble
        ensemble = DeepEnsemble.from_checkpoints(resolved_paths, device=device)

        # Load Normalizer from scaled dataset
        normalizer = _load_normalizer_from_h5(Path(reference_dataset_path))

        # Load Conformal Calibrators from calibration results JSON
        cal_path = Path(calibration_results_path)
        pointwise_calibrator = None
        simultaneous_calibrator = None
        cal_dataset_id = "unknown"

        if cal_path.is_file():
            with open(cal_path, encoding="utf-8") as f:
                cal_data = json.load(f)
            cal_info = cal_data.get("calibration_dataset_info", {})
            cal_dataset_id = cal_info.get("path", str(cal_path))

            # Retrieve fitted quantiles
            cal_src = (
                cal_data.get("conformal_calibration", {})
                .get("calibration_source", {})
                .get("independent_calibration", {})
            )
            if "pointwise_95" in cal_src and "simultaneous_95" in cal_src:
                q_point = cal_src["pointwise_95"]["quantile_multiplier_q_hat"]
                q_sim = cal_src["simultaneous_95"]["quantile_multiplier_q_hat"]

                pointwise_calibrator = ConformalIntervalCalibrator(
                    confidence_level=0.95, coverage_type="pointwise"
                )
                pointwise_calibrator.quantile_multiplier = float(q_point)

                simultaneous_calibrator = ConformalIntervalCalibrator(
                    confidence_level=0.95, coverage_type="simultaneous"
                )
                simultaneous_calibrator.quantile_multiplier = float(q_sim)

        provenance = InferenceProvenance(
            ensemble_member_checkpoints=[str(p) for p in resolved_paths],
            ensemble_member_sha256=sha256_digests,
            normalizer_method=normalizer.method,
            temperature_mean_kelvin=normalizer.t_stats.mean if normalizer.t_stats else 0.0,
            temperature_std_kelvin=normalizer.t_stats.std if normalizer.t_stats else 1.0,
            calibration_dataset_id=cal_dataset_id,
            calibration_confidence_level=0.95,
            solver_name="HeatConductionSolver2D",
            solver_tolerance=1e-10,
        )

        return cls(
            ensemble=ensemble,
            normalizer=normalizer,
            solver=HeatConductionSolver2D(tolerance=1e-10),
            validator=InputValidator(),
            screener=SafetyScreener(),
            pointwise_calibrator=pointwise_calibrator,
            simultaneous_calibrator=simultaneous_calibrator,
            provenance=provenance,
        )

    def _construct_surrogate_input(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: Any,
    ) -> torch.Tensor:
        """Construct the canonical 12-channel input tensor for FNO-2D forward pass."""
        k_feat = self.normalizer.transform_k(k_field)
        q_feat = self.normalizer.transform_q(q_field)

        lx = max(grid.lx, 1e-12)
        ly = max(grid.ly, 1e-12)
        norm_x_mesh = ((grid.x_mesh - grid.x_min) / lx).astype(np.float32)
        norm_y_mesh = ((grid.y_mesh - grid.y_min) / ly).astype(np.float32)

        bc_channels = encode_boundary_conditions(bcs, grid=grid, normalizer=self.normalizer)

        channels = [
            k_feat.astype(np.float32),
            q_feat.astype(np.float32),
            norm_x_mesh,
            norm_y_mesh,
        ]
        for c in range(8):
            channels.append(bc_channels[c])

        stacked = np.stack(channels, axis=0)  # Shape (12, ny, nx)
        tensor = torch.from_numpy(stacked).unsqueeze(0)  # Shape (1, 12, ny, nx)
        return tensor

    def _evaluate_diagnostics(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: Any,
        temperature_k: np.ndarray,
    ) -> PhysicalDiagnosticsOutput:
        """Compute independent discrete PDE residuals and global energy balance.

        Preserves general face areas (face_area_x = dy * thickness, face_area_y = dx * thickness)
        and explicitly calculates signed net imbalance Q_out - Q_gen separately from absolute error.
        """
        pde_res = compute_discrete_pde_residual(
            grid=grid,
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            temperature_pred=temperature_k,
        )

        heat_diag = compute_heat_balance(
            grid=grid,
            temperature=temperature_k,
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
        )

        signed_imbalance = heat_diag.q_boundary_out_total - heat_diag.q_generated_total

        return PhysicalDiagnosticsOutput(
            pde_residual_l2_w=pde_res["pde_residual_l2_w"],
            pde_residual_relative=pde_res["pde_residual_relative"],
            pde_residual_linf_w=pde_res["pde_residual_linf_w"],
            q_generated_total_w=heat_diag.q_generated_total,
            q_boundary_out_total_w=heat_diag.q_boundary_out_total,
            signed_net_imbalance_w=signed_imbalance,
            absolute_flux_imbalance_w=heat_diag.flux_imbalance,
            relative_flux_imbalance=heat_diag.relative_imbalance,
            boundary_heat_flow_out_w=heat_diag.boundary_heat_flow_out,
            boundary_flux_densities_w_m2=heat_diag.boundary_flux_densities,
        )

    def _execute_solver_fallback(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: Any,
        gauge_pin: tuple[int, int] | None,
        gauge_temperature: float | None,
        timeout_seconds: float,
        isolation: str = "thread",
    ) -> SolverResult:
        """Execute numerical reference solver guarded by timeout with thread or process isolation."""
        if isolation == "process":
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            q = ctx.Queue()
            proc = ctx.Process(
                target=_process_solve_worker,
                args=(
                    grid,
                    k_field,
                    q_field,
                    bcs,
                    gauge_pin,
                    gauge_temperature,
                    self.solver.tolerance,
                    q,
                ),
            )
            proc.start()
            proc.join(timeout=timeout_seconds)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
                raise TimeoutError(
                    f"Numerical solver exceeded timeout of {timeout_seconds:.1f}s (process terminated)."
                )
            if q.empty():
                raise RuntimeError("Solver process terminated without producing a result.")
            success, payload = q.get_nowait()
            if not success:
                raise payload
            return payload

        # Thread isolation (non-blocking shutdown on timeout to prevent hanging on context exit)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self.solver.solve,
            grid=grid,
            conductivity=k_field,
            source=q_field,
            boundary_conditions=bcs,
            gauge_pin=gauge_pin,
            gauge_temperature=gauge_temperature,
        )
        try:
            result = future.result(timeout=timeout_seconds)
            executor.shutdown(wait=False, cancel_futures=True)
            return result
        except concurrent.futures.TimeoutError as exc:
            executor.shutdown(wait=False, cancel_futures=True)
            raise TimeoutError(
                f"Numerical solver exceeded timeout of {timeout_seconds:.1f}s."
            ) from exc
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    def predict(self, request: InferenceRequest) -> HybridInferenceResult:
        """Execute the Hybrid Digital Twin inference pipeline.

        Coordinates:
        1. Input validation and physical well-posedness checks.
        2. Surrogate architectural screening (e.g. grid resolution compatibility).
        3. Frozen deep ensemble forward pass and uncertainty evaluation.
        4. Independent discrete PDE residual and boundary flux diagnostics.
        5. Safety screening (disagreement, contrast, heat source, residual).
        6. Fail-closed fallback to numerical solver when required.
        7. Construction of verified result container with provenance and timings.

        Args:
            request: InferenceRequest specifying grid, fields, boundary conditions, and options.

        Returns:
            HybridInferenceResult with strict status and fail-closed guarantees.
        """
        t_pipeline_start = time.perf_counter()
        options = request.options

        # =====================================================================
        # Stage 1: Input Validation
        # =====================================================================
        t0 = time.perf_counter()
        val_res = self.validator.validate(request)
        t_val_ms = (time.perf_counter() - t0) * 1000.0

        if not val_res.is_valid:
            # Fatal input validation failure: abort pipeline immediately
            t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            latency = ExecutionLatency(
                input_validation_ms=t_val_ms,
                surrogate_inference_ms=0.0,
                diagnostics_evaluation_ms=0.0,
                solver_fallback_ms=0.0,
                total_pipeline_ms=t_total_ms,
            )
            return HybridInferenceResult(
                temperature=None,  # Fail-closed: never return invalid field
                status=InferenceStatus.INVALID_INPUT,
                execution_path="aborted",
                rejection=val_res.rejection,
                ensemble_disagreement_kelvin=None,
                mean_disagreement_kelvin=None,
                max_disagreement_kelvin=None,
                conformal_intervals=None,
                diagnostics=None,
                latency=latency,
                provenance=self.provenance,
            )

        k_field = val_res.evaluated_k
        q_field = val_res.evaluated_q
        assert k_field is not None and q_field is not None

        # =====================================================================
        # Stage 2: Surrogate Compatibility & Inference
        # =====================================================================
        surrogate_accepted = False
        y_pred_k: np.ndarray | None = None
        y_std_k: np.ndarray | None = None
        mean_s_t: float | None = None
        max_s_t: float | None = None
        screen_rejection = val_res.rejection
        surr_diagnostics: PhysicalDiagnosticsOutput | None = None
        conformal_out: ConformalIntervalOutput | None = None
        t_surr_ms = 0.0
        t_diag_ms = 0.0

        if val_res.surrogate_compatible:
            # Construct 12-channel input tensor
            t0 = time.perf_counter()
            x_tensor = self._construct_surrogate_input(
                grid=request.grid,
                k_field=k_field,
                q_field=q_field,
                bcs=request.boundary_conditions,
            )

            # Ensemble forward pass
            pred = self.ensemble.predict(
                x_tensor,
                t_mean=self.provenance.temperature_mean_kelvin,
                t_std=self.provenance.temperature_std_kelvin,
            )
            y_pred_k = pred.mean_kelvin.squeeze(0).squeeze(0).cpu().numpy()
            y_std_k = pred.std_kelvin.squeeze(0).squeeze(0).cpu().numpy()
            t_surr_ms = (time.perf_counter() - t0) * 1000.0

            # Evaluate physical diagnostics on surrogate prediction
            t0 = time.perf_counter()
            surr_diagnostics = self._evaluate_diagnostics(
                grid=request.grid,
                k_field=k_field,
                q_field=q_field,
                bcs=request.boundary_conditions,
                temperature_k=y_pred_k,
            )
            t_diag_ms = (time.perf_counter() - t0) * 1000.0

            # Screen surrogate prediction
            screen_res = self.screener.screen(
                predicted_temperature_k=y_pred_k,
                ensemble_std_k=y_std_k,
                k_field=k_field,
                q_field=q_field,
                diagnostics=surr_diagnostics,
                options=options,
            )
            mean_s_t = screen_res.mean_disagreement_kelvin
            max_s_t = screen_res.max_disagreement_kelvin
            surrogate_accepted = screen_res.accepted
            screen_rejection = screen_res.rejection

            if surrogate_accepted:
                # Construct calibrated prediction intervals
                q_point = (
                    self.pointwise_calibrator.quantile_multiplier
                    if self.pointwise_calibrator and self.pointwise_calibrator.quantile_multiplier
                    else 3.9301
                )
                q_sim = (
                    self.simultaneous_calibrator.quantile_multiplier
                    if self.simultaneous_calibrator
                    and self.simultaneous_calibrator.quantile_multiplier
                    else 31.1723
                )
                eps = 1e-4
                half_point = q_point * (y_std_k + eps)
                half_sim = q_sim * (y_std_k + eps)

                conformal_out = ConformalIntervalOutput(
                    pointwise_lower_kelvin=y_pred_k - half_point,
                    pointwise_upper_kelvin=y_pred_k + half_point,
                    simultaneous_lower_kelvin=y_pred_k - half_sim,
                    simultaneous_upper_kelvin=y_pred_k + half_sim,
                    pointwise_mean_width_kelvin=float(np.mean(2.0 * half_point)),
                    simultaneous_mean_width_kelvin=float(np.mean(2.0 * half_sim)),
                    nominal_confidence=options.conformal_confidence_level,
                    pointwise_q_hat=q_point,
                    simultaneous_q_hat=q_sim,
                )

                t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
                latency = ExecutionLatency(
                    input_validation_ms=t_val_ms,
                    surrogate_inference_ms=t_surr_ms,
                    diagnostics_evaluation_ms=t_diag_ms,
                    solver_fallback_ms=0.0,
                    total_pipeline_ms=t_total_ms,
                )

                return HybridInferenceResult(
                    temperature=y_pred_k,
                    status=InferenceStatus.ACCEPTED,
                    execution_path="surrogate_fast_path",
                    rejection=screen_rejection,
                    ensemble_disagreement_kelvin=y_std_k,
                    mean_disagreement_kelvin=mean_s_t,
                    max_disagreement_kelvin=max_s_t,
                    conformal_intervals=conformal_out,
                    diagnostics=surr_diagnostics,
                    latency=latency,
                    provenance=self.provenance,
                )

        # =====================================================================
        # Stage 3: Fallback Routing to Numerical Reference Solver
        # =====================================================================
        if not options.enable_fallback:
            # Fallback is disabled by policy: fail closed without invoking solver
            t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            latency = ExecutionLatency(
                input_validation_ms=t_val_ms,
                surrogate_inference_ms=t_surr_ms,
                diagnostics_evaluation_ms=t_diag_ms,
                solver_fallback_ms=0.0,
                total_pipeline_ms=t_total_ms,
            )
            return HybridInferenceResult(
                temperature=None,  # Fail-closed: never return rejected surrogate output
                status=InferenceStatus.FALLBACK_REQUIRED,
                execution_path="aborted",
                rejection=RejectionRecord(
                    code=RejectionCode.FALLBACK_DISABLED,
                    message=f"Surrogate rejected ({screen_rejection.message}), and fallback is disabled.",
                    triggering_metric=screen_rejection.triggering_metric,
                    observed_value=screen_rejection.observed_value,
                    threshold_value=screen_rejection.threshold_value,
                ),
                ensemble_disagreement_kelvin=y_std_k,
                mean_disagreement_kelvin=mean_s_t,
                max_disagreement_kelvin=max_s_t,
                conformal_intervals=None,
                diagnostics=surr_diagnostics,
                latency=latency,
                provenance=self.provenance,
            )

        # Execute numerical solver fallback
        t0 = time.perf_counter()
        solver_failed = False
        solver_rejection = screen_rejection
        solver_t_field: np.ndarray | None = None
        solver_diagnostics: PhysicalDiagnosticsOutput | None = None

        try:
            solver_res = self._execute_solver_fallback(
                grid=request.grid,
                k_field=k_field,
                q_field=q_field,
                bcs=request.boundary_conditions,
                gauge_pin=request.gauge_pin,
                gauge_temperature=request.gauge_temperature,
                timeout_seconds=options.solver_timeout_seconds,
                isolation=options.fallback_isolation,
            )
            solver_t_field = solver_res.temperature

            # Validate solver convergence and output integrity
            if not solver_res.diagnostics.is_converged:
                solver_failed = True
                solver_rejection = RejectionRecord(
                    code=RejectionCode.SOLVER_DIVERGENCE,
                    message="Numerical reference solver terminated without convergence.",
                    triggering_metric="relative_residual",
                    observed_value=solver_res.diagnostics.relative_residual,
                    threshold_value=self.solver.tolerance,
                )
            elif not np.all(np.isfinite(solver_t_field)) or np.min(solver_t_field) <= 0.0:
                solver_failed = True
                solver_rejection = RejectionRecord(
                    code=RejectionCode.SOLVER_DIVERGENCE,
                    message="Numerical reference solver produced non-finite or non-positive temperature.",
                )
            else:
                # Package solver diagnostics
                signed_imb = (
                    solver_res.diagnostics.energy_balance.q_boundary_out_total
                    - solver_res.diagnostics.energy_balance.q_generated_total
                )
                solver_diagnostics = PhysicalDiagnosticsOutput(
                    pde_residual_l2_w=solver_res.diagnostics.linear_residual_l2,
                    pde_residual_relative=solver_res.diagnostics.relative_residual,
                    pde_residual_linf_w=float(
                        np.max(
                            np.abs(
                                solver_res.linear_system.A.dot(solver_t_field.ravel())
                                - solver_res.linear_system.b
                            )
                        )
                    ),
                    q_generated_total_w=solver_res.diagnostics.energy_balance.q_generated_total,
                    q_boundary_out_total_w=solver_res.diagnostics.energy_balance.q_boundary_out_total,
                    signed_net_imbalance_w=signed_imb,
                    absolute_flux_imbalance_w=solver_res.diagnostics.energy_balance.flux_imbalance,
                    relative_flux_imbalance=solver_res.diagnostics.energy_balance.relative_imbalance,
                    boundary_heat_flow_out_w=solver_res.diagnostics.energy_balance.boundary_heat_flow_out,
                    boundary_flux_densities_w_m2=solver_res.diagnostics.energy_balance.boundary_flux_densities,
                )

        except TimeoutError as err:
            solver_failed = True
            solver_rejection = RejectionRecord(
                code=RejectionCode.SOLVER_TIMEOUT,
                message=str(err),
                triggering_metric="timeout_seconds",
                observed_value=options.solver_timeout_seconds,
                threshold_value=options.solver_timeout_seconds,
            )
        except Exception as err:
            solver_failed = True
            solver_rejection = RejectionRecord(
                code=RejectionCode.SOLVER_EXCEPTION,
                message=f"Numerical reference solver raised unexpected exception: {err}",
            )

        t_solver_ms = (time.perf_counter() - t0) * 1000.0
        t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        latency = ExecutionLatency(
            input_validation_ms=t_val_ms,
            surrogate_inference_ms=t_surr_ms,
            diagnostics_evaluation_ms=t_diag_ms,
            solver_fallback_ms=t_solver_ms,
            total_pipeline_ms=t_total_ms,
        )

        if solver_failed:
            # Fatal fallback failure: fail closed, NEVER return rejected surrogate temperature!
            return HybridInferenceResult(
                temperature=None,
                status=InferenceStatus.SOLVER_FAILED,
                execution_path="aborted",
                rejection=solver_rejection,
                ensemble_disagreement_kelvin=y_std_k,
                mean_disagreement_kelvin=mean_s_t,
                max_disagreement_kelvin=max_s_t,
                conformal_intervals=None,
                diagnostics=surr_diagnostics,
                latency=latency,
                provenance=self.provenance,
            )

        # Fallback succeeded: return verified numerical reference solution
        return HybridInferenceResult(
            temperature=solver_t_field,
            status=InferenceStatus.FALLBACK_SUCCEEDED,
            execution_path="solver_fallback_path",
            rejection=screen_rejection,  # Preserves why surrogate was rejected
            ensemble_disagreement_kelvin=y_std_k,
            mean_disagreement_kelvin=mean_s_t,
            max_disagreement_kelvin=max_s_t,
            conformal_intervals=None,  # Intervals not applicable to deterministic reference solver
            diagnostics=solver_diagnostics,
            latency=latency,
            provenance=self.provenance,
        )
