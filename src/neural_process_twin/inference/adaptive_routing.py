"""Adaptive reliability-aware hybrid routing engine for Milestone 08E.

Implements research routing candidates:
- R0: Direct FVM Reference (PyAMG)
- R1: Existing frozen 5-member production hybrid
- R2: Pre-screening followed by U1 single-pass uncertainty and physical diagnostics
- R3: Pre-screening followed by U3 T=3 multi-pass uncertainty and physical diagnostics
- R4: Pre-screening followed by U3 T=5 multi-pass uncertainty and physical diagnostics
- R5: Adaptive cascade (pre-screening -> U1 single-pass -> optional U3 escalation -> FVM fallback)

Maintains fail-closed semantics and complete execution timing breakdowns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.inference.contracts import (
    InferenceStatus,
    RejectionCode,
    RejectionRecord,
)
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.models.uncertainty_fno import HeteroscedasticFNO2D, MCDropoutFNO2D
from neural_process_twin.solvers.boundary import (
    BoundaryConditions,
    BoundaryType,
)
from neural_process_twin.solvers.diagnostics import EnergyBalanceDiagnostics, compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D

logger = logging.getLogger("adaptive_routing")


class RoutingPolicy(StrEnum):
    """Routing policy candidate identifiers."""

    R0_DIRECT_FVM = "R0_direct_fvm"
    R1_PRODUCTION_HYBRID = "R1_production_hybrid"
    R2_U1_SINGLE_PASS = "R2_u1_single_pass"
    R3_U3_T3_MULTI_PASS = "R3_u3_t3_multi_pass"
    R4_U3_T5_MULTI_PASS = "R4_u3_t5_multi_pass"
    R5_ADAPTIVE_CASCADE = "R5_adaptive_cascade"


@dataclass(frozen=True)
class RoutingThresholds:
    """Configurable fail-closed thresholds for routing gates."""

    tau_k_max: float = 20.0
    tau_q_max: float = 1800.0
    tau_u1_unc: float = 5.60
    tau_u3_t3_unc: float = 7.44
    tau_u3_t5_unc: float = 7.08
    tau_u1_cascade_safe: float = 4.67
    tau_u1_cascade_ambiguous: float = 5.98
    tau_flux_imb_pct: float = 115.16


@dataclass
class AdaptiveRoutingResult:
    """Complete result from adaptive routing execution."""

    temperature: np.ndarray
    uncertainty: np.ndarray | None
    status: InferenceStatus
    policy: RoutingPolicy
    escalated_to_u3: bool = False
    escalated_to_fvm: bool = False
    rejection_record: RejectionRecord | None = None
    diagnostics: EnergyBalanceDiagnostics | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)


class AdaptiveHybridEngine:
    """Adaptive reliability-aware routing inference engine."""

    def __init__(
        self,
        u1_checkpoint: Path | None = None,
        u3_checkpoint: Path | None = None,
        m3_checkpoint: Path | None = None,
        ensemble_dir: Path | None = None,
        thresholds: RoutingThresholds | None = None,
        device: torch.device | None = None,
        num_threads: int = 8,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.thresholds = thresholds or RoutingThresholds()
        torch.set_num_threads(num_threads)

        self.u1_ckpt_path = u1_checkpoint or Path(
            "artifacts/checkpoints/milestone_08d/U1_heteroscedastic/best_model.pt"
        )
        self.u3_ckpt_path = u3_checkpoint or Path(
            "artifacts/checkpoints/milestone_08d/U3_mc_dropout/best_model.pt"
        )
        self.m3_ckpt_path = m3_checkpoint or Path(
            "artifacts/checkpoints/milestone_08c/M3_logk_fno_mixed/best_model.pt"
        )
        self.ensemble_dir = ensemble_dir or Path("artifacts/checkpoints")

        self.fvm_solver = HeatConductionSolver2D()
        self._load_normalizers_and_models()

    def _load_normalizers_and_models(self) -> None:
        """Load model weights and field normalizers."""

        # Helper for safe loading
        def _safe_torch_load(path: Path) -> dict[str, Any]:
            try:
                with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                    return torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                return torch.load(path, map_location="cpu", weights_only=False)

        # 1. Normalizer from M3
        if self.m3_ckpt_path.exists():
            m3_ckpt = _safe_torch_load(self.m3_ckpt_path)
            self.normalizer = FieldNormalizer.from_dict(m3_ckpt["normalizer"])
            self.temp_std = float(self.normalizer.t_stats.std)
            self.temp_mean = float(self.normalizer.t_stats.mean)
        else:
            self.normalizer = FieldNormalizer()
            self.temp_std = 18.2
            self.temp_mean = 334.8

        # 2. U1 model
        self.u1_model: HeteroscedasticFNO2D | None = None
        if self.u1_ckpt_path.exists():
            ckpt = _safe_torch_load(self.u1_ckpt_path)
            self.u1_model = HeteroscedasticFNO2D(modes_x=12, modes_y=12, width=32, depth=4)
            self.u1_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            self.u1_model.eval().to(self.device)

        # 3. U3 model
        self.u3_model: MCDropoutFNO2D | None = None
        if self.u3_ckpt_path.exists():
            ckpt = _safe_torch_load(self.u3_ckpt_path)
            self.u3_model = MCDropoutFNO2D(
                modes_x=12, modes_y=12, width=32, depth=4, dropout_p=0.10
            )
            self.u3_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
            self.u3_model.to(self.device)

        # 4. Production Ensemble (for R1)
        self.ensemble_models: list[FourierNeuralOperator2D] = []
        for seed in [42, 43, 44, 45, 46]:
            ckpt_p = self.ensemble_dir / f"scaled_fno_supervised_seed{seed}.pt"
            if ckpt_p.exists():
                m = FourierNeuralOperator2D(
                    in_channels=12, out_channels=1, modes_x=12, modes_y=12, width=32, depth=4
                )
                st = _safe_torch_load(ckpt_p)
                m.load_state_dict(st.get("model_state_dict", st))
                m.eval().to(self.device)
                self.ensemble_models.append(m)

    def prepare_input_tensor(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
    ) -> torch.Tensor:
        """Construct the 12-channel continuous physical input tensor."""
        res_y, res_x = k_field.shape
        x_norm = ((grid.x_mesh - grid.x_min) / grid.lx).astype(np.float32)
        y_norm = ((grid.y_mesh - grid.y_min) / grid.ly).astype(np.float32)

        x_arr = np.zeros((1, 12, res_y, res_x), dtype=np.float32)
        log_k = np.log(np.maximum(k_field, 1e-12))
        x_arr[0, 0] = ((log_k - 0.5) / 1.0).astype(np.float32)
        x_arr[0, 1] = self.normalizer.transform_q(q_field).astype(np.float32)
        x_arr[0, 2] = x_norm
        x_arr[0, 3] = y_norm

        weight_west = 1.0 - x_norm
        weight_east = x_norm
        weight_south = 1.0 - y_norm
        weight_north = y_norm

        # West
        if bcs.left.bc_type == BoundaryType.DIRICHLET:
            l_norm = self.normalizer.transform_temperature(bcs.left.value).astype(np.float32)
            x_arr[0, 4] = weight_west * l_norm[:, np.newaxis]
        else:
            l_norm = (bcs.left.value / 1000.0).astype(np.float32)
            x_arr[0, 6] = weight_west * l_norm[:, np.newaxis]

        # East
        if bcs.right.bc_type == BoundaryType.DIRICHLET:
            r_norm = self.normalizer.transform_temperature(bcs.right.value).astype(np.float32)
            x_arr[0, 5] = weight_east * r_norm[:, np.newaxis]
        else:
            r_norm = (bcs.right.value / 1000.0).astype(np.float32)
            x_arr[0, 7] = weight_east * r_norm[:, np.newaxis]

        # South
        if bcs.bottom.bc_type == BoundaryType.DIRICHLET:
            b_norm = self.normalizer.transform_temperature(bcs.bottom.value).astype(np.float32)
            x_arr[0, 8] = weight_south * b_norm[np.newaxis, :]
        else:
            b_norm = (bcs.bottom.value / 1000.0).astype(np.float32)
            x_arr[0, 10] = weight_south * b_norm[np.newaxis, :]

        # North
        if bcs.top.bc_type == BoundaryType.DIRICHLET:
            t_norm = self.normalizer.transform_temperature(bcs.top.value).astype(np.float32)
            x_arr[0, 9] = weight_north * t_norm[np.newaxis, :]
        else:
            t_norm = (bcs.top.value / 1000.0).astype(np.float32)
            x_arr[0, 11] = weight_north * t_norm[np.newaxis, :]

        return torch.from_numpy(x_arr)

    def route_inference(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
        policy: RoutingPolicy = RoutingPolicy.R5_ADAPTIVE_CASCADE,
    ) -> AdaptiveRoutingResult:
        """Execute end-to-end routing inference under the specified policy."""
        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        # -------------------------------------------------------------
        # Step 1: Input Validation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        if np.any(k_field <= 0.0) or not np.all(np.isfinite(k_field)):
            return AdaptiveRoutingResult(
                temperature=np.zeros_like(k_field),
                uncertainty=None,
                status=InferenceStatus.INVALID_INPUT,
                policy=policy,
                rejection_record=RejectionRecord(
                    code=RejectionCode.NON_POSITIVE_CONDUCTIVITY,
                    message="Conductivity field contains non-positive or non-finite values.",
                ),
                timings_ms={"t_validation": (time.perf_counter() - t0) * 1000.0},
            )
        timings["t_validation"] = (time.perf_counter() - t0) * 1000.0

        # R0: Direct FVM
        if policy == RoutingPolicy.R0_DIRECT_FVM:
            return self._solve_fvm_direct(grid, k_field, q_field, bcs, policy, timings, t_start)

        # -------------------------------------------------------------
        # Step 2: Pre-Inference Screening
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        c_k = float(np.max(k_field) / np.maximum(np.min(k_field), 1e-12))
        q_max = float(np.max(np.abs(q_field)))
        timings["t_prescreening"] = (time.perf_counter() - t0) * 1000.0

        if c_k > self.thresholds.tau_k_max:
            return self._fallback_fvm(
                grid,
                k_field,
                q_field,
                bcs,
                policy,
                timings,
                t_start,
                RejectionRecord(
                    code=RejectionCode.HIGH_INPUT_CONTRAST,
                    message=f"Conductivity contrast {c_k:.1f} exceeds threshold {self.thresholds.tau_k_max:.1f}",
                    observed_value=c_k,
                    threshold_value=self.thresholds.tau_k_max,
                ),
            )

        if q_max > self.thresholds.tau_q_max:
            return self._fallback_fvm(
                grid,
                k_field,
                q_field,
                bcs,
                policy,
                timings,
                t_start,
                RejectionRecord(
                    code=RejectionCode.HIGH_HEAT_SOURCE,
                    message=f"Heat source {q_max:.1f} exceeds threshold {self.thresholds.tau_q_max:.1f}",
                    observed_value=q_max,
                    threshold_value=self.thresholds.tau_q_max,
                ),
            )

        # -------------------------------------------------------------
        # Step 3: Feature Preparation
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        x_tensor = self.prepare_input_tensor(grid, k_field, q_field, bcs).to(self.device)
        timings["t_surrogate_prep"] = (time.perf_counter() - t0) * 1000.0

        # -------------------------------------------------------------
        # Step 4: Policy-Specific Inference Execution
        # -------------------------------------------------------------
        if policy == RoutingPolicy.R1_PRODUCTION_HYBRID:
            return self._route_r1_production_hybrid(
                grid, k_field, q_field, bcs, x_tensor, timings, t_start
            )

        elif policy == RoutingPolicy.R2_U1_SINGLE_PASS:
            return self._route_r2_u1(grid, k_field, q_field, bcs, x_tensor, timings, t_start)

        elif policy == RoutingPolicy.R3_U3_T3_MULTI_PASS:
            return self._route_u3_fixed_passes(
                grid,
                k_field,
                q_field,
                bcs,
                x_tensor,
                3,
                policy,
                self.thresholds.tau_u3_t3_unc,
                timings,
                t_start,
            )

        elif policy == RoutingPolicy.R4_U3_T5_MULTI_PASS:
            return self._route_u3_fixed_passes(
                grid,
                k_field,
                q_field,
                bcs,
                x_tensor,
                5,
                policy,
                self.thresholds.tau_u3_t5_unc,
                timings,
                t_start,
            )

        elif policy == RoutingPolicy.R5_ADAPTIVE_CASCADE:
            return self._route_r5_adaptive_cascade(
                grid, k_field, q_field, bcs, x_tensor, timings, t_start
            )

        else:
            raise ValueError(f"Unknown routing policy: {policy}")

    def _route_r1_production_hybrid(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
        x_tensor: torch.Tensor,
        timings: dict[str, float],
        t_start: float,
    ) -> AdaptiveRoutingResult:
        """R1: Existing production 5-member ensemble hybrid."""
        t0 = time.perf_counter()
        if not self.ensemble_models:
            # Fallback if checkpoints not present
            return self._fallback_fvm(
                grid, k_field, q_field, bcs, RoutingPolicy.R1_PRODUCTION_HYBRID, timings, t_start
            )

        preds = []
        with torch.no_grad():
            for m in self.ensemble_models:
                out = m(x_tensor)
                preds.append(self.normalizer.inverse_transform_temperature(out.cpu().numpy()[0, 0]))

        stack = np.stack(preds, axis=0)
        mean_t = np.mean(stack, axis=0)
        std_t = np.std(stack, axis=0, ddof=1)
        timings["t_surrogate_forward"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        max_disagreement = float(np.max(std_t))
        hb = compute_heat_balance(grid, mean_t, k_field, q_field, bcs)
        timings["t_diagnostics"] = (time.perf_counter() - t0) * 1000.0

        # Production thresholds
        if max_disagreement > 3.0:
            return self._fallback_fvm(
                grid,
                k_field,
                q_field,
                bcs,
                RoutingPolicy.R1_PRODUCTION_HYBRID,
                timings,
                t_start,
                RejectionRecord(
                    code=RejectionCode.HIGH_PEAK_DISAGREEMENT,
                    message=f"Ensemble disagreement {max_disagreement:.2f} K > 3.0 K",
                    observed_value=max_disagreement,
                    threshold_value=3.0,
                ),
            )

        if hb.relative_imbalance * 100.0 > 100.0:
            return self._fallback_fvm(
                grid,
                k_field,
                q_field,
                bcs,
                RoutingPolicy.R1_PRODUCTION_HYBRID,
                timings,
                t_start,
                RejectionRecord(
                    code=RejectionCode.EXCESSIVE_BOUNDARY_FLUX_IMBALANCE,
                    message=f"Relative flux imbalance {hb.relative_imbalance * 100.0:.1f}% > 100.0%",
                    observed_value=hb.relative_imbalance * 100.0,
                    threshold_value=100.0,
                ),
            )

        timings["t_total"] = (time.perf_counter() - t_start) * 1000.0
        return AdaptiveRoutingResult(
            temperature=mean_t,
            uncertainty=std_t,
            status=InferenceStatus.ACCEPTED,
            policy=RoutingPolicy.R1_PRODUCTION_HYBRID,
            diagnostics=hb,
            timings_ms=timings,
        )

    def _route_r2_u1(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
        x_tensor: torch.Tensor,
        timings: dict[str, float],
        t_start: float,
    ) -> AdaptiveRoutingResult:
        """R2: U1 Single-pass heteroscedastic uncertainty and diagnostics."""
        if self.u1_model is None:
            return self._fallback_fvm(
                grid, k_field, q_field, bcs, RoutingPolicy.R2_U1_SINGLE_PASS, timings, t_start
            )

        t0 = time.perf_counter()
        with torch.no_grad():
            mean_n, logvar_n = self.u1_model(x_tensor)
            mean_t = self.normalizer.inverse_transform_temperature(mean_n.cpu().numpy()[0, 0])
            std_t = np.sqrt(np.exp(logvar_n.cpu().numpy()[0, 0])) * self.temp_std
        timings["t_surrogate_forward"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        max_unc = float(np.max(std_t))
        hb = compute_heat_balance(grid, mean_t, k_field, q_field, bcs)
        timings["t_diagnostics"] = (time.perf_counter() - t0) * 1000.0

        if max_unc > self.thresholds.tau_u1_unc:
            return self._fallback_fvm(
                grid,
                k_field,
                q_field,
                bcs,
                RoutingPolicy.R2_U1_SINGLE_PASS,
                timings,
                t_start,
                RejectionRecord(
                    code=RejectionCode.HIGH_PEAK_DISAGREEMENT,
                    message=f"U1 uncertainty {max_unc:.2f} K exceeds threshold {self.thresholds.tau_u1_unc:.2f} K",
                    observed_value=max_unc,
                    threshold_value=self.thresholds.tau_u1_unc,
                ),
            )

        if hb.relative_imbalance * 100.0 > self.thresholds.tau_flux_imb_pct:
            return self._fallback_fvm(
                grid,
                k_field,
                q_field,
                bcs,
                RoutingPolicy.R2_U1_SINGLE_PASS,
                timings,
                t_start,
                RejectionRecord(
                    code=RejectionCode.EXCESSIVE_BOUNDARY_FLUX_IMBALANCE,
                    message=f"Relative flux imbalance {hb.relative_imbalance * 100.0:.1f}% > {self.thresholds.tau_flux_imb_pct:.1f}%",
                    observed_value=hb.relative_imbalance * 100.0,
                    threshold_value=self.thresholds.tau_flux_imb_pct,
                ),
            )

        timings["t_total"] = (time.perf_counter() - t_start) * 1000.0
        return AdaptiveRoutingResult(
            temperature=mean_t,
            uncertainty=std_t,
            status=InferenceStatus.ACCEPTED,
            policy=RoutingPolicy.R2_U1_SINGLE_PASS,
            diagnostics=hb,
            timings_ms=timings,
        )

    def _route_u3_fixed_passes(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
        x_tensor: torch.Tensor,
        passes: int,
        policy: RoutingPolicy,
        tau_unc: float,
        timings: dict[str, float],
        t_start: float,
    ) -> AdaptiveRoutingResult:
        """R3/R4: U3 with fixed T stochastic passes."""
        if self.u3_model is None:
            return self._fallback_fvm(grid, k_field, q_field, bcs, policy, timings, t_start)

        t0 = time.perf_counter()
        self.u3_model.train()
        preds = []
        with torch.no_grad():
            for _ in range(passes):
                out = self.u3_model(x_tensor)
                preds.append(self.normalizer.inverse_transform_temperature(out.cpu().numpy()[0, 0]))

        stack = np.stack(preds, axis=0)
        mean_t = np.mean(stack, axis=0)
        std_t = np.std(stack, axis=0, ddof=1)
        timings["t_surrogate_forward"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        max_unc = float(np.max(std_t))
        hb = compute_heat_balance(grid, mean_t, k_field, q_field, bcs)
        timings["t_diagnostics"] = (time.perf_counter() - t0) * 1000.0

        if max_unc > tau_unc:
            return self._fallback_fvm(
                grid,
                k_field,
                q_field,
                bcs,
                policy,
                timings,
                t_start,
                RejectionRecord(
                    code=RejectionCode.HIGH_PEAK_DISAGREEMENT,
                    message=f"U3 uncertainty {max_unc:.2f} K exceeds threshold {tau_unc:.2f} K",
                    observed_value=max_unc,
                    threshold_value=tau_unc,
                ),
            )

        if hb.relative_imbalance * 100.0 > self.thresholds.tau_flux_imb_pct:
            return self._fallback_fvm(
                grid,
                k_field,
                q_field,
                bcs,
                policy,
                timings,
                t_start,
                RejectionRecord(
                    code=RejectionCode.EXCESSIVE_BOUNDARY_FLUX_IMBALANCE,
                    message=f"Relative flux imbalance {hb.relative_imbalance * 100.0:.1f}% > {self.thresholds.tau_flux_imb_pct:.1f}%",
                    observed_value=hb.relative_imbalance * 100.0,
                    threshold_value=self.thresholds.tau_flux_imb_pct,
                ),
            )

        timings["t_total"] = (time.perf_counter() - t_start) * 1000.0
        return AdaptiveRoutingResult(
            temperature=mean_t,
            uncertainty=std_t,
            status=InferenceStatus.ACCEPTED,
            policy=policy,
            diagnostics=hb,
            timings_ms=timings,
        )

    def _route_r5_adaptive_cascade(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
        x_tensor: torch.Tensor,
        timings: dict[str, float],
        t_start: float,
    ) -> AdaptiveRoutingResult:
        """R5: Adaptive Cascade (Fast U1 single-pass -> ambiguous escalation to U3 -> FVM)."""
        if self.u1_model is None:
            return self._fallback_fvm(
                grid, k_field, q_field, bcs, RoutingPolicy.R5_ADAPTIVE_CASCADE, timings, t_start
            )

        # 1. First Tier: U1 single forward pass
        t0 = time.perf_counter()
        with torch.no_grad():
            mean_n, logvar_n = self.u1_model(x_tensor)
            u1_mean_t = self.normalizer.inverse_transform_temperature(mean_n.cpu().numpy()[0, 0])
            u1_std_t = np.sqrt(np.exp(logvar_n.cpu().numpy()[0, 0])) * self.temp_std
        timings["t_u1_forward"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        u1_max_unc = float(np.max(u1_std_t))
        hb_u1 = compute_heat_balance(grid, u1_mean_t, k_field, q_field, bcs)
        timings["t_u1_diagnostics"] = (time.perf_counter() - t0) * 1000.0

        # Tier 1 Gate A: Clearly Safe -> Accept immediately
        if (
            u1_max_unc <= self.thresholds.tau_u1_cascade_safe
            and hb_u1.relative_imbalance * 100.0 <= self.thresholds.tau_flux_imb_pct
        ):
            timings["t_total"] = (time.perf_counter() - t_start) * 1000.0
            return AdaptiveRoutingResult(
                temperature=u1_mean_t,
                uncertainty=u1_std_t,
                status=InferenceStatus.ACCEPTED,
                policy=RoutingPolicy.R5_ADAPTIVE_CASCADE,
                escalated_to_u3=False,
                diagnostics=hb_u1,
                timings_ms=timings,
            )

        # Tier 1 Gate B: Clearly Unsafe -> Direct to FVM (skip U3)
        if u1_max_unc > self.thresholds.tau_u1_cascade_ambiguous:
            return self._fallback_fvm(
                grid,
                k_field,
                q_field,
                bcs,
                RoutingPolicy.R5_ADAPTIVE_CASCADE,
                timings,
                t_start,
                RejectionRecord(
                    code=RejectionCode.HIGH_PEAK_DISAGREEMENT,
                    message=f"U1 uncertainty {u1_max_unc:.2f} K clearly unsafe (> {self.thresholds.tau_u1_cascade_ambiguous:.2f} K)",
                    observed_value=u1_max_unc,
                    threshold_value=self.thresholds.tau_u1_cascade_ambiguous,
                ),
            )

        # Tier 2: Ambiguous Region -> Escalate to U3 (T=5 stochastic passes)
        if self.u3_model is None:
            return self._fallback_fvm(
                grid, k_field, q_field, bcs, RoutingPolicy.R5_ADAPTIVE_CASCADE, timings, t_start
            )

        t0 = time.perf_counter()
        self.u3_model.train()
        preds = []
        with torch.no_grad():
            for _ in range(5):
                out = self.u3_model(x_tensor)
                preds.append(self.normalizer.inverse_transform_temperature(out.cpu().numpy()[0, 0]))

        stack = np.stack(preds, axis=0)
        u3_mean_t = np.mean(stack, axis=0)
        u3_std_t = np.std(stack, axis=0, ddof=1)
        timings["t_u3_forward"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        u3_max_unc = float(np.max(u3_std_t))
        hb_u3 = compute_heat_balance(grid, u3_mean_t, k_field, q_field, bcs)
        timings["t_u3_diagnostics"] = (time.perf_counter() - t0) * 1000.0

        # Tier 2 Gate: U3 acceptance
        if (
            u3_max_unc <= self.thresholds.tau_u3_t5_unc
            and hb_u3.relative_imbalance * 100.0 <= self.thresholds.tau_flux_imb_pct
        ):
            timings["t_total"] = (time.perf_counter() - t_start) * 1000.0
            return AdaptiveRoutingResult(
                temperature=u3_mean_t,
                uncertainty=u3_std_t,
                status=InferenceStatus.ACCEPTED,
                policy=RoutingPolicy.R5_ADAPTIVE_CASCADE,
                escalated_to_u3=True,
                diagnostics=hb_u3,
                timings_ms=timings,
            )

        # Otherwise fallback to FVM
        return self._fallback_fvm(
            grid,
            k_field,
            q_field,
            bcs,
            RoutingPolicy.R5_ADAPTIVE_CASCADE,
            timings,
            t_start,
            RejectionRecord(
                code=RejectionCode.HIGH_PEAK_DISAGREEMENT,
                message=f"U3 uncertainty after cascade {u3_max_unc:.2f} K exceeds threshold {self.thresholds.tau_u3_t5_unc:.2f} K",
                observed_value=u3_max_unc,
                threshold_value=self.thresholds.tau_u3_t5_unc,
            ),
            escalated_to_u3=True,
        )

    def _solve_fvm_direct(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
        policy: RoutingPolicy,
        timings: dict[str, float],
        t_start: float,
    ) -> AdaptiveRoutingResult:
        """R0: Direct FVM solve."""
        t0 = time.perf_counter()
        fvm_res = self.fvm_solver.solve(grid, k_field, q_field, bcs)
        timings["t_fvm_solve"] = (time.perf_counter() - t0) * 1000.0
        timings["t_total"] = (time.perf_counter() - t_start) * 1000.0

        hb = compute_heat_balance(grid, fvm_res.temperature, k_field, q_field, bcs)
        return AdaptiveRoutingResult(
            temperature=fvm_res.temperature,
            uncertainty=None,
            status=InferenceStatus.FALLBACK_SUCCEEDED,
            policy=policy,
            diagnostics=hb,
            timings_ms=timings,
        )

    def _fallback_fvm(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
        policy: RoutingPolicy,
        timings: dict[str, float],
        t_start: float,
        rejection_record: RejectionRecord | None = None,
        escalated_to_u3: bool = False,
    ) -> AdaptiveRoutingResult:
        """Execute fail-closed FVM fallback when neural gates reject."""
        t0 = time.perf_counter()
        fvm_res = self.fvm_solver.solve(grid, k_field, q_field, bcs)
        timings["t_fallback"] = (time.perf_counter() - t0) * 1000.0
        timings["t_total"] = (time.perf_counter() - t_start) * 1000.0

        hb = compute_heat_balance(grid, fvm_res.temperature, k_field, q_field, bcs)
        return AdaptiveRoutingResult(
            temperature=fvm_res.temperature,
            uncertainty=None,
            status=InferenceStatus.FALLBACK_SUCCEEDED,
            policy=policy,
            escalated_to_u3=escalated_to_u3,
            escalated_to_fvm=True,
            rejection_record=rejection_record,
            diagnostics=hb,
            timings_ms=timings,
        )


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Compute point estimate and Wilson score confidence interval.

    Args:
        k: Number of observed events (e.g. failures).
        n: Total number of trials.
        confidence: Desired confidence level (default: 0.95).

    Returns:
        tuple (p_hat, ci_lower, ci_upper)
    """
    if n == 0:
        return 0.0, 0.0, 1.0
    p_hat = float(k / n)
    z = 1.959963984540054  # 95% two-sided normal quantile
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + ((z**2) / (4 * n**2)))) / denom
    ci_lower = 0.0 if k == 0 else max(0.0, float(center - margin))
    ci_upper = 1.0 if k == n else min(1.0, float(center + margin))
    return p_hat, ci_lower, ci_upper


def calculate_wilson_interval(
    k: int, n: int, confidence: float = 0.95
) -> tuple[float | None, float | None, float | None]:
    """Calculate exact Wilson score confidence interval handling empty cohorts.

    Args:
        k: Number of events (e.g. failures).
        n: Total trials.
        confidence: Desired confidence level (default: 0.95).

    Returns:
        (p_hat, ci_lower, ci_upper), or (None, None, None) if n == 0.
    """
    if n == 0:
        return None, None, None
    p_hat = float(k / n)
    z = 1.959963984540054
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + ((z**2) / (4 * n**2)))) / denom
    ci_lower = 0.0 if k == 0 else max(0.0, float(center - margin))
    ci_upper = 1.0 if k == n else min(1.0, float(center + margin))
    return p_hat, ci_lower, ci_upper
