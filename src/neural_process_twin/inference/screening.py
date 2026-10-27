"""Safety screening and acceptance decision logic for hybrid inference.

Evaluates surrogate predictions against provisional ensemble disagreement thresholds,
input domain bounds, output finiteness/positivity, and optional configurable
physical residual gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_process_twin.inference.contracts import (
    InferenceOptions,
    PhysicalDiagnosticsOutput,
    RejectionCode,
    RejectionRecord,
)


@dataclass(frozen=True)
class ScreeningResult:
    """Outcome of multi-tiered surrogate safety screening.

    Attributes:
        accepted: True if all active screening gates passed without violation.
        rejection: Diagnostic audit record identifying the triggering rejection gate, if any.
        mean_disagreement_kelvin: Spatial arithmetic mean of ensemble disagreement s_T [K].
        max_disagreement_kelvin: Maximum single-cell ensemble disagreement s_T [K].
        k_contrast: Evaluated thermal conductivity contrast ratio k_max / k_min.
        q_max: Peak volumetric heat source rate [W/m^3].
    """

    accepted: bool
    rejection: RejectionRecord
    mean_disagreement_kelvin: float
    max_disagreement_kelvin: float
    k_contrast: float
    q_max: float


class SafetyScreener:
    """Evaluates surrogate predictions against multi-tiered safety gates."""

    def screen(
        self,
        predicted_temperature_k: np.ndarray,
        ensemble_std_k: np.ndarray,
        k_field: np.ndarray,
        q_field: np.ndarray,
        diagnostics: PhysicalDiagnosticsOutput | None,
        options: InferenceOptions,
    ) -> ScreeningResult:
        """Execute safety screening gates in conservative priority order.

        Gate 1: Output finiteness and physical positivity.
        Gate 2: Input domain bounds (conductivity contrast ratio and peak heat source).
        Gate 3: Ensemble disagreement thresholds (spatial mean and peak cell).
        Gate 4: Optional configurable PDE residual and boundary flux imbalance gates.

        Args:
            predicted_temperature_k: 2D predicted temperature field in Kelvin (ny, nx).
            ensemble_std_k: 2D predictive sample standard deviation in Kelvin (ny, nx).
            k_field: 2D conductivity field in W/(m K) (ny, nx).
            q_field: 2D heat source field in W/m^3 (ny, nx).
            diagnostics: Optional physical residual and boundary flux diagnostics.
            options: InferenceOptions specifying active thresholds.

        Returns:
            ScreeningResult indicating acceptance or rejection details.
        """
        # 1. Compute summary metrics with strict definitions
        # Disagreement aggregation:
        # - mean_s_t: exact spatial arithmetic mean over all ny * nx grid cells
        # - max_s_t: maximum single-cell standard deviation over all cells
        mean_s_t = float(np.mean(ensemble_std_k))
        max_s_t = float(np.max(ensemble_std_k))

        k_min = float(np.min(k_field))
        k_max = float(np.max(k_field))
        k_contrast = float(k_max / max(k_min, 1e-12))
        q_max_val = float(np.max(q_field))

        # Gate 1: Check surrogate output finiteness and positivity
        if not np.all(np.isfinite(predicted_temperature_k)) or not np.all(
            np.isfinite(ensemble_std_k)
        ):
            return ScreeningResult(
                accepted=False,
                rejection=RejectionRecord(
                    code=RejectionCode.SURROGATE_OUTPUT_NON_FINITE,
                    message="Surrogate produced NaN or Inf values in temperature or uncertainty field.",
                ),
                mean_disagreement_kelvin=mean_s_t,
                max_disagreement_kelvin=max_s_t,
                k_contrast=k_contrast,
                q_max=q_max_val,
            )

        t_min_pred = float(np.min(predicted_temperature_k))
        if t_min_pred <= 0.0:
            return ScreeningResult(
                accepted=False,
                rejection=RejectionRecord(
                    code=RejectionCode.SURROGATE_OUTPUT_NON_FINITE,
                    message=f"Surrogate predicted non-positive physical temperature: min {t_min_pred:.2f} K.",
                    triggering_metric="min_predicted_temperature",
                    observed_value=t_min_pred,
                    threshold_value=0.0,
                ),
                mean_disagreement_kelvin=mean_s_t,
                max_disagreement_kelvin=max_s_t,
                k_contrast=k_contrast,
                q_max=q_max_val,
            )

        # Gate 2: Input domain bounds
        if k_contrast > options.input_contrast_threshold:
            return ScreeningResult(
                accepted=False,
                rejection=RejectionRecord(
                    code=RejectionCode.HIGH_INPUT_CONTRAST,
                    message=(
                        f"Conductivity contrast ratio ({k_contrast:.2f}:1) exceeds "
                        f"training domain threshold ({options.input_contrast_threshold:.2f}:1)."
                    ),
                    triggering_metric="k_contrast_ratio",
                    observed_value=k_contrast,
                    threshold_value=options.input_contrast_threshold,
                ),
                mean_disagreement_kelvin=mean_s_t,
                max_disagreement_kelvin=max_s_t,
                k_contrast=k_contrast,
                q_max=q_max_val,
            )

        if q_max_val > options.input_q_max_threshold:
            return ScreeningResult(
                accepted=False,
                rejection=RejectionRecord(
                    code=RejectionCode.HIGH_HEAT_SOURCE,
                    message=(
                        f"Peak volumetric heat source ({q_max_val:.2f} W/m^3) exceeds "
                        f"training domain threshold ({options.input_q_max_threshold:.2f} W/m^3)."
                    ),
                    triggering_metric="q_max_w_m3",
                    observed_value=q_max_val,
                    threshold_value=options.input_q_max_threshold,
                ),
                mean_disagreement_kelvin=mean_s_t,
                max_disagreement_kelvin=max_s_t,
                k_contrast=k_contrast,
                q_max=q_max_val,
            )

        # Gate 3: Provisional ensemble disagreement screening
        # Mean spatial disagreement
        if mean_s_t > options.provisional_mean_disagreement_threshold_k:
            return ScreeningResult(
                accepted=False,
                rejection=RejectionRecord(
                    code=RejectionCode.HIGH_ENSEMBLE_DISAGREEMENT,
                    message=(
                        f"Spatial mean ensemble disagreement ({mean_s_t:.4f} K) exceeds "
                        f"provisional screening threshold ({options.provisional_mean_disagreement_threshold_k:.4f} K)."
                    ),
                    triggering_metric="spatial_mean_disagreement_k",
                    observed_value=mean_s_t,
                    threshold_value=options.provisional_mean_disagreement_threshold_k,
                ),
                mean_disagreement_kelvin=mean_s_t,
                max_disagreement_kelvin=max_s_t,
                k_contrast=k_contrast,
                q_max=q_max_val,
            )

        # Peak per-cell disagreement
        if max_s_t > options.provisional_max_disagreement_threshold_k:
            return ScreeningResult(
                accepted=False,
                rejection=RejectionRecord(
                    code=RejectionCode.HIGH_PEAK_DISAGREEMENT,
                    message=(
                        f"Peak cell ensemble disagreement ({max_s_t:.4f} K) exceeds "
                        f"provisional peak threshold ({options.provisional_max_disagreement_threshold_k:.4f} K)."
                    ),
                    triggering_metric="peak_cell_disagreement_k",
                    observed_value=max_s_t,
                    threshold_value=options.provisional_max_disagreement_threshold_k,
                ),
                mean_disagreement_kelvin=mean_s_t,
                max_disagreement_kelvin=max_s_t,
                k_contrast=k_contrast,
                q_max=q_max_val,
            )

        # Gate 4: Optional configurable PDE residual and boundary flux gates (if enabled)
        if diagnostics is not None:
            if (
                options.pde_residual_threshold_w is not None
                and diagnostics.pde_residual_l2_w > options.pde_residual_threshold_w
            ):
                return ScreeningResult(
                    accepted=False,
                    rejection=RejectionRecord(
                        code=RejectionCode.EXCESSIVE_PDE_RESIDUAL,
                        message=(
                            f"Discrete PDE residual L2 norm ({diagnostics.pde_residual_l2_w:.2f} W) "
                            f"exceeds threshold ({options.pde_residual_threshold_w:.2f} W)."
                        ),
                        triggering_metric="pde_residual_l2_w",
                        observed_value=diagnostics.pde_residual_l2_w,
                        threshold_value=options.pde_residual_threshold_w,
                    ),
                    mean_disagreement_kelvin=mean_s_t,
                    max_disagreement_kelvin=max_s_t,
                    k_contrast=k_contrast,
                    q_max=q_max_val,
                )

            if (
                options.relative_pde_residual_threshold is not None
                and diagnostics.pde_residual_relative > options.relative_pde_residual_threshold
            ):
                return ScreeningResult(
                    accepted=False,
                    rejection=RejectionRecord(
                        code=RejectionCode.EXCESSIVE_PDE_RESIDUAL,
                        message=(
                            f"Relative discrete PDE residual ({diagnostics.pde_residual_relative:.4f}) "
                            f"exceeds threshold ({options.relative_pde_residual_threshold:.4f})."
                        ),
                        triggering_metric="pde_residual_relative",
                        observed_value=diagnostics.pde_residual_relative,
                        threshold_value=options.relative_pde_residual_threshold,
                    ),
                    mean_disagreement_kelvin=mean_s_t,
                    max_disagreement_kelvin=max_s_t,
                    k_contrast=k_contrast,
                    q_max=q_max_val,
                )

            if (
                options.boundary_flux_imbalance_threshold_w is not None
                and diagnostics.absolute_flux_imbalance_w
                > options.boundary_flux_imbalance_threshold_w
            ):
                return ScreeningResult(
                    accepted=False,
                    rejection=RejectionRecord(
                        code=RejectionCode.EXCESSIVE_BOUNDARY_FLUX_IMBALANCE,
                        message=(
                            f"Absolute boundary flux imbalance ({diagnostics.absolute_flux_imbalance_w:.2f} W) "
                            f"exceeds threshold ({options.boundary_flux_imbalance_threshold_w:.2f} W)."
                        ),
                        triggering_metric="absolute_flux_imbalance_w",
                        observed_value=diagnostics.absolute_flux_imbalance_w,
                        threshold_value=options.boundary_flux_imbalance_threshold_w,
                    ),
                    mean_disagreement_kelvin=mean_s_t,
                    max_disagreement_kelvin=max_s_t,
                    k_contrast=k_contrast,
                    q_max=q_max_val,
                )

            if (
                options.relative_flux_imbalance_threshold is not None
                and diagnostics.relative_flux_imbalance > options.relative_flux_imbalance_threshold
            ):
                return ScreeningResult(
                    accepted=False,
                    rejection=RejectionRecord(
                        code=RejectionCode.EXCESSIVE_BOUNDARY_FLUX_IMBALANCE,
                        message=(
                            f"Relative boundary flux imbalance ({diagnostics.relative_flux_imbalance:.4f}) "
                            f"exceeds threshold ({options.relative_flux_imbalance_threshold:.4f})."
                        ),
                        triggering_metric="relative_flux_imbalance",
                        observed_value=diagnostics.relative_flux_imbalance,
                        threshold_value=options.relative_flux_imbalance_threshold,
                    ),
                    mean_disagreement_kelvin=mean_s_t,
                    max_disagreement_kelvin=max_s_t,
                    k_contrast=k_contrast,
                    q_max=q_max_val,
                )

        # All gates passed successfully
        return ScreeningResult(
            accepted=True,
            rejection=RejectionRecord(
                code=RejectionCode.NONE, message="All safety screening gates passed."
            ),
            mean_disagreement_kelvin=mean_s_t,
            max_disagreement_kelvin=max_s_t,
            k_contrast=k_contrast,
            q_max=q_max_val,
        )
