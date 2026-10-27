"""Physical input validation and domain compatibility checks for hybrid inference.

Provides comprehensive validation of domain geometries, spatial fields, boundary
conditions, physical positivity, and surrogate architectural constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neural_process_twin.inference.contracts import (
    InferenceRequest,
    RejectionCode,
    RejectionRecord,
)
from neural_process_twin.solvers.boundary import (
    BoundaryConditions,
    BoundaryType,
)

# Canonical grid resolution supported by the trained FNO-2D surrogate models
SURROGATE_REQUIRED_NX = 32
SURROGATE_REQUIRED_NY = 32


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of physical and dimensional problem validation.

    Attributes:
        is_valid: True if problem is mathematically and physically well-posed.
        surrogate_compatible: True if problem matches the surrogate's fixed grid geometry.
        rejection: Diagnostic audit record if validation failed or surrogate is bypassed.
        evaluated_k: 2D conductivity field (ny, nx) in W/(m K), if valid.
        evaluated_q: 2D volumetric heat source field (ny, nx) in W/m^3, if valid.
    """

    is_valid: bool
    surrogate_compatible: bool
    rejection: RejectionRecord
    evaluated_k: np.ndarray | None = None
    evaluated_q: np.ndarray | None = None


class InputValidator:
    """Validates input requests against physical constraints and solver/surrogate requirements."""

    def __init__(
        self,
        surrogate_nx: int = SURROGATE_REQUIRED_NX,
        surrogate_ny: int = SURROGATE_REQUIRED_NY,
    ) -> None:
        """Initialize input validator.

        Args:
            surrogate_nx: Required x-resolution for the trained surrogate (default 32).
            surrogate_ny: Required y-resolution for the trained surrogate (default 32).
        """
        self.surrogate_nx = surrogate_nx
        self.surrogate_ny = surrogate_ny

    def validate(self, request: InferenceRequest) -> ValidationResult:
        """Perform comprehensive physical, dimensional, and boundary-condition validation.

        Args:
            request: InferenceRequest containing grid, k, q, BCs, and options.

        Returns:
            ValidationResult indicating validity, surrogate compatibility, and fields.
        """
        grid = request.grid
        options = request.options

        # 1. Evaluate and validate fields on the grid
        try:
            k_field = grid.evaluate_field(request.conductivity, field_name="conductivity")
        except (ValueError, TypeError) as err:
            return ValidationResult(
                is_valid=False,
                surrogate_compatible=False,
                rejection=RejectionRecord(
                    code=RejectionCode.NON_FINITE_VALUES,
                    message=f"Failed to evaluate conductivity field: {err}",
                ),
            )

        try:
            q_field = grid.evaluate_field(request.source, field_name="source")
        except (ValueError, TypeError) as err:
            return ValidationResult(
                is_valid=False,
                surrogate_compatible=False,
                rejection=RejectionRecord(
                    code=RejectionCode.NON_FINITE_VALUES,
                    message=f"Failed to evaluate source field: {err}",
                ),
            )

        # 2. Check strict positivity of thermal conductivity (k > 0)
        k_min = float(np.min(k_field))
        if k_min <= 0.0 or not np.isfinite(k_min):
            return ValidationResult(
                is_valid=False,
                surrogate_compatible=False,
                rejection=RejectionRecord(
                    code=RejectionCode.NON_POSITIVE_CONDUCTIVITY,
                    message=f"Thermal conductivity must be strictly positive (k > 0); min observed: {k_min:.4e}",
                    triggering_metric="min_conductivity",
                    observed_value=k_min,
                    threshold_value=0.0,
                ),
            )

        # 3. Check for non-finite values in fields
        if not np.all(np.isfinite(k_field)):
            return ValidationResult(
                is_valid=False,
                surrogate_compatible=False,
                rejection=RejectionRecord(
                    code=RejectionCode.NON_FINITE_VALUES,
                    message="Conductivity field contains NaN or Inf values.",
                ),
            )
        if not np.all(np.isfinite(q_field)):
            return ValidationResult(
                is_valid=False,
                surrogate_compatible=False,
                rejection=RejectionRecord(
                    code=RejectionCode.NON_FINITE_VALUES,
                    message="Heat source field contains NaN or Inf values.",
                ),
            )

        # 4. Validate boundary conditions
        bcs = request.boundary_conditions
        if not isinstance(bcs, BoundaryConditions):
            return ValidationResult(
                is_valid=False,
                surrogate_compatible=False,
                rejection=RejectionRecord(
                    code=RejectionCode.UNSUPPORTED_BOUNDARY_CONDITION,
                    message=f"Expected BoundaryConditions instance, got {type(bcs).__name__}.",
                ),
            )

        # Evaluate BC values across faces and check finiteness
        try:
            bc_evals = bcs.evaluate_all(grid)
        except Exception as err:
            return ValidationResult(
                is_valid=False,
                surrogate_compatible=False,
                rejection=RejectionRecord(
                    code=RejectionCode.NON_FINITE_VALUES,
                    message=f"Failed to evaluate boundary conditions: {err}",
                ),
            )

        for side_name, arr in bc_evals.items():
            if not np.all(np.isfinite(arr)):
                return ValidationResult(
                    is_valid=False,
                    surrogate_compatible=False,
                    rejection=RejectionRecord(
                        code=RejectionCode.NON_FINITE_VALUES,
                        message=f"Boundary condition for side '{side_name}' contains NaN or Inf values.",
                    ),
                )

        # Check Dirichlet temperatures are strictly positive
        for side in (bcs.left, bcs.right, bcs.bottom, bcs.top):
            if side.bc_type == BoundaryType.DIRICHLET:
                vals = bc_evals[side.side.value]
                t_bnd_min = float(np.min(vals))
                if t_bnd_min <= 0.0:
                    return ValidationResult(
                        is_valid=False,
                        surrogate_compatible=False,
                        rejection=RejectionRecord(
                            code=RejectionCode.NON_POSITIVE_CONDUCTIVITY,
                            message=f"Dirichlet boundary temperature on side '{side.side.value}' must be > 0 K; min: {t_bnd_min:.2f} K",
                            triggering_metric=f"dirichlet_min_t_{side.side.value}",
                            observed_value=t_bnd_min,
                            threshold_value=0.0,
                        ),
                    )

        # Check well-posedness: at least one Dirichlet OR gauge pin
        has_dirichlet = any(
            side.bc_type == BoundaryType.DIRICHLET
            for side in (bcs.left, bcs.right, bcs.bottom, bcs.top)
        )
        if not has_dirichlet and request.gauge_pin is None:
            return ValidationResult(
                is_valid=False,
                surrogate_compatible=False,
                rejection=RejectionRecord(
                    code=RejectionCode.PURE_NEUMANN_WITHOUT_GAUGE,
                    message="All boundaries are Neumann but no gauge_pin was provided (singular system).",
                ),
            )

        # 5. Check surrogate grid compatibility
        surrogate_compatible = grid.nx == self.surrogate_nx and grid.ny == self.surrogate_ny

        if not surrogate_compatible:
            # If fallback is enabled, the request is physically valid for the solver, but bypassed by surrogate
            rejection = RejectionRecord(
                code=RejectionCode.INCOMPATIBLE_GRID_DIMENSIONS,
                message=(
                    f"Grid resolution ({grid.nx}x{grid.ny}) does not match surrogate "
                    f"requirement ({self.surrogate_nx}x{self.surrogate_ny})."
                ),
                triggering_metric="grid_resolution",
                observed_value=float(grid.nx),
                threshold_value=float(self.surrogate_nx),
            )
            # If fallback is disabled, grid incompatibility is a fatal validation error
            is_valid = options.enable_fallback
            return ValidationResult(
                is_valid=is_valid,
                surrogate_compatible=False,
                rejection=rejection,
                evaluated_k=k_field if is_valid else None,
                evaluated_q=q_field if is_valid else None,
            )

        return ValidationResult(
            is_valid=True,
            surrogate_compatible=True,
            rejection=RejectionRecord(code=RejectionCode.NONE, message="Validation passed."),
            evaluated_k=k_field,
            evaluated_q=q_field,
        )
