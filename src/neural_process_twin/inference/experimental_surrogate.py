"""Experimental Multi-Resolution Neural Surrogate Service for Milestone 09.

Provides isolated execution of the frozen A3 RefinedFNO2D surrogate and optional
Selective Boundary Jacobi physics smoother (A4). Strictly separates experimental neural
approximations from verified numerical FVM solutions.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neural_process_twin.data.continuous_encoding import encode_continuous_physical_inputs
from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.refined_fno import RefinedFNO2D
from neural_process_twin.solvers.assembly import LinearSystem, assemble_steady_heat_system
from neural_process_twin.solvers.boundary import BoundaryConditions
from neural_process_twin.solvers.diagnostics import EnergyBalanceDiagnostics, compute_heat_balance
from neural_process_twin.solvers.grid import Grid2D

logger = logging.getLogger(__name__)

# Authoritative checkpoint paths and constants
DEFAULT_A3_CHECKPOINT_PATH = Path(
    "artifacts/checkpoints/milestone_08f/A3_hybrid_refined_fno/best_model.pt"
)
VALIDATED_RESOLUTIONS = {32, 64, 128, 256}


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    if not path.is_file():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ExperimentalSurrogateResult:
    """Result of experimental neural surrogate execution with optional diagnostics."""

    temperature: np.ndarray
    resolution: tuple[int, int]
    physics_corrected: bool
    inference_time_ms: float
    correction_time_ms: float
    total_time_ms: float
    pde_residual_l2_w: float | None
    pde_residual_linf_w: float | None
    pde_residual_relative: float | None
    jacobi_residual_max_k: float | None
    heat_balance: EnergyBalanceDiagnostics | None
    warnings: list[str]
    disclaimers: list[str]
    provenance: dict[str, Any]


class ExperimentalSurrogateService:
    """Service for running the frozen A3 RefinedFNO2D model with optional physics smoothing."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_A3_CHECKPOINT_PATH,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.model: RefinedFNO2D | None = None
        self.normalizer: FieldNormalizer | None = None
        self.log_k_mean: float = 0.8623669147491455
        self.log_k_std: float = 0.4588298201560974
        self.checkpoint_sha256: str = ""
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load and verify frozen A3 RefinedFNO2D checkpoint."""
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"A3 RefinedFNO2D checkpoint not found at {self.checkpoint_path}. "
                "Ensure Milestone 08F training artifacts are present."
            )

        self.checkpoint_sha256 = compute_file_sha256(self.checkpoint_path)
        logger.info(
            "Loading A3 RefinedFNO2D checkpoint from %s (SHA-256: %s)",
            self.checkpoint_path,
            self.checkpoint_sha256,
        )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Normalization parameters from authoritative training fit
        if "normalizer" in checkpoint:
            self.normalizer = FieldNormalizer.from_dict(checkpoint["normalizer"])
        else:
            raise KeyError("Checkpoint missing 'normalizer' metadata.")

        self.log_k_mean = float(checkpoint.get("log_k_mean", self.log_k_mean))
        self.log_k_std = float(checkpoint.get("log_k_std", self.log_k_std))

        # Model architecture
        self.model = RefinedFNO2D(
            in_channels=12,
            out_channels=1,
            modes_x=12,
            modes_y=12,
            width=32,
            depth=4,
            refinement_hidden=32,
            refinement_layers=3,
            activation="gelu",
        ).to(self.device)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        logger.info("A3 RefinedFNO2D loaded and frozen in eval mode.")

    def apply_selective_boundary_jacobi(
        self,
        t_init: np.ndarray,
        system: LinearSystem,
        grid: Grid2D,
        ring_width: int = 2,
        omega: float = 0.67,
        num_iters: int = 2,
    ) -> np.ndarray:
        """Apply selective boundary Jacobi smoothing (A4 physics correction)."""
        ny, nx = grid.ny, grid.nx
        t_flat = t_init.ravel().copy()

        bnd_mask = np.zeros((ny, nx), dtype=bool)
        rw = min(ring_width, ny // 2)
        bnd_mask[:rw, :] = True
        bnd_mask[-rw:, :] = True
        bnd_mask[:, :rw] = True
        bnd_mask[:, -rw:] = True
        bnd_indices = np.where(bnd_mask.ravel())[0]

        a_csr = system.A.tocsr()
        diag_a = a_csr.diagonal()[bnd_indices]
        inv_diag = np.where(np.abs(diag_a) > 1e-12, 1.0 / diag_a, 0.0)

        for _ in range(num_iters):
            r_bnd = system.b[bnd_indices] - a_csr[bnd_indices, :].dot(t_flat)
            t_flat[bnd_indices] += omega * inv_diag * r_bnd

        return t_flat.reshape((ny, nx))

    def predict(
        self,
        grid: Grid2D,
        k_field: np.ndarray,
        q_field: np.ndarray,
        bcs: BoundaryConditions,
        apply_physics_correction: bool = False,
        compute_diagnostics: bool = True,
    ) -> ExperimentalSurrogateResult:
        """Run experimental neural surrogate inference with physical diagnostics.

        Args:
            grid: 2D Cartesian spatial grid.
            k_field: Thermal conductivity field [W/(m K)].
            q_field: Volumetric heat source field [W/m^3].
            bcs: Domain boundary conditions.
            apply_physics_correction: Whether to apply Selective Boundary Jacobi (A4).
            compute_diagnostics: Whether to assemble system and compute PDE residual / heat balance.

        Returns:
            ExperimentalSurrogateResult containing predicted temperature and scientific diagnostics.

        Raises:
            ValueError: If grid resolution is not in VALIDATED_RESOLUTIONS.
        """
        if grid.nx != grid.ny or grid.nx not in VALIDATED_RESOLUTIONS:
            raise ValueError(
                f"Resolution {grid.nx}x{grid.ny} is not supported by the experimental surrogate. "
                f"Validated multi-resolution grids are strictly: {sorted(VALIDATED_RESOLUTIONS)}. "
                "Use Verified Numerical Mode (FVM) for arbitrary grid geometries."
            )

        if self.model is None or self.normalizer is None:
            raise RuntimeError("Experimental surrogate model is not initialized.")

        t_start = time.perf_counter()

        # 1. Input encoding
        x_in = encode_continuous_physical_inputs(
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            grid=grid,
            normalizer=self.normalizer,
            use_log_k=True,
            log_k_mean=self.log_k_mean,
            log_k_std=self.log_k_std,
        )
        x_tensor = torch.from_numpy(x_in).unsqueeze(0).to(self.device)

        # 2. Neural Forward Pass
        t_inf_start = time.perf_counter()
        with torch.no_grad():
            out_norm = self.model(x_tensor)
            t_pred_raw = self.normalizer.inverse_transform_temperature(
                out_norm.cpu().numpy()[0, 0]
            ).astype(np.float64)
        t_inference_ms = (time.perf_counter() - t_inf_start) * 1000.0

        # 3. Optional Physics Correction (Selective Boundary Jacobi)
        t_corr_ms = 0.0
        sys = None
        if apply_physics_correction or compute_diagnostics:
            sys = assemble_steady_heat_system(grid, k_field, q_field, bcs)

        if apply_physics_correction and sys is not None:
            t_corr_start = time.perf_counter()
            t_final = self.apply_selective_boundary_jacobi(
                t_init=t_pred_raw,
                system=sys,
                grid=grid,
                ring_width=2,
                omega=0.67,
                num_iters=2,
            )
            t_corr_ms = (time.perf_counter() - t_corr_start) * 1000.0
        else:
            t_final = t_pred_raw

        t_total_ms = (time.perf_counter() - t_start) * 1000.0

        # 4. Independent Physical Diagnostics
        r_l2 = None
        r_linf = None
        r_rel = None
        jac_res_max_k = None
        heat_balance = None

        if compute_diagnostics and sys is not None:
            a_csr = sys.A.tocsr()
            t_flat = t_final.ravel()
            r = sys.b - a_csr.dot(t_flat)
            r_l2 = float(np.linalg.norm(r))
            r_linf = float(np.max(np.abs(r)))
            b_norm = float(np.linalg.norm(sys.b))
            r_rel = r_l2 / max(b_norm, 1e-12)

            diag_a = a_csr.diagonal()
            inv_diag = np.where(np.abs(diag_a) > 1e-12, 1.0 / diag_a, 0.0)
            jac_res_max_k = float(np.max(np.abs(inv_diag * r)))

            heat_balance = compute_heat_balance(grid, t_final, k_field, q_field, bcs)

        # Explicit warnings and scientific disclaimers
        warnings = [
            "EXPERIMENTAL SURROGATE: This prediction was generated by an experimental neural operator "
            f"({'A4 Refined FNO + Selective Jacobi' if apply_physics_correction else 'A3 Refined FNO'}).",
            "UNCERTIFIED ERROR: Empirical audits prove that 36% to 80% of neural surrogate predictions "
            "exceed the engineering tolerance of MaxAE <= 10 K on fine grids. Use Verified Numerical Mode "
            "for certified engineering calculations.",
        ]

        disclaimers = [
            "The Pointwise Jacobi Residual is a numerical error estimator, NOT a mathematically certified upper bound.",
            "Rigorous mathematical error certification for neural operators was proven in Milestone 08H to be "
            "computationally self-defeating (slower than direct FVM).",
        ]

        provenance = {
            "model_architecture": "RefinedFNO2D (Global Spectral FNO + Local 3-Layer Conv Residual)",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "physics_correction": "Selective Boundary Jacobi (omega=0.67, ring=2, iters=2)"
            if apply_physics_correction
            else "None",
            "normalizer": {
                "log_k_mean": self.log_k_mean,
                "log_k_std": self.log_k_std,
                "t_mean": self.normalizer.t_stats.mean if self.normalizer.t_stats else None,
                "t_std": self.normalizer.t_stats.std if self.normalizer.t_stats else None,
            },
        }

        return ExperimentalSurrogateResult(
            temperature=t_final,
            resolution=(grid.nx, grid.ny),
            physics_corrected=apply_physics_correction,
            inference_time_ms=t_inference_ms,
            correction_time_ms=t_corr_ms,
            total_time_ms=t_total_ms,
            pde_residual_l2_w=r_l2,
            pde_residual_linf_w=r_linf,
            pde_residual_relative=r_rel,
            jacobi_residual_max_k=jac_res_max_k,
            heat_balance=heat_balance,
            warnings=warnings,
            disclaimers=disclaimers,
            provenance=provenance,
        )
