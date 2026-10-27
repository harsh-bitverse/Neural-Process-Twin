"""Deep Ensemble surrogate architecture for spatial field uncertainty quantification.

Implements an ensemble of independently initialized and trained neural operator / CNN
models. Produces predictive mean fields and unbiased sample standard deviation fields
(Kelvin) representing model epistemic disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from neural_process_twin.training.checkpoint import load_checkpoint


@dataclass(frozen=True)
class EnsemblePrediction:
    """Structured container for spatial ensemble predictions and uncertainty fields.

    Attributes:
        mean_kelvin: Predictive mean temperature field in Kelvin [K] of shape (B, 1, ny, nx).
        std_kelvin: Per-cell predictive sample standard deviation in Kelvin [K] of shape (B, 1, ny, nx).
        members_kelvin: Individual member temperature predictions of shape (M, B, 1, ny, nx).
        mean_normalized: Predictive mean in normalized model output space of shape (B, 1, ny, nx).
        std_normalized: Predictive sample std in normalized model output space of shape (B, 1, ny, nx).
        num_members: Number of active ensemble members M.
    """

    mean_kelvin: torch.Tensor
    std_kelvin: torch.Tensor
    members_kelvin: torch.Tensor
    mean_normalized: torch.Tensor
    std_normalized: torch.Tensor
    num_members: int

    @property
    def per_sample_mean_uncertainty(self) -> torch.Tensor:
        """Spatial average of predictive standard deviation for each sample in batch (B,)."""
        return torch.mean(self.std_kelvin, dim=(-3, -2, -1))

    @property
    def per_sample_max_uncertainty(self) -> torch.Tensor:
        """Spatial maximum of predictive standard deviation for each sample in batch (B,)."""
        return torch.amax(self.std_kelvin, dim=(-3, -2, -1))


class DeepEnsemble(nn.Module):
    """Ensemble of independently trained surrogate models.

    Computes predictive mean and unbiased sample standard deviation:
        \\bar{T}(x, y) = \\frac{1}{M} \\sum_{m=1}^M T_m(x, y)
        s_T(x, y) = \\sqrt{\\frac{1}{M-1} \\sum_{m=1}^M (T_m(x, y) - \\bar{T}(x, y))^2}

    Note:
        Sample standard deviation s_T(x, y) quantifies epistemic disagreement across
        different optimization trajectories and local minima. In finite ensembles (e.g. M=5),
        disagreement is not a complete Bayesian posterior or a calibrated prediction interval.
    """

    def __init__(
        self,
        models: list[nn.Module],
        member_names: list[str] | None = None,
        normalizer_stats: dict[str, Any] | None = None,
    ) -> None:
        """Initialize DeepEnsemble with a list of instantiated PyTorch models.

        Args:
            models: Non-empty list of trained surrogate models (M >= 2 for std dev).
            member_names: Optional string identifiers for ensemble members.
            normalizer_stats: Optional dict containing 't_mean' and 't_std' for Kelvin scaling.
        """
        super().__init__()
        if not models:
            raise ValueError("DeepEnsemble requires at least one model member.")
        if len(models) < 2:
            raise ValueError(
                f"DeepEnsemble requires at least 2 members to compute sample standard deviation, got {len(models)}."
            )

        self.members = nn.ModuleList(models)
        self.num_members = len(models)
        self.member_names = (
            list(member_names) if member_names else [f"member_{i}" for i in range(len(models))]
        )
        if len(self.member_names) != self.num_members:
            raise ValueError(
                f"Length of member_names ({len(self.member_names)}) must match models ({self.num_members})."
            )

        self.normalizer_stats = normalizer_stats or {}

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: list[str | Path],
        device: str | torch.device = "cpu",
        strict: bool = True,
    ) -> DeepEnsemble:
        """Instantiate a DeepEnsemble directly from a list of checkpoint file paths.

        Args:
            checkpoint_paths: List of paths to saved checkpoint files (e.g. 'best_model.pt').
            device: Computing device to map model tensors to.
            strict: Whether to enforce strict key matching during state restoration.

        Returns:
            Initialized DeepEnsemble ready for inference.
        """
        if not checkpoint_paths:
            raise ValueError("No checkpoint paths provided.")
        if len(checkpoint_paths) < 2:
            raise ValueError(
                f"DeepEnsemble requires at least 2 checkpoints for uncertainty estimation, got {len(checkpoint_paths)}."
            )

        models: list[nn.Module] = []
        member_names: list[str] = []
        normalizer_stats: dict[str, Any] = {}

        for p in checkpoint_paths:
            path_obj = Path(p)
            if not path_obj.is_file():
                raise FileNotFoundError(f"Checkpoint file not found: {path_obj}")

            ckpt_data = load_checkpoint(path_obj, device=device, strict=strict)
            model = ckpt_data["model"]
            model.eval()
            models.append(model)
            member_names.append(path_obj.parent.name)

            if not normalizer_stats and "normalizer_stats" in ckpt_data:
                normalizer_stats = ckpt_data["normalizer_stats"]

        return cls(
            models=models,
            member_names=member_names,
            normalizer_stats=normalizer_stats,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute ensemble predictive mean in normalized space.

        Enables DeepEnsemble to serve as a drop-in replacement for any single surrogate
        model in standard evaluation and benchmarking routines.

        Args:
            x: Input tensor of shape (batch, channels, ny, nx).

        Returns:
            Predictive mean tensor \\bar{y} of shape (batch, 1, ny, nx).
        """
        preds = [m(x) for m in self.members]
        stacked = torch.stack(preds, dim=0)
        return torch.mean(stacked, dim=0)

    def predict(
        self,
        x: torch.Tensor,
        t_mean: float | None = None,
        t_std: float | None = None,
    ) -> EnsemblePrediction:
        """Perform full ensemble inference, computing predictive mean and standard deviation.

        Args:
            x: Input tensor of shape (B, channels, ny, nx).
            t_mean: Explicit mean temperature scale [K]. If None, reads from normalizer_stats.
            t_std: Explicit std temperature scale [K]. If None, reads from normalizer_stats.

        Returns:
            EnsemblePrediction with mean, sample std, and member predictions in Kelvin.
        """
        # Resolve Kelvin scaling factors
        scale_mean = t_mean
        scale_std = t_std
        if scale_mean is None or scale_std is None:
            stats = self.normalizer_stats.get("temperature", {}) or self.normalizer_stats.get(
                "t_stats", {}
            )
            if isinstance(stats, dict):
                scale_mean = stats.get("mean", 0.0) if scale_mean is None else scale_mean
                scale_std = stats.get("std", 1.0) if scale_std is None else scale_std
            else:
                scale_mean = getattr(stats, "mean", 0.0) if scale_mean is None else scale_mean
                scale_std = getattr(stats, "std", 1.0) if scale_std is None else scale_std

        scale_mean_f = float(scale_mean if scale_mean is not None else 0.0)
        scale_std_f = float(scale_std if scale_std is not None else 1.0)

        was_training = self.training
        self.eval()

        with torch.no_grad():
            preds_norm = [m(x) for m in self.members]
            # Shape: (M, B, 1, ny, nx)
            stacked_norm = torch.stack(preds_norm, dim=0)

            # Normalized statistics
            mean_norm = torch.mean(stacked_norm, dim=0)
            # Unbiased sample standard deviation (ddof=1, denominator M-1)
            std_norm = torch.std(stacked_norm, dim=0, correction=1)

            # Physical Kelvin fields
            members_k = stacked_norm * scale_std_f + scale_mean_f
            mean_k = mean_norm * scale_std_f + scale_mean_f
            std_k = std_norm * scale_std_f  # Note: std scales linearly with scale_std

        if was_training:
            self.train()

        return EnsemblePrediction(
            mean_kelvin=mean_k,
            std_kelvin=std_k,
            members_kelvin=members_k,
            mean_normalized=mean_norm,
            std_normalized=std_norm,
            num_members=self.num_members,
        )
