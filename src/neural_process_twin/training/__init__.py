"""Training pipeline and loss formulations for Neural Process Twin.

This package manages surrogate model optimization, loss formulation,
validation loops, checkpoint persistence, and training diagnostics:
    - RelativeL2Loss and build_loss for supervised continuous field objectives.
    - save_checkpoint and load_checkpoint for reproducible experiment states.
    - Trainer class coordinating gradient clipping, early stopping, and schedulers.
"""

from __future__ import annotations

from neural_process_twin.training.checkpoint import load_checkpoint, save_checkpoint
from neural_process_twin.training.face_consistent_loss import (
    FaceConsistentBoundaryLoss,
    FaceConsistentPhysicsLoss,
)
from neural_process_twin.training.loss import RelativeL2Loss, build_loss
from neural_process_twin.training.physics_loss import (
    BoundaryConditionLoss,
    DimensionlessPDELoss,
    DiscretePDEResidual,
    PhysicsAwareLoss,
)
from neural_process_twin.training.trainer import Trainer

__all__ = [
    "BoundaryConditionLoss",
    "DimensionlessPDELoss",
    "DiscretePDEResidual",
    "FaceConsistentBoundaryLoss",
    "FaceConsistentPhysicsLoss",
    "PhysicsAwareLoss",
    "RelativeL2Loss",
    "Trainer",
    "build_loss",
    "load_checkpoint",
    "save_checkpoint",
]
