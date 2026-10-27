"""Physics definitions and problem specifications for Neural Process Twin.

This package defines the physical formulations, governing PDE representations,
domain geometries, boundary condition schemas, and analytical verification
solutions.
"""

from neural_process_twin.physics.manufactured import (
    LinearConduction1D,
    SinusoidalMMSConstantK,
    SinusoidalMMSVariableK,
)

__all__ = [
    "LinearConduction1D",
    "SinusoidalMMSConstantK",
    "SinusoidalMMSVariableK",
]
