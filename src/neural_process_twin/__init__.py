"""Neural Process Twin.

A physics-aware machine learning surrogate simulator designed to predict
physical fields governed by partial differential equations (PDEs).

Phase 1 focuses on 2D steady-state heat conduction:
    -\\nabla \\cdot (k(x, y) \\nabla T(x, y)) = q(x, y)
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
