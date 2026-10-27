"""Global pytest configuration and environment stabilization."""

from __future__ import annotations

import os

# Prevent OpenBLAS thread allocation exhaustion in multiprocessing workers on Windows
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
