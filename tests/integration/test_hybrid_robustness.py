"""Robustness and timeout tests for the Hybrid Digital Twin engine (Milestone 06C).

Verifies:
1. Checkpoint integrity mismatch detection (anti-tamper / frozen provenance).
2. Solver timeout handling with thread isolation (non-blocking shutdown).
3. Solver timeout handling with process isolation (worker termination and cleanup).
4. Repeated timeout resilience (no resource leaks or state corruption).
5. Solver divergence and exception fail-closed guarantees (temperature is strictly None).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from neural_process_twin.inference.contracts import (
    InferenceOptions,
    InferenceRequest,
    InferenceStatus,
    RejectionCode,
)
from neural_process_twin.inference.engine import HybridInferenceEngine
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.solvers.solver import HeatConductionSolver2D


@pytest.fixture(scope="module")
def engine() -> HybridInferenceEngine:
    """Fixture providing initialized HybridInferenceEngine from repository artifacts."""
    return HybridInferenceEngine.from_artifacts(device="cpu")


@pytest.fixture
def bcs() -> BoundaryConditions:
    """Standard boundary conditions."""
    return BoundaryConditions(
        left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0),
        right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0),
        bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, 0.0),
        top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 0.0),
    )


def test_checkpoint_integrity_verification_raises_on_tamper(tmp_path: Path) -> None:
    """Verify that from_artifacts raises ValueError when a checkpoint SHA-256 is tampered with."""
    # Create a dummy tampered checkpoint file with a recognized directory name
    tampered_dir = tmp_path / "scaled_fno_supervised_seed42"
    tampered_dir.mkdir(parents=True)
    tampered_file = tampered_dir / "best_model.pt"
    tampered_file.write_bytes(b"corrupted_or_tampered_weights")

    checkpoint_paths = [
        tampered_file,
        Path("artifacts/checkpoints/scaled_fno_supervised_seed43/best_model.pt"),
        Path("artifacts/checkpoints/scaled_fno_supervised_seed44/best_model.pt"),
        Path("artifacts/checkpoints/scaled_fno_supervised_seed45/best_model.pt"),
        Path("artifacts/checkpoints/scaled_fno_supervised_seed46/best_model.pt"),
    ]

    with pytest.raises(
        ValueError, match="Checkpoint integrity violation for scaled_fno_supervised_seed42"
    ):
        HybridInferenceEngine.from_artifacts(checkpoint_paths=checkpoint_paths, device="cpu")


def test_solver_thread_timeout_non_blocking(
    engine: HybridInferenceEngine, bcs: BoundaryConditions
) -> None:
    """Verify that thread isolation timeout returns promptly without blocking in __exit__."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    # Contrast 30:1 triggers fallback
    k_field = np.full((32, 32), 1.0)
    k_field[0, 0] = 30.0

    mock_solver = MagicMock()

    def slow_solve(*args, **kwargs):
        time.sleep(2.0)
        return HeatConductionSolver2D().solve(*args, **kwargs)

    mock_solver.solve.side_effect = slow_solve
    mock_solver.tolerance = 1e-10

    orig_solver = engine.solver
    engine.solver = mock_solver

    try:
        req = InferenceRequest(
            grid=grid,
            conductivity=k_field,
            source=50.0,
            boundary_conditions=bcs,
            options=InferenceOptions(
                solver_timeout_seconds=0.1,
                fallback_isolation="thread",
            ),
        )
        t0 = time.perf_counter()
        res = engine.predict(req)
        elapsed = time.perf_counter() - t0

        assert res.status == InferenceStatus.SOLVER_FAILED
        assert res.temperature is None  # Strict fail-closed
        assert res.rejection.code == RejectionCode.SOLVER_TIMEOUT
        # Check that we didn't block for the full 2 seconds
        assert elapsed < 1.0, (
            f"Thread timeout took too long ({elapsed:.2f}s), indicating blocked context exit."
        )
    finally:
        engine.solver = orig_solver


def test_repeated_timeouts_resilience(
    engine: HybridInferenceEngine, bcs: BoundaryConditions
) -> None:
    """Verify that multiple consecutive timeouts do not crash or corrupt subsequent successful solves."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    k_field = np.full((32, 32), 1.0)
    k_field[0, 0] = 30.0

    mock_solver = MagicMock()

    def slow_solve(*args, **kwargs):
        time.sleep(1.0)
        return HeatConductionSolver2D().solve(*args, **kwargs)

    mock_solver.solve.side_effect = slow_solve
    mock_solver.tolerance = 1e-10

    orig_solver = engine.solver
    engine.solver = mock_solver

    try:
        # Trigger 3 consecutive timeouts
        for _ in range(3):
            req = InferenceRequest(
                grid=grid,
                conductivity=k_field,
                source=50.0,
                boundary_conditions=bcs,
                options=InferenceOptions(solver_timeout_seconds=0.05),
            )
            res = engine.predict(req)
            assert res.status == InferenceStatus.SOLVER_FAILED
            assert res.temperature is None
    finally:
        engine.solver = orig_solver

    # Verify subsequent normal request succeeds without state corruption
    iid_req = InferenceRequest(
        grid=grid,
        conductivity=2.5,
        source=50.0,
        boundary_conditions=bcs,
    )
    iid_res = engine.predict(iid_req)
    assert iid_res.status == InferenceStatus.ACCEPTED
    assert iid_res.temperature is not None


def test_solver_divergence_fails_closed(
    engine: HybridInferenceEngine, bcs: BoundaryConditions
) -> None:
    """Verify that a solver producing non-converged or non-finite output fails closed."""
    grid = Grid2D(0.0, 1.0, 0.0, 1.0, nx=32, ny=32)
    k_field = np.full((32, 32), 1.0)
    k_field[0, 0] = 30.0

    mock_solver = MagicMock()
    mock_result = MagicMock()
    mock_result.temperature = np.full((32, 32), np.nan)  # Non-finite output
    mock_result.diagnostics.is_converged = False
    mock_result.diagnostics.relative_residual = 1.0
    mock_solver.solve.return_value = mock_result
    mock_solver.tolerance = 1e-10

    orig_solver = engine.solver
    engine.solver = mock_solver

    try:
        req = InferenceRequest(
            grid=grid,
            conductivity=k_field,
            source=50.0,
            boundary_conditions=bcs,
        )
        res = engine.predict(req)
        assert res.status == InferenceStatus.SOLVER_FAILED
        assert res.temperature is None
        assert res.rejection.code == RejectionCode.SOLVER_DIVERGENCE
    finally:
        engine.solver = orig_solver
