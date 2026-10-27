"""Scientific Unit and Regression Tests for Multi-Resolution Neural Operator Generalization.

Milestone 08C Test Suite:
- Dataset integrity and resolution provenance
- Continuous physical boundary encoding exactness and round-trip fidelity
- Escrow lock invariants on the Touchstone Confirmation Dataset
- FNO-2D resolution invariance and mode truncation safety
- Multi-resolution discrete PDE residual operator consistency
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from neural_process_twin.data.continuous_encoding import (
    decode_continuous_boundary_conditions,
    encode_continuous_physical_inputs,
)
from neural_process_twin.data.normalization import FieldNormalizer
from neural_process_twin.models.fno2d import FourierNeuralOperator2D
from neural_process_twin.solvers.boundary import (
    BoundaryCondition,
    BoundaryConditions,
    BoundarySide,
    BoundaryType,
)
from neural_process_twin.solvers.grid import Grid2D
from neural_process_twin.training.physics_loss import DiscretePDEResidual

EXPECTED_CONFIRMATION_SHA256 = "ca615ab6c4cd27436f66551728bb5e9aca24d76149782b68335523678deadf20"


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 of file."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestMultiResDatasetIntegrity:
    """Validate multi-resolution dataset structure, physics, and provenance."""

    @pytest.fixture
    def h5_file(self) -> Path:
        p = Path("artifacts/data/multires_dataset.h5")
        if not p.is_file():
            pytest.skip(f"Multi-resolution dataset not found: {p}")
        return p

    def test_resolution_groups_exist(self, h5_file: Path) -> None:
        with h5py.File(h5_file, "r") as f:
            for res in [32, 64, 128, 256]:
                grp_name = f"res_{res}"
                assert grp_name in f, f"Missing resolution group: {grp_name}"

    def test_physical_field_bounds(self, h5_file: Path) -> None:
        with h5py.File(h5_file, "r") as f:
            for res in [32, 64]:
                train_grp = f[f"res_{res}/train"]
                k = train_grp["k"][:]
                q = train_grp["q"][:]
                t = train_grp["temperature"][:]

                assert np.all(k > 0.0), f"Negative conductivity detected in res_{res}"
                assert np.all(t >= 150.0) and np.all(t <= 600.0), (
                    f"Temperature out of physical bounds in res_{res}"
                )
                assert np.all(np.isfinite(k)), f"NaN/Inf in k for res_{res}"
                assert np.all(np.isfinite(q)), f"NaN/Inf in q for res_{res}"
                assert np.all(np.isfinite(t)), f"NaN/Inf in temperature for res_{res}"

    def test_ood_physical_split_criterion(self, h5_file: Path) -> None:
        with h5py.File(h5_file, "r") as f:
            ood_k = f["res_32/test_ood/k"][:]
            n_samples = ood_k.shape[0]
            assert n_samples == 200

            contrasts = [float(np.max(ood_k[i]) / np.min(ood_k[i])) for i in range(n_samples)]
            high_contrast = sum(1 for c in contrasts if c >= 20.0)
            grf = sum(1 for c in contrasts if c < 20.0)

            assert high_contrast > 0, "No high-contrast inclusion samples found in test_ood"
            assert grf > 0, "No short-correlation GRF samples found in test_ood"
            assert high_contrast + grf == n_samples


class TestContinuousBoundaryEncoding:
    """Validate continuous boundary representation properties."""

    @pytest.mark.parametrize("n", [32, 64, 128])
    def test_encoding_shape_and_range(self, n: int) -> None:
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, n, n, thickness=1.0)
        k_field = 2.0 * np.ones((n, n), dtype=np.float32)
        q_field = 100.0 * np.ones((n, n), dtype=np.float32)
        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, 300.0 * np.ones(n)),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, 350.0 * np.ones(n)),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, np.zeros(n)),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, 50.0 * np.ones(n)),
        )

        tensor = encode_continuous_physical_inputs(
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            grid=grid,
            normalizer=None,
        )

        assert tensor.shape == (12, n, n)
        assert np.all(np.isfinite(tensor))

        # Check coordinates channels
        assert np.isclose(np.min(tensor[2]), (0.5 / n), atol=1e-3)
        assert np.isclose(np.max(tensor[2]), 1.0 - (0.5 / n), atol=1e-3)

    def test_roundtrip_decoding(self) -> None:
        n = 32
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, n, n, thickness=1.0)
        k_field = np.ones((n, n), dtype=np.float32)
        q_field = np.zeros((n, n), dtype=np.float32)

        t_left = 310.0 * np.ones(n)
        t_right = 340.0 * np.ones(n)
        j_bottom = 50.0 * np.ones(n)
        j_top = 100.0 * np.ones(n)

        bcs = BoundaryConditions(
            left=BoundaryCondition(BoundarySide.LEFT, BoundaryType.DIRICHLET, t_left),
            right=BoundaryCondition(BoundarySide.RIGHT, BoundaryType.DIRICHLET, t_right),
            bottom=BoundaryCondition(BoundarySide.BOTTOM, BoundaryType.NEUMANN, j_bottom),
            top=BoundaryCondition(BoundarySide.TOP, BoundaryType.NEUMANN, j_top),
        )

        normalizer = FieldNormalizer()
        normalizer.fit(
            k=np.ones((2, n, n)),
            q=np.zeros((2, n, n)),
            temperature=325.0 * np.ones((2, n, n)),
        )

        tensor = encode_continuous_physical_inputs(
            k_field=k_field,
            q_field=q_field,
            bcs=bcs,
            grid=grid,
            normalizer=normalizer,
        )

        decoded = decode_continuous_boundary_conditions(
            tensor=tensor,
            grid=grid,
            normalizer=normalizer,
        )

        evals_orig = bcs.evaluate_all(grid)
        evals_dec = decoded.evaluate_all(grid)

        assert decoded.left.bc_type == BoundaryType.DIRICHLET
        assert decoded.right.bc_type == BoundaryType.DIRICHLET
        assert decoded.bottom.bc_type == BoundaryType.NEUMANN
        assert decoded.top.bc_type == BoundaryType.NEUMANN

        np.testing.assert_allclose(evals_orig["left"], evals_dec["left"], rtol=1e-2)
        np.testing.assert_allclose(evals_orig["right"], evals_dec["right"], rtol=1e-2)
        np.testing.assert_allclose(evals_orig["bottom"], evals_dec["bottom"], atol=1e-1)
        np.testing.assert_allclose(evals_orig["top"], evals_dec["top"], atol=1e-1)


class TestTouchstoneEscrowInvariants:
    """Verify escrow invariants of the touchstone confirmation dataset."""

    @pytest.fixture
    def conf_path(self) -> Path:
        p = Path("artifacts/data/confirmation_dataset.h5")
        if not p.is_file():
            pytest.skip(f"Confirmation dataset not found: {p}")
        return p

    def test_file_hash_matches_escrow_record(self, conf_path: Path) -> None:
        actual_hash = compute_sha256(conf_path)
        assert actual_hash == EXPECTED_CONFIRMATION_SHA256, (
            f"Escrow tamper detection: expected {EXPECTED_CONFIRMATION_SHA256}, got {actual_hash}"
        )

    def test_touchstone_manifest(self) -> None:
        m_path = Path("artifacts/data/confirmation_manifest.json")
        assert m_path.is_file(), "Confirmation manifest missing"
        with open(m_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["governance_status"] == "ESCROW_LOCKED"
        assert data["sha256"] == EXPECTED_CONFIRMATION_SHA256
        assert data["splits"]["confirmation_iid"]["seed_range"][0] >= 500000
        assert data["splits"]["confirmation_ood"]["seed_range"][0] >= 500000


class TestNeuralOperatorResolutionInvariance:
    """Verify FNO-2D processes varying spatial resolutions without altering weights."""

    def test_forward_multi_resolution(self) -> None:
        model = FourierNeuralOperator2D(
            in_channels=12,
            out_channels=1,
            modes_x=12,
            modes_y=12,
            width=32,
            depth=4,
        )
        model.eval()

        # Check inference across 32, 64, 128
        for n in [32, 64, 128]:
            x = torch.randn(2, 12, n, n)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (2, 1, n, n), f"Output shape mismatch at resolution {n}"
            assert torch.all(torch.isfinite(out)), f"NaN/Inf generated at resolution {n}"


class TestDiscretePDEResidualMultiRes:
    """Verify discrete PDE residual operator at multiple resolutions."""

    @pytest.mark.parametrize("n", [32, 64])
    def test_residual_matches_expected_dimensions(self, n: int) -> None:
        grid = Grid2D(0.0, 1.0, 0.0, 1.0, n, n, thickness=1.0)
        pde_op = DiscretePDEResidual(grid=grid)

        t_tensor = 300.0 * torch.ones(1, 1, n, n)
        k_tensor = 2.0 * torch.ones(1, 1, n, n)
        q_tensor = torch.zeros(1, 1, n, n)
        bc_tensor = torch.zeros(1, 8, n, n)

        r, b = pde_op(t_tensor, k_tensor, q_tensor, bc_tensor)
        assert r.shape == (1, 1, n, n)
        assert b.shape == (1, 1, n, n)
        assert torch.all(torch.isfinite(r))
