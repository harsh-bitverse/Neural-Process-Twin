"""Unit tests verifying package structure and import integrity."""

from __future__ import annotations

import importlib
import re

import pytest


@pytest.mark.unit
def test_package_import_and_version() -> None:
    """Verify that neural_process_twin can be imported and exports a valid __version__."""
    import neural_process_twin

    assert hasattr(neural_process_twin, "__version__")
    version = neural_process_twin.__version__
    assert isinstance(version, str)
    # Check PEP 440 semantic version pattern (e.g., 0.1.0 or 0.1.0.dev0)
    semver_pattern = r"^\d+\.\d+\.\d+(\.(dev|a|b|rc)\d+)?$"
    assert re.match(semver_pattern, version) is not None, f"Invalid version format: {version}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "subpackage_name",
    [
        "neural_process_twin.physics",
        "neural_process_twin.solvers",
        "neural_process_twin.data",
        "neural_process_twin.models",
        "neural_process_twin.training",
        "neural_process_twin.evaluation",
        "neural_process_twin.uncertainty",
        "neural_process_twin.utils",
    ],
)
def test_subpackages_importable_with_docstrings(subpackage_name: str) -> None:
    """Verify that all core subpackages exist, are importable, and contain informative docstrings."""
    module = importlib.import_module(subpackage_name)
    assert module is not None
    assert module.__doc__ is not None
    assert len(module.__doc__.strip()) > 20, (
        f"Module {subpackage_name} is missing a substantive docstring."
    )
