"""Environment and configuration verification script for Neural Process Twin.

Executes end-to-end checks to ensure Python runtime, dependencies, package
installation, and default configurations are fully operational.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path


def main() -> int:
    print("=" * 70)
    print("Neural Process Twin - Environment and Configuration Verification")
    print("=" * 70)

    # 1. Platform & Python version check
    py_ver = sys.version.split()[0]
    print(f"[1/4] Python Runtime : {py_ver} on {platform.system()} ({platform.machine()})")
    if sys.version_info < (3, 11):  # noqa: UP036 - intentional runtime check for older Python installations
        print("      ERROR: Python 3.11 or newer is required.")
        return 1

    # 2. Dependency imports
    print("[2/4] Checking Core Dependencies:")
    deps = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("torch", "PyTorch"),
        ("yaml", "PyYAML"),
        ("matplotlib", "Matplotlib"),
        ("pytest", "pytest"),
        ("ruff", "Ruff"),
    ]
    all_deps_ok = True
    for mod_name, label in deps:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "available")
            print(f"      - {label:<12}: {ver}")
        except ImportError as exc:
            print(f"      - {label:<12}: MISSING ({exc})")
            all_deps_ok = False

    if not all_deps_ok:
        print("      ERROR: One or more required dependencies are missing.")
        return 1

    # 3. Package imports
    print("[3/4] Checking Package Imports (neural_process_twin):")
    subpackages = [
        "neural_process_twin",
        "neural_process_twin.physics",
        "neural_process_twin.solvers",
        "neural_process_twin.data",
        "neural_process_twin.models",
        "neural_process_twin.training",
        "neural_process_twin.evaluation",
        "neural_process_twin.uncertainty",
        "neural_process_twin.utils",
    ]
    all_pkg_ok = True
    for subpkg in subpackages:
        try:
            __import__(subpkg)
            print(f"      - {subpkg:<32}: OK")
        except ImportError as exc:
            print(f"      - {subpkg:<32}: FAILED ({exc})")
            all_pkg_ok = False

    if not all_pkg_ok:
        print(
            "      ERROR: Package import failed. Ensure package is installed via 'pip install -e .'."
        )
        return 1

    # 4. Configuration validation
    print("[4/4] Validating Default Configuration Files:")
    repo_root = Path(__file__).resolve().parent.parent
    data_cfg_path = repo_root / "configs" / "data.yaml"
    training_cfg_path = repo_root / "configs" / "training.yaml"

    try:
        from neural_process_twin.utils import load_data_config, load_training_config

        data_cfg = load_data_config(data_cfg_path)
        print(
            f"      - configs/data.yaml     : Validated (status: {data_cfg.config_status}, grid: {data_cfg.grid.nx}x{data_cfg.grid.ny})"
        )

        train_cfg = load_training_config(training_cfg_path)
        print(
            f"      - configs/training.yaml : Validated (status: {train_cfg.config_status}, model: {train_cfg.model.architecture})"
        )
    except Exception as exc:
        print(f"      ERROR: Configuration validation failed: {exc}")
        return 1

    print("=" * 70)
    print("SUCCESS: Neural Process Twin project foundation is verified.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
