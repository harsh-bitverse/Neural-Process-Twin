# Neural Process Twin

> **Physics-Aware Machine Learning Surrogate Simulator for Physical Field Prediction**

---

## 1. Project Motivation

Engineering digital twins require rapid, real-time prediction of physical fields—such as temperature distributions, structural stresses, and fluid flow velocities—to perform online monitoring, predictive maintenance, and closed-loop control. Conventional numerical solvers (Finite Element, Finite Volume, and Finite Difference Methods) provide high spatial fidelity but suffer from computational latency that precludes real-time interactive simulation and large-scale design optimization.

**Neural Process Twin** is an open-source scientific machine learning (SciML) research framework designed to train neural operators and surrogate models on high-fidelity numerical reference solutions. By learning mappings between parameter function spaces (e.g., thermal conductivity distributions and volumetric heat generation) and solution fields (e.g., temperature fields), Neural Process Twin aims to deliver orders-of-magnitude faster inference while preserving physical consistency and quantifying predictive uncertainty.

---

## 2. Scientific Problem: 2D Steady-State Heat Conduction

Phase 1 investigates two-dimensional linear steady-state thermal conduction on a rectangular domain $\Omega = [0, L_x] \times [0, L_y]$:

$$-\nabla \cdot \left( k(x, y) \nabla T(x, y) \right) = q(x, y), \quad (x, y) \in \Omega$$

Expanding in Cartesian coordinates:

$$-\frac{\partial}{\partial x}\left( k(x, y) \frac{\partial T}{\partial x} \right) - \frac{\partial}{\partial y}\left( k(x, y) \frac{\partial T}{\partial y} \right) = q(x, y)$$

subject to Dirichlet (prescribed temperature $T = T_0$) and Neumann (prescribed heat flux $-k \frac{\partial T}{\partial n} = g$) boundary conditions.

### Physical Variables & SI Units
- $T(x, y)$: Temperature field $[\text{K}]$
- $k(x, y)$: Thermal conductivity field $[\text{W}/(\text{m}\cdot\text{K})]$, with $k(x, y) \ge k_{\min} > 0$
- $q(x, y)$: Volumetric heat source generation $[\text{W}/\text{m}^3]$
- $\mathbf{q}''(x, y) = -k(x, y) \nabla T(x, y)$: Conductive heat flux density $[\text{W}/\text{m}^2]$

---

## 3. Implementation Status and Roadmap

| Milestone | Scope / Deliverable | Status |
| :--- | :--- | :--- |
| **Milestone 00** | **Project Foundation & Repository Scaffold** | **Completed** |
| **Milestone 01** | **Numerical Reference Solver & Scientific Verification (MMS)** | **Completed** |
| **Milestone 02** | **Synthetic Dataset Generation, Archival & PyTorch Ingestion** | **Completed** |
| **Milestone 03** | **CNN Baseline, FNO-2D, and Evaluation Framework** | **Completed** |
| **Milestone 04** | **Physics-Aware Training, Loss Formulations & Ablation Studies** | **Completed** |
| Milestone 05 | Comprehensive Benchmark Studies & Resolution Generalization | *Planned* |
| Milestone 06 | Spatial Uncertainty Quantification & Conformal Prediction | *Planned* |

> [!NOTE]
> Milestone 04 is complete with preflight residual audit regression tests, differentiable vectorized PyTorch finite-volume stencil matching SciPy CSR assembly to machine precision ($< 10^{-10}$ W), dimensionless scale-invariant PDE residual loss, boundary condition penalty loss, compound PhysicsAwareLoss, and controlled loss ablations across CNN and FNO-2D architectures.

---

## 4. Planned Capabilities

- **Resolution-Invariant Operator Learning**: 2D Fourier Neural Operators (FNO) capable of continuous-field evaluation across spatial resolutions.
- **Physics-Informed Regularization**: Discrete interior PDE residual loss and boundary flux conservation penalties.
- **Strict Out-of-Distribution (OOD) Evaluation**: Evaluating generalization across unseen conductivity contrast ratios ($k_{\max}/k_{\min} \gg 100$), sharp inclusion topologies, and localized thermal sources.
- **Spatial Uncertainty Quantification**: Ensembles and conformal prediction intervals providing verified safety margins for digital twin predictions.
- **Fair Latency Profiling**: Standardized apples-to-apples throughput and wall-clock benchmarking against optimized sparse numerical solvers.

---

## 5. Repository Structure

```
Neural Process Twin/
├── .gitignore                      # Git exclusion rules for environments, caches, datasets, and weights
├── pyproject.toml                  # PEP 517/621 project configuration, dependencies, and tool settings
├── README.md                       # Project overview, quickstart, and scientific documentation
├── configs/                        # System configuration files
│   ├── data.yaml                   # Provisional data generation & domain parameters
│   ├── training.yaml               # Provisional surrogate training & optimization parameters
│   └── experiments/                # Experiment configuration overrides
│       ├── README.md               # Guide to experiment hierarchy and override workflows
│       └── baseline_heat2d.yaml    # Baseline 2D FNO experiment configuration
├── docs/                           # Research specifications and architectural documentation
│   ├── scientific_specification.md # PDE governing equations, SI units, MMS verification, and OOD holdouts
│   ├── architecture.md             # Subsystem responsibilities, data flow diagrams, and interfaces
│   └── experiment_protocol.md      # Determinism, dual-metric evaluation, fair latency, and failure capture
├── scripts/                        # Automation and CLI utility scripts
│   ├── generate_dataset.py         # CLI to synthesize, verify, and archive problem ensembles
│   └── inspect_dataset.py          # CLI to inspect schema version, partitions, and field ranges
├── src/neural_process_twin/        # Core library source code
│   ├── __init__.py                 # Top-level package initializer & version declaration
│   ├── physics/                    # Physical definitions, domains, and manufactured solutions
│   │   ├── __init__.py             # Exports analytical solution classes
│   │   └── manufactured.py         # MMS solutions with constant and variable conductivity
│   ├── solvers/                    # Reference numerical solvers (FVM) (Implemented: Milestone 01)
│   │   ├── __init__.py             # Exposes Grid2D, HeatConductionSolver2D, GCI, diagnostics
│   │   ├── grid.py                 # Cell-centred 2D Cartesian grid representation
│   │   ├── boundary.py             # Dirichlet and Neumann boundary conditions with evaluation
│   │   ├── conductivity.py         # Harmonic interface averaging and face conductances
│   │   ├── assembly.py             # Sparse linear system assembly (CSR format)
│   │   ├── diagnostics.py          # Heat balance conservation and algebraic residual metrics
│   │   ├── solver.py               # Top-level solver API and structured SolverResult
│   │   └── convergence.py          # Relative L2, Linf error norms, and Roache GCI
│   ├── data/                       # Dataset generation, storage, and ingestion (Implemented: Milestone 02)
│   │   ├── __init__.py             # Exposes generators, storage, normalizer, and PyTorch dataset
│   │   ├── generators.py           # Spectral GRF, composite inclusions, heat source, BC samplers
│   │   ├── normalization.py        # Strict training-partition fitting and reversible transformations
│   │   ├── splits.py               # 4 disjoint splits, seed offsets, and explicit physical OOD rule
│   │   ├── orchestration.py        # Solver orchestration, quality control, and failure audit policy
│   │   ├── storage.py              # Compressed HDF5 archive (v1.0.0) writer and inspector
│   │   └── dataset.py              # Multi-worker safe PyTorch HeatFieldDataset & DataLoader
│   ├── models/                     # FNO-2D and baseline surrogate architectures (Planned: M03)
│   ├── training/                   # Loss formulations, trainer loops, and checkpointing (Planned: M04)
│   ├── evaluation/                 # Metrics, energy balance, and latency benchmarks (Planned: M05)
│   ├── uncertainty/                # Uncertainty estimation and conformal prediction (Planned: M06)
│   └── utils/                      # Utilities, config validation, and helpers (Implemented: M00)
│       ├── __init__.py             # Exposes typed loaders and configuration error types
│       └── config.py               # Typed dataclasses and schema validation logic
├── scripts/                        # Automation and verification utilities
│   ├── README.md                   # Inventory of scripts
│   └── verify_setup.py             # Runtime, dependency, import, and configuration validator
├── tests/                          # Automated test suites
│   ├── unit/                       # Unit tests for standalone components and configuration parsing
│   │   ├── test_package.py         # Verifies package imports and semantic versioning
│   │   └── test_config.py          # Verifies typed parsing, bounds checking, and error handling
│   ├── integration/                # Integration tests for end-to-end configuration workflows
│   │   └── test_config_workflow.py # Tests repository configs and experiment override merging
│   └── scientific/                 # Numerical solver verification (Implemented: Milestone 01)
│       ├── README.md               # Guide to scientific tests and verification criteria
│       ├── test_manufactured_solution.py # MMS convergence study and Roache GCI
│       └── test_physical_verification.py  # 10 independent physical verification checks
├── experiments/                    # Local experiment run manifests, logs, and checkpoints
│   └── README.md                   # Experiment layout and data management guidelines
├── artifacts/                      # Exportable models, calibration tables, and published weights
│   └── README.md                   # Artifact retention guidelines
└── reports/                        # Milestone reports, benchmark summaries, and failure audits
    ├── README.md                   # Reporting conventions
    ├── milestone_00_foundation.md  # Milestone 00 audit record
    └── milestone_01_numerical_solver.md # Milestone 01 audit record
```

---

## 6. Environment Setup and Installation

### Prerequisites
- Python 3.11 or newer (Python 3.12 recommended).
- [uv](https://docs.astral.sh/uv/) (recommended for fast package management) or standard Python `venv` + `pip`.

### Setup with `uv` (Recommended)

```powershell
# Create virtual environment
uv venv .venv

# Activate virtual environment
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# Install package in editable mode with development dependencies
uv pip install -e ".[dev]"
```

### Setup with Standard `pip`

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev]"
```

---

## 7. Verification, Testing, and Code Quality

Execute the following commands from the repository root:

### 1. Verify Complete Setup
Runs runtime verification, dependency checks, package import tests, and config validation:

```powershell
python scripts/verify_setup.py
```

### 2. Run Test Suite
Executes unit and integration test suites:

```powershell
pytest
```

To run with verbose output and test breakdown:

```powershell
pytest -v
```

### 3. Linting and Formatting (Ruff)

Check linting rules:

```powershell
ruff check .
```

Check code formatting:

```powershell
ruff format --check .
```

Auto-format code:

```powershell
ruff format .
```

### 4. Synthesize Reference Datasets
Generate and archive reference solution ensembles using the conservative FVM solver:

```powershell
# Rapid smoke-test verification (16x16 grid, 14 samples):
python scripts/generate_dataset.py --smoke-test --output artifacts/data/smoke_dataset.h5

# Full dataset generation from configs/data.yaml:
python scripts/generate_dataset.py --config configs/data.yaml --output artifacts/data/heat_conduction_dataset.h5
```

### 5. Inspect Dataset Archive
Inspect an archived HDF5 dataset for schema compliance, sample counts, field ranges, and conservation residuals:

```powershell
python scripts/inspect_dataset.py --path artifacts/data/smoke_dataset.h5
```

