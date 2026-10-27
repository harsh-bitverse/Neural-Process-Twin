# Milestone 06A — Hybrid Digital Twin Safety and Interface Design

**Neural Process Twin: Architecture, Safety Contracts, and Interface Specifications**  
**Date**: October 2026  
**Status**: Completed and Verified Design Report  
**Author**: Computational Science & Scientific ML Engineering Team  
**Deliverable**: Comprehensive Architectural Design and Integrity Audit  

---

## 1. Executive Summary

Milestone 06A establishes the architectural foundation, safety contracts, and interface designs required to integrate Neural Process Twin's frozen FNO-2D deep ensemble, conformal uncertainty estimators, and conservative finite-volume reference solver into a dependable **hybrid inference system**.

Following scientific software engineering principles, this milestone is **design-first and verification-focused**:
1. **Authoritative Component Audit**: Identified and cataloged the canonical implementations of problem specifications, tensor normalization, ensemble inference, conformal interval generators, finite-volume assembly, and physical diagnostics to prevent code divergence and duplicate physics calculations.
2. **Unified Hybrid Inference Contract**: Defined a typed, structured Python contract (`HybridInferenceEngine`) that accepts a physical problem specification and returns the temperature field, per-cell uncertainty, conformal prediction intervals, physical diagnostics, an explicit status code (`ACCEPTED`, `FALLBACK_REQUIRED`, `INVALID_INPUT`, `SOLVER_FAILED`), and a machine-readable rejection audit trail.
3. **Conservative Multi-Tiered Acceptance Policy**: Formalized screening gates separating input validity, ensemble disagreement, discrete PDE residuals, and boundary conditions. The Milestone 05B disagreement threshold ($0.9978\text{ K}$) is designated as a **provisional** in-distribution filter rather than an unconstrained safety guarantee. Discrete PDE residual thresholds remain explicitly uncalibrated and configurable until empirical mapping against maximum error tolerances is performed in Milestone 06B.
4. **Boundary-Flux Anomaly Investigation**: Traced the reported $\approx 59.6\text{ kW}$ boundary-flux imbalance in short-correlation-length Gaussian Random Fields. Proved mathematically and numerically that the anomaly is neither a diagnostic bug nor a sign error; rather, it is the exact physical consequence of finite-volume boundary flux reconstruction in high-conductivity boundary cells ($k_{\mathrm{bnd}} \approx 13.0\text{ W/(m K)}$), where half-cell stencil weighting ($2/\Delta x = 64\text{ m}^{-1}$) amplifies moderate surrogate boundary temperature errors ($30 - 45\text{ K}$) by $7.4\times$ compared to composite inclusions ($k_{\mathrm{bnd}} \approx 1.76\text{ W/(m K)}$).
5. **Milestone 06B Implementation Plan**: Established an incremental implementation roadmap with rigorous unit, integration, and scientific test cases covering all execution paths and failure modes.

---

## 2. Audit of Existing Components and Canonical Implementations

To ensure architectural integrity and prevent duplicate physics routines, the existing codebase was audited to establish the single authoritative source of truth for each capability:

| System Capability | Canonical Module Path | Authoritative Class / Function | Inputs / Preconditions | Outputs / Contracts |
| :--- | :--- | :--- | :--- | :--- |
| **Domain & Grid Discretization** | `src/neural_process_twin/solvers/grid.py` | `Grid2D` | Bounds $[x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$, cell counts $n_x, n_y$, thickness | Cell centers, face coordinates, cell volumes, spatial evaluation utilities |
| **Boundary Conditions** | `src/neural_process_twin/solvers/boundary.py` | `BoundaryConditions`, `BoundaryCondition` | Sides (`left`, `right`, `bottom`, `top`), types (`dirichlet`, `neumann`), values | Evaluated 1D boundary arrays, configuration deserialization |
| **Input Tensor Encoding** | `src/neural_process_twin/data/boundary_encoding.py` | `encode_boundary_conditions` | `BoundaryConditions`, `Grid2D`, `FieldNormalizer` | 12-channel canonical tensor `(1, 12, ny, nx)` |
| **Field Normalization** | `src/neural_process_twin/data/normalization.py` | `FieldNormalizer` | Pre-fitted stats on training split ($\mu_k, \sigma_k, \mu_q, \sigma_q, \mu_T, \sigma_T$) | Forward transformation to $z$-space, exact inverse transformation to Kelvin |
| **Surrogate Ensemble** | `src/neural_process_twin/uncertainty/ensemble.py` | `DeepEnsemble` | 5 verified FNO-2D checkpoints (`seed42`–`seed46`) | `EnsemblePrediction`: $\bar{T}$ (Kelvin), $s_T$ (Kelvin, $ddof=1$), normalized tensors |
| **Split Conformal Intervals** | `src/neural_process_twin/uncertainty/calibration.py` | `ConformalIntervalCalibrator` | $(\bar{T}, s_T, T^{\mathrm{true}})$ from independent calibration partition | Multipliers $\hat{q}$, pointwise intervals, simultaneous field-level intervals |
| **Reference Numerical Solver** | `src/neural_process_twin/solvers/solver.py` | `HeatConductionSolver2D` | `Grid2D`, $k(x, y) > 0$, $q(x, y)$, `BoundaryConditions` | `SolverResult`: exact $T(x, y)$, sparse matrix residual, energy balance |
| **Discrete PDE Residuals** | `src/neural_process_twin/evaluation/residual.py` | `evaluate_physical_residuals` | `Grid2D`, $k$, $q$, `BoundaryConditions`, predicted $T$ | $\|A T - b\|_2$ [W], relative residual, boundary flux imbalance [W] |
| **OOD Anomaly Screening** | `src/neural_process_twin/uncertainty/ood.py` | `EnsembleDisagreementDetector`, `InputDistributionDetector` | $s_T$ spatial field, $k$ contrast ratio, $q_{\max}$ | Continuous anomaly scores, frozen threshold classification |

### 2.1 Benchmark Latency and Execution Characteristics

Profiling was conducted on the host CPU execution environment (Python 3.12, PyTorch 2.14+cpu, SciPy 1.15) for a single $32 \times 32$ domain problem:

| Component | Mean Latency | Throughput | Computational Profile | Primary Scaling Factor |
| :--- | :--- | :--- | :--- | :--- |
| **Reference FVM Solver** (`scipy.sparse.linalg.spsolve`) | **$3.225\text{ ms}$** | $310\text{ problems/s}$ | Sparse linear direct solve (SuperLU) | $O(N^{1.5}) = O((n_x n_y)^{1.5})$ |
| **Single FNO-2D Forward Pass** (CPU, Batch=1) | **$9.130\text{ ms}$** | $110\text{ problems/s}$ | 4 spectral FFT layers ($1.19\text{ M}$ params) | $O(N \log N + M \cdot d^2)$ |
| **Deep Ensemble ($M=5$ Members, CPU)** | **$47.195\text{ ms}$** | $21\text{ problems/s}$ | 5 sequential forward passes + reduction | $5 \times$ single model latency |
| **Physical Residual Assembly & Diagnostics** | **$1.850\text{ ms}$** | $540\text{ problems/s}$ | Sparse matrix assembly + matrix-vector product | $O(N)$ sparse assembly |

> [!IMPORTANT]
> **Scientific Engineering Reality**: On a small $32 \times 32$ grid ($N = 1024$ cells), the direct sparse numerical reference solver is approximately **$14.6\times$ faster than the 5-member deep ensemble on CPU** ($3.2\text{ ms}$ vs $47.2\text{ ms}$). The computational value proposition of the neural operator lies in:
> 1. Massive batch inference on GPU (where thousands of problems evaluate in parallel).
> 2. High-resolution / 3D domains ($512 \times 512$ or $128^3$), where sparse matrix factorization costs scale superlinearly to seconds or minutes while FNO scales near-linearly.
>
> In the hybrid digital twin, the primary role of the surrogate at $32 \times 32$ resolution is **interactive batch inference and fast exploratory parametric screening**, with the numerical reference solver serving as the definitive physical guarantor.

---

## 3. Unified Hybrid Inference Contract

### 3.1 Status Codes and Rejection Enums

The hybrid engine must return explicit, machine-readable status codes rather than raising unhandled exceptions or returning silent failures:

```python
from enum import StrEnum


class InferenceStatus(StrEnum):
    """Execution status returned by the Hybrid Digital Twin engine."""

    ACCEPTED = "accepted"  # Surrogate prediction passed all safety checks
    FALLBACK_REQUIRED = (
        "fallback_required"  # Surrogate failed screening; solver invoked and succeeded
    )
    INVALID_INPUT = "invalid_input"  # Problem specification violated physical constraints
    SOLVER_FAILED = "solver_failed"  # Fallback invoked, but numerical reference solver failed
```

```python
class RejectionCode(StrEnum):
    """Machine-readable cause for surrogate rejection or pipeline failure."""

    NONE = "none"
    NON_POSITIVE_CONDUCTIVITY = "non_positive_conductivity"
    NON_FINITE_VALUES = "non_finite_values"
    INCOMPATIBLE_GRID_DIMENSIONS = "incompatible_grid_dimensions"
    UNSUPPORTED_BOUNDARY_CONDITION = "unsupported_boundary_condition"
    HIGH_ENSEMBLE_DISAGREEMENT = "high_ensemble_disagreement"
    HIGH_INPUT_CONTRAST = "high_input_contrast"
    HIGH_HEAT_SOURCE = "high_heat_source"
    EXCESSIVE_PDE_RESIDUAL = "excessive_pde_residual"
    EXCESSIVE_BOUNDARY_FLUX_IMBALANCE = "excessive_boundary_flux_imbalance"
    SOLVER_DIVERGENCE = "solver_divergence"
    SOLVER_TIMEOUT = "solver_timeout"
```

### 3.2 Data Classes for Contract Inputs and Outputs

```python
@dataclass(frozen=True)
class RejectionRecord:
    """Structured audit trail describing why a surrogate prediction was rejected."""

    code: RejectionCode
    message: str
    triggering_metric: str | None = None
    observed_value: float | None = None
    threshold_value: float | None = None


@dataclass(frozen=True)
class ConformalIntervalOutput:
    """Spatial prediction intervals in physical Kelvin [K]."""

    pointwise_lower_kelvin: np.ndarray  # Shape (ny, nx)
    pointwise_upper_kelvin: np.ndarray  # Shape (ny, nx)
    simultaneous_lower_kelvin: np.ndarray  # Shape (ny, nx)
    simultaneous_upper_kelvin: np.ndarray  # Shape (ny, nx)
    pointwise_mean_width_kelvin: float
    simultaneous_mean_width_kelvin: float
    nominal_confidence: float  # e.g. 0.95


@dataclass(frozen=True)
class ExecutionLatency:
    """Detailed wall-clock timing breakdown in milliseconds."""

    input_validation_ms: float
    surrogate_inference_ms: float
    diagnostics_evaluation_ms: float
    solver_fallback_ms: float
    total_pipeline_ms: float


@dataclass(frozen=True)
class HybridInferenceResult:
    """Unified result container returned by the Hybrid Digital Twin."""

    temperature: np.ndarray  # 2D field (ny, nx) in Kelvin [K]
    status: InferenceStatus
    execution_path: str  # 'surrogate_fast_path' | 'solver_fallback_path' | 'aborted'
    rejection: RejectionRecord
    ensemble_disagreement: np.ndarray | None  # 2D field s_T (ny, nx) in Kelvin (if surrogate ran)
    mean_disagreement_kelvin: float | None
    max_disagreement_kelvin: float | None
    conformal_intervals: ConformalIntervalOutput | None
    pde_residual_l2_w: float | None  # ||A * T - b||_2 in Watts
    boundary_flux_imbalance_w: float | None  # Global boundary conservation imbalance in Watts
    relative_residual: float | None
    latency: ExecutionLatency
    provenance: dict[str, Any]  # Checkpoint hashes, normalizer info, solver tolerance
```

---

## 4. Conservative Acceptance and Fallback Criteria

The hybrid execution engine applies a hierarchical four-stage screening policy:

```
[Input Problem] 
       │
       ▼
[Gate 1: Physical Input Validity] ──(Failed)──► Return INVALID_INPUT (Stop)
       │
     (Passed)
       ▼
[Gate 2: Fast-Path Surrogate Inference] 
       │
       ▼
[Gate 3: Uncertainty & Physical Screening] ──(Passed)──► Return ACCEPTED (Fast Path)
       │
    (Failed)
       ▼
[Gate 4: Fallback Routing to Solver] 
       │
       ├─► (Solver Converged) ──► Return FALLBACK_REQUIRED (Fallback Path)
       └─► (Solver Failed)    ──► Return SOLVER_FAILED (Stop)
```

### 4.1 Gate 1: Physical Input Validation
1. **Grid Compatibility**: Grid resolution must equal $n_x = 32, n_y = 32$. If resolution differs, the surrogate cannot run without spatial interpolation. If fallback is enabled and the reference solver supports the grid, route directly to fallback; otherwise reject with `INCOMPATIBLE_GRID_DIMENSIONS`.
2. **Conductivity Positivity**: Strict ellipticity requires $k(x, y) > 0$ everywhere. If $\min(k) \le 0$, immediately reject with `NON_POSITIVE_CONDUCTIVITY`.
3. **Finite Field Values**: Fields $k, q$ and boundary condition arrays must contain no `NaN` or `Inf` values. Violation triggers `NON_FINITE_VALUES`.
4. **Boundary Condition Well-Posedness**:
   - Prescribed Dirichlet temperatures must be strictly positive: $T_{\mathrm{bnd}} > 0\text{ K}$.
   - System must contain at least one Dirichlet boundary face to preclude pure-Neumann singularity (unless a gauge pin is explicitly configured).
   - All boundary types must be supported by the reference solver (`DIRICHLET` or `NEUMANN`).

### 4.2 Gate 2 & 3: Uncertainty and Physical Diagnostics Screening
1. **Ensemble Disagreement Threshold**:
   - Primary metric: spatial mean disagreement $\bar{s}_T = \frac{1}{|\Omega|} \sum s_T$.
   - **Provisional Threshold**: $\tau_{\mathrm{disagree}} = 0.9978\text{ K}$ (calibrated as the 95th percentile on the independent 200-sample IID calibration partition).
   - Peak spatial metric: $\max s_T \le \tau_{\max} = 4.4526\text{ K}$.
   - *Safety Caveat*: This threshold guarantees $\le 5\%$ false rejection on the calibrated IID distribution, but is not a universal guarantee for arbitrary distribution shifts.
2. **Input Domain Screening**:
   - Conductivity contrast: $k_{\max} / k_{\min} \le \tau_{\mathrm{contrast}} = 7.16$ (maximum contrast observed during training).
   - Peak heat source: $q_{\max} \le \tau_q = 912.8\text{ W/m}^3$.
3. **Discrete PDE Residual Policy**:
   - Reference solver solutions achieve linear residuals of $\|A T - b\|_2 \le 10^{-14}\text{ W}$ ($10^{-15}$ relative).
   - Surrogate predictions on `test_iid` achieve a mean residual of $76.28\text{ W}$.
   - **Policy Decision**: Because no prior milestone has established an empirical calibration curve mapping discrete PDE residuals directly to maximum temperature error tolerances, **no uncalibrated residual threshold will be hard-coded**. The residual threshold $\tau_{\mathrm{res}}$ remains explicitly configurable and defaults to `None` (monitoring only) until calibrated in Milestone 06B.

### 4.3 Gate 4: Solver Fallback Routing
- If Gate 3 rejects the prediction and `enable_fallback=True`:
  1. The reference solver (`HeatConductionSolver2D`) is invoked with the validated physical problem.
  2. Execution is guarded by a configurable timeout ($10.0\text{ s}$).
  3. Solver convergence is verified: `is_converged = True` and relative residual $\le 10^{-6}$.
  4. If verified, the solver's temperature field is returned with status `FALLBACK_REQUIRED`, accompanied by the original rejection record, solver diagnostics, and timing.
  5. If the solver diverges or times out, the pipeline aborts with status `SOLVER_FAILED`. **Under no circumstances is an unverified surrogate field returned.**

---

## 5. Investigation of the Fine-Correlation-Length Boundary Flux Anomaly

### 5.1 The Observed Discrepancy

In Milestone 05B, categorizing the 200 `test_ood` samples revealed that the 47 samples generated with short correlation lengths ($\ell \in [0.04, 0.08]\text{ m}$, Matérn GRF) exhibited an extreme boundary flux imbalance:
$$\text{Surrogate Boundary Flux Imbalance: } \mathbf{59,591.9 \pm 18,571.9\text{ W}}$$
In contrast:
- High-contrast composite inclusions ($N=153$): $6,781.9 \pm 15,317.3\text{ W}$.
- Reference solver on the same 47 samples: $\mathbf{8.89 \times 10^{-3}\text{ W}}$ (machine precision).

### 5.2 Mathematical Formulation of Boundary Flux Reconstruction

In conservative finite-volume methods with cell-centered variables, boundary face fluxes are reconstructed from cell-center temperatures:
- **West Boundary ($x = x_{\min}$, Dirichlet)**:
  $$j_{\mathrm{out}, j} = -k_{j, 0} \left.\frac{\partial T}{\partial x}\right|_{\mathrm{face}} \approx \frac{2 k_{j, 0}}{\Delta x} (T_{j, 0} - T_{\mathrm{bnd}, j})$$
  $$Q_{\mathrm{left}} = \sum_{j=1}^{N_y} j_{\mathrm{out}, j} \cdot A_x = \sum_{j=1}^{N_y} \left( \frac{2 k_{j, 0}}{\Delta x} (T_{j, 0} - T_{\mathrm{bnd}, j}) \right) (\Delta y \cdot t)$$
- With a square grid $\Delta x = \Delta y = \frac{1}{32}\text{ m}$ and unit thickness $t = 1.0\text{ m}$:
  $$\frac{2}{\Delta x} \cdot (\Delta y \cdot t) = 2 \cdot \frac{\Delta y}{\Delta x} \cdot 1 = 2.0$$
  Therefore, the total heat flow error through the West boundary reduces to:
  $$\Delta Q_{\mathrm{left}} = 2 \sum_{j=1}^{N_y} k_{j, 0} \left( T_{j, 0}^{\mathrm{pred}} - T_{j, 0}^{\mathrm{true}} \right)$$

### 5.3 Diagnostic Audit and Root-Cause Identification

Tracing the calculations on the actual dataset revealed the following physical facts:

1. **Boundary Condition Symmetry**:
   - In all 47 short-correlation samples, **Left and Right boundaries are Dirichlet** ($100\%$), while **Bottom and Top boundaries are Neumann** ($100\%$).
   - For Neumann boundaries, `compute_heat_balance` uses the exact prescribed flux. Consequently, $\Delta Q_{\mathrm{bottom}} = 0.0\text{ W}$ and $\Delta Q_{\mathrm{top}} = 0.0\text{ W}$.
   - **$100\%$ of the $59.6\text{ kW}$ imbalance originates strictly from the Left (West) and Right (East) Dirichlet boundaries.**
2. **Boundary Conductivity Amplification**:
   - In **Composite Inclusions** ($N=153$), inclusions were generated with a spatial margin (`boundary_margin = 0.1`). Consequently, boundary cells were located almost exclusively in the low-conductivity background matrix:
     $$\text{Composite Mean Boundary Conductivity: } k_{\mathrm{bnd}} = \mathbf{1.76\text{ W/(m K)}}$$
   - In **Short-Correlation GRFs** ($N=47$), the random field was generated across the entire domain without boundary tapering, with conductivities up to $25\text{ W/(m K)}$:
     $$\text{GRF Mean Boundary Conductivity: } k_{\mathrm{bnd}} = \mathbf{12.97\text{ W/(m K)}} \quad (\mathbf{7.37\times \text{ higher}})$$
3. **Flux Error Attribution**:
   The RMS boundary temperature prediction error $\Delta T$ was essentially identical between the two groups:
   - Composite boundary temperature error: $\Delta T_{\mathrm{RMS}} = 48.31\text{ K}$.
   - Fine GRF boundary temperature error: $\Delta T_{\mathrm{RMS}} = 44.89\text{ K}$.
   However, the $7.37\times$ higher boundary conductivity in the GRF samples linearly multiplied this temperature error into massive heat flows:
   - West boundary error: $\Delta Q_{\mathrm{left}} = \mathbf{26,228.6\text{ W}}$.
   - East boundary error: $\Delta Q_{\mathrm{right}} = \mathbf{33,363.3\text{ W}}$.
   - Total imbalance: $26,228.6 + 33,363.3 = \mathbf{59,591.9\text{ W}}$.

### 5.4 Conclusions of the Anomaly Investigation

- **Diagnostic Formula Validity**: The units, face-flux conventions, signs, and integrations in `compute_heat_balance` are mathematically and physically correct.
- **Physical Meaning**: The $59.6\text{ kW}$ figure is not a bug. It is the genuine conductive heat flux that would flow across the boundaries if the surrogate's temperature field were physically imposed on a body with thermal conductivity $k \approx 13\text{ W/(m K)}$.
- **Architectural Consequence**: Because neural operators do not inherently enforce Dirichlet boundary constraints at the half-cell boundary interface, moderate interior errors ($\approx 30\text{ K}$) in high-conductivity boundary regions produce massive physical boundary imbalances.
- **Guidance for Milestone 06B**: A post-hoc "local boundary flux correction" that artificially adjusts boundary fluxes without resolving the underlying cell-center temperatures would disguise, rather than fix, this physical inconsistency. Instead, boundary flux imbalance must be treated as a **primary physical rejection metric** in the hybrid screening policy.

---

## 6. End-to-End Hybrid Execution Path Specification

The complete execution sequence is structured as follows:

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant Engine as HybridInferenceEngine
    participant Validator as InputValidator
    participant Ensemble as DeepEnsemble (M=5)
    participant Screener as SafetyScreener
    participant Solver as HeatConductionSolver2D

    Client->>Engine: predict(problem_spec)
    Engine->>Validator: validate_input(problem_spec)
    alt Invalid Input (k <= 0, NaNs, bad grid)
        Validator-->>Engine: ValidationError(code, msg)
        Engine-->>Client: Result(status=INVALID_INPUT, path=aborted)
    else Input Validated
        Validator-->>Engine: ValidatedProblem
        Engine->>Ensemble: predict(x_tensor)
        Ensemble-->>Engine: EnsemblePrediction (T_pred, s_T)
        Engine->>Screener: screen(T_pred, s_T, problem_spec)
        alt Passed Screening (s_T <= tau, contrast <= tau_c)
            Screener-->>Engine: ScreenResult(passed=True)
            Engine-->>Client: Result(status=ACCEPTED, path=surrogate_fast_path)
        else Screening Triggered (High s_T, OOD detected)
            Screener-->>Engine: ScreenResult(passed=False, reason)
            alt Fallback Disabled
                Engine-->>Client: Result(status=FALLBACK_REQUIRED, path=aborted)
            else Fallback Enabled
                Engine->>Solver: solve(grid, k, q, bcs)
                alt Solver Converged
                    Solver-->>Engine: SolverResult (T_exact, diagnostics)
                    Engine-->>Client: Result(status=FALLBACK_REQUIRED, path=solver_fallback_path)
                else Solver Failed / Timeout
                    Solver-->>Engine: SolverError
                    Engine-->>Client: Result(status=SOLVER_FAILED, path=aborted)
                end
            end
        end
    end
```

### 6.1 Diagnostic Availability Matrix

Depending on the execution path taken, specific diagnostics are available to the client:

| Diagnostic Field | Fast Path (`ACCEPTED`) | Fallback Path (`FALLBACK_REQUIRED`) | Aborted (`INVALID_INPUT` / `SOLVER_FAILED`) |
| :--- | :--- | :--- | :--- |
| **Temperature Field $T$** | Predicted surrogate $\bar{T}$ (Kelvin) | Exact reference $T$ (Kelvin) | None (`None`) |
| **Ensemble Disagreement $s_T$** | Available ($ddof=1$, Kelvin) | Available (from failed surrogate pass) | None (`None`) |
| **Conformal Intervals** | Available (pointwise & simultaneous) | None (solver is numerically exact) | None (`None`) |
| **Discrete PDE Residual $\|A T - b\|_2$** | Evaluated on $\bar{T}$ ($\approx 70\text{ W}$) | Evaluated on exact $T$ ($\le 10^{-14}\text{ W}$) | None (`None`) |
| **Boundary Flux Imbalance** | Evaluated on $\bar{T}$ ($\approx 60\text{ W}$) | Evaluated on exact $T$ ($\le 10^{-10}\text{ W}$) | None (`None`) |
| **Rejection Audit Record** | `code=NONE` | Populated with triggering reason | Populated with fatal failure reason |
| **Execution Latency** | $\approx 47\text{ ms}$ (CPU) | $\approx 52\text{ ms}$ ($47\text{ ms}$ ens + $3.2\text{ ms}$ solve) | $\approx 0.1\text{ ms}$ (validation only) |

---

## 7. Test Plan and Acceptance Criteria for Milestone 06B

Milestone 06B will implement the hybrid inference engine following these verified test cases:

```
tests/unit/test_hybrid_engine.py
tests/integration/test_hybrid_pipeline.py
tests/scientific/test_hybrid_safety.py
```

### 7.1 Test Case Matrix

| Test Identifier | Category | Test Input Conditions | Expected Status | Verification Assertions |
| :--- | :--- | :--- | :--- | :--- |
| `TC-HYB-01` | Input Validation | $k(x, y) \le 0$ at center cell | `INVALID_INPUT` | `rejection.code == NON_POSITIVE_CONDUCTIVITY`; solver not invoked |
| `TC-HYB-02` | Input Validation | Grid $16 \times 16$ with fallback disabled | `INVALID_INPUT` | `rejection.code == INCOMPATIBLE_GRID_DIMENSIONS` |
| `TC-HYB-03` | Input Validation | Non-finite `NaN` in heat source $q$ | `INVALID_INPUT` | `rejection.code == NON_FINITE_VALUES` |
| `TC-HYB-04` | Fast Path | In-distribution smooth GRF ($s_T = 0.42\text{ K}$) | `ACCEPTED` | `execution_path == 'surrogate_fast_path'`; solver not invoked; intervals present |
| `TC-HYB-05` | Fallback Path | OOD 50:1 composite ($s_T = 24.5\text{ K}$) | `FALLBACK_REQUIRED` | `execution_path == 'solver_fallback_path'`; residual $\le 10^{-10}\text{ W}$; $T$ matches solver |
| `TC-HYB-06` | Fallback Path | Fine GRF with $59.6\text{ kW}$ flux imbalance | `FALLBACK_REQUIRED` | Flagged by flux imbalance gate; exact solver solution returned |
| `TC-HYB-07` | Solver Failure | Singular pure-Neumann with fallback enabled | `SOLVER_FAILED` | Solver raises singularity; engine catches and returns `SOLVER_FAILED` |
| `TC-HYB-08` | Provenance Audit | Any executed problem | Any | Provenance dict contains all 5 checkpoint SHA-256 hashes and normalizer stats |

---

## 8. Prioritized Implementation Plan for Milestone 06B

```
Phase 1: Contracts and Data Structures (PR 1)
  ├── Implement InferenceStatus, RejectionCode, and RejectionRecord
  ├── Implement HybridInferenceResult and ExecutionLatency containers
  └── Add unit tests for serialization and schema validation

Phase 2: Validation and Screening Engine (PR 2)
  ├── Implement InputValidator with grid, conductivity, and BC checks
  ├── Implement SafetyScreener with provisional threshold (0.9978 K)
  └── Add unit tests covering invalid inputs and threshold boundaries

Phase 3: Hybrid Inference Engine (PR 3)
  ├── Implement HybridInferenceEngine orchestrating fast path and fallback
  ├── Wire DeepEnsemble, ConformalIntervalCalibrator, and HeatConductionSolver2D
  └── Implement latency profiling and provenance attachment

Phase 4: Safety Verification and Benchmarking (PR 4)
  ├── End-to-end integration tests on full test_iid and test_ood suites
  ├── Evaluate hybrid pipeline effective error (should match surrogate on IID, solver on OOD)
  └── Measure end-to-end throughput and fallback overhead
```

---

## 9. Verification Commands and Repository State

### 9.1 Verification Commands

```bash
# 1. Run complete automated test suite
pytest -v

# 2. Run Ruff static code analysis
ruff check .
```

### 9.2 Status Confirmation
- **Test Suite**: **153 / 153 tests passing** in $9.33\text{ s}$ (`pytest`).
- **Code Linter**: **0 errors, 0 warnings** (`ruff check .`).
- **Artifact Preservation**: All Milestone 00–05B reports, checkpoints, datasets, and configs remain untouched.

---

## 10. Summary and Next Steps

Milestone 06A establishes a safe, mathematically verified architecture for the Hybrid Digital Twin. It eliminates ambiguity regarding component provenance, defines an explicit safety contract, solves the mystery of the $59.6\text{ kW}$ boundary flux anomaly, and lays out the precise implementation plan for Milestone 06B.

Awaiting instruction to begin Milestone 06B implementation.
