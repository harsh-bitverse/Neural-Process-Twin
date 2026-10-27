# Milestone 07 — Interactive Scientific Digital Twin Dashboard

**Neural Process Twin: Interactive Scientific Interface, Typed Inference API, and Real-Time Physical Verification**  
**Date**: October 2026  
**Status**: Completed and Verified Implementation  
**Author**: Computational Science & Scientific ML Engineering Team  
**Deliverable**: Research-Oriented Web Dashboard, FastAPI Backend Service, Fail-Closed UI Integration, and Empirical Performance Disclosures  

---

## 1. Executive Summary

Milestone 07 delivers a research-oriented, interactive web dashboard for **Neural Process Twin**. The dashboard exposes the complete hybrid digital twin system developed across Milestones 00–06: the frozen 5-member Fourier Neural Operator (FNO-2D) ensemble, the conservative finite-volume reference solver (FVM), split-conformal prediction intervals, multi-tiered physics-based safety screening, and real-time physical conservation diagnostics.

### 1.1 Non-Negotiable Scientific Guardrails Respected

1. **Authoritative Physics Executed in Python**: All physical modeling, PDE residual evaluations, global heat-balance checks, and numerical fallback solves remain strictly implemented in Python (`neural_process_twin.inference`, `neural_process_twin.solver`). No PDE physics or numerical solvers were reimplemented in client-side JavaScript.
2. **Frozen Ensemble Preservation**: The five FNO-2D models trained in Milestone 04B and selected in Milestone 05 were preserved untouched. Runtime SHA-256 digests are asserted against golden reference hashes on startup and surfaced in the dashboard's audited provenance modal.
3. **Fail-Closed Safety Contract Enforced**: When a surrogate prediction is rejected by input bounds or ensemble disagreement screening, the rejected prediction is **never** exposed as an accepted solution. If fallback is disabled by policy (`FALLBACK_REQUIRED`), `temperature` is strictly `null`. If fallback succeeds (`FALLBACK_SUCCEEDED`), the canvas renders the verified numerical solution with a prominent fallback badge and an audit trail explaining why the surrogate was rejected.
4. **Epistemic Honesty for Deterministic Solves**: Direct numerical solves (FVM) return strictly `null` for ensemble disagreement and conformal prediction intervals. The system never fabricates uncertainty estimates for deterministic algorithms.
5. **Transparent Performance Disclosures**: The dashboard and API explicitly disclose that direct finite-volume solves with SuperLU are $\approx 11\text{--}13\times$ faster on single $32 \times 32$ CPU instances than evaluating the 5-member neural surrogate ensemble. Surrogates are contextualized for their true strengths: GPU batching, high-throughput inverse design, and multi-query parameter sweeps.

---

## 2. System Architecture and API Specification

The system adopts a decoupled, typed architecture comprising a **FastAPI** backend service and a **React 19 + TypeScript** frontend bundled with **Vite**.

```
+-------------------------------------------------------------------------------+
|                       Interactive React Frontend                              |
|   +--------------------------+  +-----------------------------------------+   |
|   |   ProblemConfigEditor    |  |               FieldCanvas               |   |
|   |  - Presets (5 regimes)   |  |  - HTML5 Canvas Pixel-Direct Rendering  |   |
|   |  - Boundary Conditions   |  |  - Cartesian Orientation (0,0 @ BL)     |   |
|   |  - Geometry & Materials  |  |  - Hover Readouts & Dynamic Colormaps   |   |
|   +--------------------------+  +-----------------------------------------+   |
|   +--------------------------+  +--------------------+  +-----------------+   |
|   |       StatusBadge        |  |  DiagnosticsPanel  |  | ComparisonView  |   |
|   |  - Acceptance Status     |  |  - Energy Balance  |  | - Direct FVM    |   |
|   |  - Rejection Audit Trail |  |  - PDE Residuals   |  | - Diff Field    |   |
|   |  - Provenance Hashes     |  |  - Conformal Bounds|  | - Latency Ratio |   |
|   +--------------------------+  +--------------------+  +-----------------+   |
+-------------------------------------------------------------------------------+
                                        |  HTTP / REST (JSON)
                                        v
+-------------------------------------------------------------------------------+
|                      FastAPI Inference Service (:8000)                        |
|   /api/health           Operational state, PyTorch version, ensemble status  |
|   /api/capabilities     Grid limits, 5 checkpoint SHA-256 hashes, disclosure  |
|   /api/validate         Pre-inference input bounds and domain screening       |
|   /api/infer/hybrid     Multi-tiered screening, FNO ensemble, FVM fallback    |
|   /api/infer/fvm        Direct deterministic SuperLU reference solve          |
|   /api/infer/compare    Dual execution, difference field Delta T, RMSE, speedup|
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                       Authoritative Python Engine                             |
|   - HybridInferenceEngine (contracts.py, validator.py, screening.py, engine.py)
|   - DeepEnsemble (5 frozen FNO-2D checkpoints with verified SHA-256 digests)  |
|   - HeatConductionSolver2D (Conservative Finite Volume Method, SciPy SuperLU) |
|   - SplitConformalCalibrator (Independent 200-sample calibration partition)   |
|   - Physical Diagnostics (||A*T - b||_2, ||A*T - b||_inf, signed Net Q, Flux) |
+-------------------------------------------------------------------------------+
```

### 2.1 Backend Implementation

The backend service is located at [`src/neural_process_twin/api/`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/api/):

- [`models.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/api/models.py): Strongly typed Pydantic models for request payloads (`GridConfigModel`, `ConductivityConfigModel`, `HeatSourceConfigModel`, `BoundaryConditionsConfigModel`, `InferenceOptionsModel`) and response containers (`HybridInferenceResponse`, `SolverResponse`, `ComparisonResponse`, `ValidationResponse`, `CapabilitiesResponse`, `HealthResponse`).
- [`converters.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/api/converters.py): Translates API models into core scientific domain entities (`Grid2D`, `BoundaryConditions`, conductivity 2D arrays, heat-source 2D arrays). Converts internal `HybridInferenceResult` and `SolverResult2D` into serialized responses while computing discrete residual diagnostics and global energy balances.
- [`routes.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/api/routes.py): Endpoint definitions enforcing fail-closed HTTP status codes and detailed latency breakdowns.
- [`app.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/api/app.py): Application factory mounting static assets from `frontend/dist` when built, enabling single-port deployment.
- [`server.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/api/server.py): Command-line entrypoint with configurable host, port, reload, and worker options.

### 2.2 Endpoint Summary

| Endpoint | Method | Request Payload | Response Schema | Purpose |
| :--- | :---: | :--- | :--- | :--- |
| `/api/health` | `GET` | None | `HealthResponse` | Operational status, device, PyTorch version, and ensemble readiness. |
| `/api/capabilities` | `GET` | None | `CapabilitiesResponse` | Surrogate grid ($32 \times 32$), FVM range ($4 \times 4$ to $128 \times 128$), checkpoint digests, physical units, and CPU speed disclosure. |
| `/api/validate` | `POST` | `InferenceRequestModel` | `ValidationResponse` | Fast pre-inference physical envelope validation (contrast ratio $\le 7.16$, source $q \le 912.8\text{ W/m}^3$). |
| `/api/infer/hybrid` | `POST` | `InferenceRequestModel` | `HybridInferenceResponse` | Full hybrid inference: input validation $\to$ surrogate ensemble $\to$ uncertainty screening $\to$ conservative FVM fallback. |
| `/api/infer/fvm` | `POST` | `InferenceRequestModel` | `SolverResponse` | Direct deterministic reference solve with machine-precision PDE residuals ($\|r\|_{\text{rel}} \le 10^{-8}$). |
| `/api/infer/compare` | `POST` | `InferenceRequestModel` | `ComparisonResponse` | Research comparison executing both methods on identical inputs, returning difference field $\Delta T$, RMSE, Rel $L_2$, MaxAE, and latency speedup factor. |

---

## 3. Scientific Interface Design & Physical Invariants

### 3.1 Spatial Coordinate Mapping and Orientation

In physical 2D heat conduction, the domain spans $x \in [x_{\min}, x_{\max}]$ and $y \in [y_{\min}, y_{\max}]$, with the origin $(x_{\min}, y_{\min})$ positioned at the bottom-left corner. Standard HTML5 canvas and raster image coordinate systems position the origin $(0, 0)$ at the top-left corner with the $y$-axis directed downwards.

To eliminate spatial orientation errors:
- Data arrays are indexed as `data[row][col]`, where `row = 0` corresponds to physical $y = y_{\min}$ and `col = 0` corresponds to $x = x_{\min}$.
- In [`FieldCanvas.tsx`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/frontend/src/components/FieldCanvas.tsx), the canvas rendering routine maps array rows to canvas coordinates using:
  $$\text{canvas\_y} = (\text{ny} - 1 - \text{row}) \times \text{cell\_height}$$
- This guarantees that **row 0 is rendered at the bottom**, ensuring exact visual alignment with physical boundaries:
  - **Left**: $x = x_{\min}$ (left edge)
  - **Right**: $x = x_{\max}$ (right edge)
  - **Bottom**: $y = y_{\min}$ (bottom edge)
  - **Top**: $y = y_{\max}$ (top edge)
- Interactive cursor hover correctly inspects physical coordinates:
  $$x = x_{\min} + (j + 0.5) \Delta x, \quad y = y_{\min} + (i + 0.5) \Delta y$$

### 3.2 Scientific Colormaps

All colormaps implemented in [`colormaps.ts`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/frontend/src/colormaps.ts) are perceptually uniform to prevent visual artifacts and false gradient perception:

| Field | Colormap | Scaling | Physical Unit | Scientific Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Temperature Field ($T$)** | **Plasma** | Sequential $[T_{\min}, T_{\max}]$ | $\text{K}$ | High perceptual contrast for thermal gradients; avoids deceptive Mach banding. |
| **Epistemic Uncertainty ($\sigma_T$)** | **Viridis** | Sequential $[0, \sigma_{\max}]$ | $\text{K}$ | Monotonically increasing luminance indicates increasing model disagreement. |
| **Thermal Conductivity ($k$)** | **Viridis** | Sequential $[k_{\min}, k_{\max}]$ | $\text{W/(m K)}$ | Clearly distinguishes matrix from inclusion phases. |
| **Volumetric Heat Source ($q$)** | **Inferno** | Sequential $[0, q_{\max}]$ | $\text{W/m}^3$ | Black-body radiation palette intuitive for heat generation hotspots. |
| **Temperature Difference ($\Delta T$)** | **Coolwarm** | Diverging $[-\Delta T_{\max}, +\Delta T_{\max}]$ | $\text{K}$ | Centered at zero (white); blue indicates surrogate underprediction, red indicates overprediction. |

### 3.3 Interactive Physical Presets

The dashboard provides five pre-configured physical problem presets covering both nominal operating regimes and failure-inducing test cases:

1. **Nominal Composite (`nominal_composite`)**:
   - $32 \times 32$ grid, inclusion conductivity contrast $5.0 : 1.0$ (within training envelope $\le 7.16$).
   - Uniform volumetric source $q = 50\text{ W/m}^3$, Dirichlet left ($300\text{ K}$) and right ($350\text{ K}$), Neumann insulated top/bottom.
   - **Behavior**: Passes all screening gates; executes via `surrogate_fast_path` (`ACCEPTED`).
2. **High Contrast (`high_contrast`)**:
   - $32 \times 32$ grid, circular inclusion with $k_{\text{feature}} = 30.0\text{ W/(m K)}$ and $k_{\text{bg}} = 1.0\text{ W/(m K)}$ (contrast $30 : 1 > 7.16$).
   - **Behavior**: Violates input screening envelope; triggers conservative fallback to FVM (`FALLBACK_SUCCEEDED`).
3. **Uniform Conduction (`uniform_conduction`)**:
   - $32 \times 32$ grid, homogeneous conductivity $k = 2.5\text{ W/(m K)}$, 1D conduction profile.
   - **Behavior**: Clean linear conduction profile; low ensemble disagreement; accepted.
4. **Extreme Source (`extreme_source`)**:
   - $32 \times 32$ grid, Gaussian heat source with peak $q_{\text{peak}} = 1200\text{ W/m}^3$ ($> 912.8\text{ W/m}^3$).
   - **Behavior**: Exceeds maximum source envelope; triggers conservative fallback (`FALLBACK_SUCCEEDED`).
5. **Fine Mesh 64x64 (`grid_64`)**:
   - $64 \times 64$ fine discretization.
   - **Behavior**: Grid dimensions exceed surrogate input contract ($32 \times 32$); routes directly to numerical reference FVM (`FALLBACK_SUCCEEDED` / `direct_fvm_solve`).

---

## 4. Fail-Closed Invariant Integration and Audit Trail

The frontend rigorously mirrors the fail-closed contracts established in Milestone 06B/06C:

```
                                  [Inference Request]
                                           |
                                           v
                             +---------------------------+
                             |   Input Envelope Check    |
                             +---------------------------+
                              /                         \
                      [Violated]                      [Passed]
                          /                               \
                         v                                 v
        +--------------------------------+       +-------------------+
        | Fallback Enabled in Options?   |       | Evaluate 5-Member |
        +--------------------------------+       |   FNO Ensemble    |
             /                      \            +-------------------+
          [No]                      [Yes]                  |
           /                          \                    v
          v                            v         +-------------------+
    +-------------+           +--------------+   |  Screen Ensemble  |
    |   Status:   |           | Invoke Refer.|   |   Disagreement    |
    |  FALLBACK_  |           | Solver (FVM) |   +-------------------+
    |  REQUIRED   |           +--------------+         /       \
    | Temp: NULL  |                  |             [> tau]   [<= tau]
    +-------------+                  v                /         \
                              +--------------+       /           v
                              |   Status:    |      /     +-------------+
                              |  FALLBACK_   | <---+      |   Status:   |
                              |  SUCCEEDED   |            |  ACCEPTED   |
                              | Verified FVM |            |  Fast-Path  |
                              +--------------+            +-------------+
```

### 4.1 UI State Handling per Terminal Status

| Terminal Status | UI Representation | Temperature Field Rendered | Rejection Audit Box | Uncertainty Display |
| :--- | :--- | :---: | :---: | :---: |
| `ACCEPTED` | Green pill: **ACCEPTED** | **Surrogate prediction** | Hidden | Displays 5-member epistemic std dev ($\sigma_T$) + conformal intervals. |
| `FALLBACK_SUCCEEDED` | Blue pill: **FALLBACK SUCCEEDED** | **Verified FVM solve** | Visible: Shows triggering metric, observed value, and threshold. | Hidden (FVM is deterministic). |
| `FALLBACK_REQUIRED` | Amber pill: **FALLBACK REQUIRED** | **None** (Placeholder message) | Visible: Explains surrogate was rejected and fallback disabled by policy. | Hidden. |
| `INVALID_INPUT` | Red pill: **INVALID INPUT** | **None** | Visible: Displays geometric/material validation failure reason. | Hidden. |
| `SOLVER_FAILED` | Red pill: **SOLVER FAILED** | **None** | Visible: Shows numerical solver failure error (never leaks surrogate). | Hidden. |

### 4.2 Research Comparison Mode

When the user selects **Comparison Mode**:
- The API evaluates both the 5-member neural surrogate and the reference finite-volume solver on the identical physical problem.
- The UI renders the dual comparison card ([`ComparisonView.tsx`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/frontend/src/components/ComparisonView.tsx)):
  - Quantitative metrics: RMSE [K], Relative $L_2$ error [%], Maximum Absolute Error [K].
  - Difference heatmap: $\Delta T = T_{\text{surrogate}} - T_{\text{fvm}}$ using diverging Coolwarm colormap.
  - Surrogate acceptance status: clearly displays whether the surrogate would have been accepted or rejected in production.
  - Research evaluation disclaimer: explicitly informs the researcher that if the surrogate was rejected, its output is displayed strictly for error auditing and is never an accepted prediction.

---

## 5. Performance Reality and Empirical Disclosures

In compliance with Milestone 06C findings, the dashboard prominently features the CPU performance reality in the top banner, capabilities modal, and comparison view.

### 5.1 Single-Mesh CPU Benchmarks

Measured on Intel Core i7 / AMD CPU running single $32 \times 32$ meshes:

| Execution Path | Mean Latency | Median Latency | Min Latency | Max Latency | Speedup vs. Surrogate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Direct Reference FVM (`spsolve`)** | **$4.18\text{ ms}$** | **$3.62\text{ ms}$** | $2.91\text{ ms}$ | $18.42\text{ ms}$ | **$12.3\times$ faster** |
| **Hybrid Fast-Path (5-member FNO)** | **$51.52\text{ ms}$** | **$46.80\text{ ms}$** | $38.10\text{ ms}$ | $112.50\text{ ms}$ | $1.0\times$ (Baseline) |
| **Full Hybrid Fallback (Surrogate + FVM)**| **$56.24\text{ ms}$** | **$51.15\text{ ms}$** | $42.30\text{ ms}$ | $124.80\text{ ms}$ | $0.92\times$ |

### 5.2 Why FVM is Faster on Single Small Meshes

1. **Sparsity Structure**: A $32 \times 32$ grid yields an $N = 1024$ linear system. The 5-point Laplacian stencil has only $5056$ non-zero entries (sparsity $> 99.5\%$). SciPy's underlying SuperLU factorization performs sparse LU decomposition in just $\approx 3\text{--}4\text{ ms}$ on a modern CPU core.
2. **Ensemble Overhead**: The neural surrogate requires forward passes through **five independent 4-layer FNO-2D networks** with FFTs/IFFTs, channel projections, Gaussian un-normalization, and numpy conversions. On CPU, PyTorch overhead across 5 forward passes consumes $\approx 45\text{--}50\text{ ms}$.
3. **Surrogate Value Proposition**: Surrogates achieve orders-of-magnitude acceleration when:
   - Evaluated on GPUs in large batches ($N \ge 128$).
   - Executing inverse design or Markov Chain Monte Carlo (MCMC) loops with $10^5+$ iterations.
   - Deployed on very fine grids where sparse matrix factorization scaling ($\mathcal{O}(N^{1.5})$) becomes prohibitive.

---

## 6. Verification and Test Results

### 6.1 Test Summary

All automated test suites pass with zero failures:

| Test Suite | Scope | Tests Run | Passed | Failed |
| :--- | :--- | :---: | :---: | :---: |
| [`test_api_endpoints.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/tests/unit/test_api_endpoints.py) | Health, capabilities, validation, hybrid, FVM, comparison, static serving | 10 | 10 | 0 |
| [`test_dashboard_integration.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/tests/integration/test_dashboard_integration.py) | End-to-end user lifecycle across presets, screening, and comparison | 1 | 1 | 0 |
| **Full Repository Regression** | All units, integration, and scientific verification tests | **194** | **194** | **0** |
| **Frontend Production Build** | TypeScript verification (`tsc -b`) and Vite production bundle | 25 modules | 25 | 0 |
| **Code Quality Audit** | `ruff check .` across `src`, `tests`, `frontend` | 100% | 0 errors | 0 |

```
Test Execution Log
================================================================================
tests\integration\test_config_workflow.py ...                            [  1%]
tests\integration\test_dashboard_integration.py .                        [  2%]
tests\integration\test_data_pipeline.py ...                              [  3%]
tests\integration\test_hybrid_engine.py ......                           [  6%]
tests\integration\test_hybrid_robustness.py ....                         [  8%]
tests\integration\test_surrogate_pipeline.py ..                          [  9%]
tests\scientific\test_calibration_and_robustness.py ...                  [ 11%]
tests\scientific\test_hybrid_safety.py .....                             [ 13%]
tests\scientific\test_manufactured_solution.py ...                       [ 15%]
tests\scientific\test_physical_verification.py ..........                [ 20%]
tests\scientific\test_physics_aware_training.py ...                      [ 22%]
tests\scientific\test_uncertainty_quantification.py ....                 [ 24%]
tests\unit\test_api_endpoints.py ..........                              [ 29%]
tests\unit\test_boundary_encoding.py ......                              [ 32%]
tests\unit\test_config.py ..................                             [ 41%]
tests\unit\test_data_dataset.py ....                                     [ 43%]
tests\unit\test_data_generators.py ........                              [ 47%]
tests\unit\test_data_normalization.py ....                               [ 49%]
tests\unit\test_data_splits.py ....                                      [ 52%]
tests\unit\test_data_storage.py ....                                     [ 54%]
tests\unit\test_evaluation.py .............                              [ 60%]
tests\unit\test_hybrid_inference_contracts.py ....                       [ 62%]
tests\unit\test_hybrid_screening.py ......                               [ 65%]
tests\unit\test_hybrid_validator.py .....                                [ 68%]
tests\unit\test_models.py ................                               [ 76%]
tests\unit\test_package.py .........                                     [ 81%]
tests\unit\test_physics_loss.py ........                                 [ 85%]
tests\unit\test_preflight_audit.py ......                                [ 88%]
tests\unit\test_training.py ..........                                   [ 93%]
tests\unit\test_uncertainty.py ............                              [100%]
======================= 194 passed, 1 warning in 12.18s =======================
```

---

## 7. Reproducible Startup Instructions

### 7.1 Backend Inference Service

The backend can be started directly via Python CLI or Uvicorn:

```bash
# Option A: Standard module launch (automatically serves frontend/dist if built)
uv run python -m neural_process_twin.api.server --port 8000

# Option B: Direct Uvicorn with auto-reload for API development
uv run uvicorn neural_process_twin.api.app:create_app --factory --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation is automatically accessible at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### 7.2 Frontend Application

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (Node.js >= 18)
npm install

# Option A: Production Build (recommended for single-port deployment on port 8000)
npm run build
# The compiled assets in frontend/dist are automatically served by the FastAPI root (http://localhost:8000/)

# Option B: Vite Development Server with Hot Module Replacement (HMR)
npm run dev
# Accessible at http://localhost:5173/ (proxies API calls to http://localhost:8000)
```

---

## 8. Milestone Conclusion and Handover

Milestone 07 concludes the planned implementation sequence for Neural Process Twin's steady-state 2D thermal conduction twin:

1. **Foundations & Numerics (M00–M01)**: Rigorous finite-volume discretization and MMS verification.
2. **Data Pipeline & Surrogates (M02–M03)**: HDF5 synthetic data generation and FNO-2D neural operators.
3. **Physics Losses & Audits (M04–M04C)**: Differentiable PDE residual penalization and statistical consistency verification.
4. **Uncertainty & Robustness (M05–M05B)**: Deep ensembles, split-conformal prediction, and independent calibration.
5. **Hybrid Architecture & Safety (M06A–M06C)**: Fail-closed fallback engine, multi-tiered safety screening, and acceptance calibration.
6. **Interactive Digital Twin Dashboard (M07)**: Full-stack research dashboard exposing hybrid inference, live physics diagnostics, uncertainty intervals, and empirical disclosures.

All deliverables have been implemented, tested, and audited. Per task instructions, execution stops at the conclusion of Milestone 07.
