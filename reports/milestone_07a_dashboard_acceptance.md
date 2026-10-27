# Milestone 07A — Dashboard Acceptance and Visual QA

**Neural Process Twin: Live Application Smoke Test, Scientific Contract Audit, Visual QA, and Robustness Verification**  
**Date**: October 2026  
**Status**: Acceptance Audit Completed and Verified  
**Author**: Computational Science & Scientific ML Engineering Team  
**Deliverable**: Comprehensive Acceptance Audit Report, Live API Benchmarks, 9 Visual QA Screenshots, and Robustness Test Suites  

---

## 1. Executive Summary

Milestone 07A performs a comprehensive acceptance and quality assurance audit of the **Neural Process Twin Interactive Scientific Digital Twin Dashboard** completed in Milestone 07. The audit tests the actual running application behavior, executes live HTTP API calls across all physical presets, verifies strict scientific contracts and fail-closed invariants, inspects visual rendering in Microsoft Edge via headless browser automation, exercises extreme robustness edge cases, and benchmarks end-to-end performance.

### 1.1 Non-Negotiable Scientific Guardrails Respected

1. **Live Runtime Execution**: All verification was performed on live, running instances of the FastAPI inference service (`http://127.0.0.1:8000`) and the compiled React 19 + TypeScript dashboard.
2. **Authoritative Physics in Python**: All PDE solves, finite-volume linear systems, residuals, and global energy balance integrations execute strictly in Python backend services; client-side JavaScript performs only raster canvas rendering.
3. **Fail-Closed Verification**: Rejected surrogate outputs are verified never to leak into client responses as accepted predictions. Fallback solutions return strictly the verified numerical solve with explicit provenance.
4. **Epistemic Honesty**: Direct finite-volume solves strictly return `null` for ensemble disagreement and conformal prediction intervals. No artificial uncertainty is fabricated for deterministic algorithms.
5. **Visual QA Ground Truth**: Rendered application states were visually inspected and captured across 9 representative states using headless browser automation with pixel-direct inspection.
6. **Performance Transparency**: The dashboard top banner, capabilities modal, and comparison views prominently disclose that direct sparse finite-volume solve with SuperLU is $\approx 11\text{--}13\times$ faster on single $32 \times 32$ CPU problems than evaluating the 5-member neural surrogate ensemble.

---

## 2. Phase 1 — Live Application Smoke Test

### 2.1 Exact Service Launch Commands

The backend service was launched using the documented CLI entrypoint:

```bash
# Set proxy bypass for localhost connections in environment
$env:NO_PROXY="localhost,127.0.0.1"

# Launch backend service on port 8000 (serves frontend/dist at root)
uv run python -m neural_process_twin.api.server --host 127.0.0.1 --port 8000
```

- **Root Endpoint (`GET /`)**: HTTP `200 OK` (serves compiled `frontend/dist/index.html`).
- **Health Endpoint (`GET /api/health`)**: HTTP `200 OK`
  ```json
  {
    "status": "healthy",
    "version": "0.1.0",
    "device": "cpu",
    "torch_version": "2.14.1+cpu",
    "ensemble_loaded": true,
    "ensemble_member_count": 5
  }
  ```
- **Capabilities Endpoint (`GET /api/capabilities`)**: HTTP `200 OK`
  - Surrogate resolution: `[32, 32]`
  - Ensemble members: 5 checkpoints with SHA-256 digests
  - Performance disclosure: `"SCIENTIFIC PERFORMANCE REALITY: On a single 32x32 mesh on CPU, direct reference finite-volume solve with SciPy SuperLU is approximately 11-13x faster..."`

### 2.2 Live Preset Execution Results

Each of the five problem presets was evaluated against the live service using [`scripts/smoke_test_presets.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/scripts/smoke_test_presets.py):

| Problem Preset | Validation Result | Hybrid Status & Path | Total Latency | Surrogate vs Fallback Latency | Output Temp Range | Rejection Reason | FVM Rel Residual | Comparison RMSE | FVM Speedup vs Surrogate |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Nominal Composite** ($32 \times 32$) | Valid: `True`<br>Surr: `True` | `fallback_succeeded`<br>`solver_fallback_path` | $86.98\text{ ms}$ | Surr: $76.24\text{ ms}$<br>FVM: $6.67\text{ ms}$ | $301.24\text{ K}$ to $349.54\text{ K}$ | `high_ensemble_disagreement` ($5.05\text{ K} > 2.50\text{ K}$) | $2.24 \times 10^{-15}$ | $2.38\text{ K}$ | **$12.27\times$ faster** |
| **High Contrast** ($30:1$, $32 \times 32$) | Valid: `True`<br>Surr: `False` | `fallback_succeeded`<br>`solver_fallback_path` | $75.09\text{ ms}$ | Surr: $65.16\text{ ms}$<br>FVM: $6.11\text{ ms}$ | $301.34\text{ K}$ to $349.41\text{ K}$ | `high_input_contrast` ($30.0 > 7.16$) | $1.67 \times 10^{-14}$ | $34.71\text{ K}$ | **$12.33\times$ faster** |
| **Uniform Conduction** ($32 \times 32$) | Valid: `True`<br>Surr: `True` | `accepted`<br>`surrogate_fast_path` | $72.06\text{ ms}$ | Surr: $67.76\text{ ms}$<br>FVM: $0.00\text{ ms}$ | $300.27\text{ K}$ to $350.29\text{ K}$ | None (`ACCEPTED`) | $1.27 \times 10^{-15}$ | $0.53\text{ K}$ | **$13.19\times$ faster** |
| **Extreme Source** ($1200\text{ W/m}^3$) | Valid: `True`<br>Surr: `False` | `fallback_succeeded`<br>`solver_fallback_path` | $76.98\text{ ms}$ | Surr: $66.69\text{ ms}$<br>FVM: $6.30\text{ ms}$ | $300.01\text{ K}$ to $312.44\text{ K}$ | `high_heat_source` ($1179.8 > 912.8$) | $8.31 \times 10^{-16}$ | $2.46\text{ K}$ | **$12.02\times$ faster** |
| **Fine Mesh 64x64** ($64 \times 64$) | Valid: `True`<br>Surr: `False` | `fallback_succeeded`<br>`solver_fallback_path` | $22.99\text{ ms}$ | Surr: $0.00\text{ ms}$<br>FVM: $22.89\text{ ms}$ | $300.62\text{ K}$ to $349.77\text{ K}$ | `incompatible_grid_dimensions` ($64 \ne 32$) | $3.63 \times 10^{-15}$ | N/A (Surr Skipped) | **$1.03\times$** |

### 2.3 Smoke Test Observations

1. **Safety Gate Effectiveness**: On the **High Contrast** preset ($30:1$), the surrogate error is catastrophic (RMSE $= 34.71\text{ K}$). The multi-tiered screening policy successfully triggered `high_input_contrast` and invoked reference FVM fallback, returning a verified solution with machine-precision residual ($1.67 \times 10^{-14}$) and preventing this $34.7\text{ K}$ error from being returned.
2. **Zero-Overhead Routing for Incompatible Grids**: On the **Fine Mesh 64x64** preset, the engine recognized that $64 \times 64$ does not match the surrogate contract ($32 \times 32$), completely bypassing the neural forward pass (surrogate latency $= 0.00\text{ ms}$) and solving directly via FVM in $22.89\text{ ms}$.
3. **Surrogate Fast-Path**: On the **Uniform Conduction** preset, the surrogate passed all gates, executing via `surrogate_fast_path` in $72.06\text{ ms}$ with an RMSE of only $0.53\text{ K}$.

---

## 3. Phase 2 — Scientific API Contract Audit

The scientific integrity audit script [`scripts/audit_scientific_contracts.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/scripts/audit_scientific_contracts.py) systematically audited core physics invariants:

### 3.1 Physical Input Validity vs. Surrogate Domain Eligibility

- **Valid Physics outside Surrogate Domain**: A problem with conductivity contrast $30:1$ returns `is_valid: true`, `surrogate_compatible: false`, with a clear warning explaining that contrast $30:1$ exceeds the training envelope ($7.16:1$).
- **Physically Invalid Negative Conductivity**: Setting $k \le 0$ ($k = -5.0\text{ W/(m K)}$) is immediately rejected at the API schema boundary with HTTP `422 Unprocessable Entity` (`"Input should be greater than 0"`).
- **Physically Invalid Pure Neumann Boundaries**: Specifying Neumann boundary conditions on all four sides is rejected with HTTP `422` (`"Pure-Neumann boundary conditions on all four boundaries are singular without a gauge reference"`).

### 3.2 High-Contrast FVM Convergence

Evaluating a severe $50:1$ conductivity contrast problem ($k_{\text{feature}} = 50.0, k_{\text{bg}} = 1.0$) with the reference solver (`POST /api/infer/fvm`) produced:
- Status: `fallback_succeeded`
- Execution path: `direct_fvm_solve`
- Relative linear algebraic residual: $\|A T - b\|_2 / \|b\|_2 = 2.74 \times 10^{-14}$ (exact machine precision).

### 3.3 Fail-Closed Guarantees (Zero Output Leakage)

- When an out-of-envelope problem is run with `enable_fallback: false`, the engine returns `status: "fallback_required"`, `temperature: null`, and `rejection.code: "fallback_disabled"`. Under no circumstances is the rejected surrogate output returned as `temperature`.
- When `enable_fallback: true`, the returned `temperature` array is identical to the independent FVM reference solve ($|T_{\text{hybrid}} - T_{\text{fvm}}|_{\max} = 0.00\text{ K}$).

### 3.4 Epistemic Honesty for Deterministic FVM

Evaluating direct FVM solves (`POST /api/infer/fvm`) verified:
- `ensemble_disagreement_k = null`
- `mean_disagreement_k = null`
- `max_disagreement_k = null`
- `conformal_intervals = null`
No uncertainty metrics are fabricated for deterministic solves.

### 3.5 Comparison Mode Research Notice

When running comparison mode on a rejected problem, the response includes:
- `surrogate_acceptance_status: "fallback_succeeded"`
- `evaluation_notice`: `"Surrogate was REJECTED (Conductivity contrast ratio (30.00:1) exceeds training domain threshold (7.16:1).) and replaced by FVM fallback in the hybrid result. The raw surrogate field is displayed here strictly for research comparison."`

### 3.6 Spatial Coordinate and Array Orientation Consistency

To confirm that the 2D array $[y, x]$ aligns with the physical domain, an asymmetric Dirichlet problem was solved:
- Left: $300\text{ K}$
- Right: $400\text{ K}$
- Bottom: $320\text{ K}$
- Top: $380\text{ K}$

Actual boundary cell temperatures in the returned array:
- **Left boundary cells** (`col = 0`): Mean $T = 304.79\text{ K}$ (matches Dirichlet $300\text{ K}$).
- **Right boundary cells** (`col = 31`): Mean $T = 395.21\text{ K}$ (matches Dirichlet $400\text{ K}$).
- **Bottom boundary cells** (`row = 0`): Mean $T = 322.87\text{ K}$ (matches Dirichlet $320\text{ K}$).
- **Top boundary cells** (`row = 31`): Mean $T = 377.13\text{ K}$ (matches Dirichlet $380\text{ K}$).

In [`FieldCanvas.tsx`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/frontend/src/components/FieldCanvas.tsx), vertical coordinates are inverted during canvas drawing:
$$\text{canvas\_y} = (ny - 1 - \text{row}) \times \text{cell\_height}$$
This guarantees that **row 0 is rendered at the bottom**, achieving strict alignment between the visual canvas, physical axes, and boundary conditions.

---

## 4. Phase 3 — Visual QA Inspection

Visual inspection was conducted using automated headless Microsoft Edge driven by `puppeteer-core` ([`scripts/visual_qa.js`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/scripts/visual_qa.js)). Nine high-resolution screenshots were captured and verified in [`reports/screenshots/`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/reports/screenshots/):

```
reports/screenshots/
├── 01_initial_dashboard.png           (77.3 KB) - Initial dashboard with preset controls and educational guide
├── 02_provenance_modal.png            (114.4 KB) - Audited provenance modal with 5 verified SHA-256 digests
├── 03_nominal_composite_simulation.png (84.3 KB) - Nominal composite simulation with plasma field and diagnostics
├── 04_epistemic_uncertainty.png       (93.3 KB) - Epistemic uncertainty field (viridis colormap)
├── 05_canvas_hover_inspection.png     (89.9 KB) - Interactive cursor inspection with coordinate and scalar readout
├── 06_high_contrast_fallback.png      (72.1 KB) - Fallback succeeded badge and structured rejection audit box
├── 07_research_comparison_mode.png    (100.4 KB) - Research comparison card with difference heatmap and metrics
├── 08_grid_64_fine_mesh.png           (86.2 KB) - 64x64 fine mesh routed directly to reference FVM
└── 09_responsive_tablet_view.png      (48.4 KB) - Tablet viewport (768x1024) single-column responsive layout
```

### 4.1 Visual Inspection Checklist

| Visual Inspection Item | Target Element / Behavior | Observed Result | Status |
| :--- | :--- | :--- | :---: |
| **Top Banner & Branding** | Blue disclosure banner & "Ensemble Online (5/5)" pill | Banner text visible, green pulse dot active, clean contrast | **VERIFIED** |
| **Audited Provenance Modal** | Click "Audited Provenance" button | Displays all 5 SHA-256 digests, normalizer bounds, training limits | **VERIFIED** |
| **Field Rendering & Colormap** | 2D Temperature Field on HTML5 Canvas | Smooth plasma colormap rendering, correct boundary gradient | **VERIFIED** |
| **Colorbar Scaling & Ticks** | Vertical colorbar alongside canvas | Top tick equals field maximum, bottom tick equals field minimum | **VERIFIED** |
| **Cartesian Orientation** | Bottom-left origin $(0, 0)$ | $y=0$ at bottom, $y=1$ at top, $x=0$ at left, $x=1$ at right | **VERIFIED** |
| **Cursor Hover Inspection** | Mouseover canvas cell | Tooltip readout: `Coords: (0.484 m, 0.516 m) | Cell: [j=16, i=15] | Value: 330.532 K` | **VERIFIED** |
| **Rejection Audit Box** | Fallback or rejection state | Amber box: `Triggering Metric: k_contrast_ratio | Observed: 30.00 | Threshold: 7.16` | **VERIFIED** |
| **Conservation Diagnostics** | Global energy balance breakdown | $Q_{\text{gen}}, Q_{\text{out}}, \Delta Q_{\text{net}}$, signed imbalance pill (green/red), 4 face heat flows | **VERIFIED** |
| **Latency Breakdown** | Pipeline timing cards | Input validation, surrogate inference, diagnostics, fallback, total | **VERIFIED** |
| **Research Comparison Card** | Dual surrogate vs FVM comparison | Coolwarm difference heatmap, RMSE ($34.71\text{ K}$ on 30:1), speedup factor ($12.33\times$) | **VERIFIED** |
| **Responsive Tablet Layout** | Viewport width 768px | Grid collapses to single column, editor on top, canvas below | **VERIFIED** |

---

## 5. Phase 4 — Robustness and Stress Testing

Edge cases and stress scenarios were audited using [`scripts/robustness_test.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/scripts/robustness_test.py):

### 5.1 Invalid and Non-Finite Inputs

1. **Negative Conductivity ($k = -10.0$)**: Rejected with HTTP `422 Unprocessable Entity` (`"Input should be greater than 0"`).
2. **Zero Conductivity ($k = 0.0$)**: Rejected with HTTP `422` (`"Input should be greater than 0"`).
3. **Negative Thickness ($t = -0.5$)**: Rejected with HTTP `422` (`"Input should be greater than 0"`).
4. **Zero Grid Cells ($nx = 0$)**: Rejected with HTTP `422` (`"Input should be greater than 0"`).

### 5.2 Resolution Handling

- **Unsupported Low Resolution ($16 \times 16$)**: The surrogate contract strictly requires $32 \times 32$. On `/api/infer/hybrid`, the engine rejects the surrogate via `incompatible_grid_dimensions`, seamlessly falling back to FVM and returning a verified $16 \times 16$ temperature field in $15.4\text{ ms}$.

### 5.3 Solver Timeout and Policy Enforcement

- **Unrealistic Timeout ($< 0.1\text{ s}$)**: Setting `solver_timeout_seconds: 0.000001` is rejected at the API schema boundary with HTTP `422` (`"Input should be greater than 0.1"`).
- **Fail-Closed on Fallback Disabled**: When an out-of-envelope problem has `enable_fallback: false`, the response returns `status: "fallback_required"`, `temperature: null`, and `rejection.code: "fallback_disabled"`. Under no failure scenario is an unverified surrogate field exposed.

### 5.4 Concurrency and Rapid Requests

- 16 rapid concurrent requests alternating between `/api/infer/hybrid` and `/api/infer/fvm` executed across 4 worker threads.
- **Results**: 100% completed with HTTP `200 OK` in $0.26\text{ s}$ total wall-clock time (mean throughput $\approx 16.4\text{ ms/request}$). No deadlocks, memory leaks, or race conditions were observed.

---

## 6. Defects Discovered and Fixes Applied

| Defect / Anomaly Discovered | Root Cause Analysis | Fix & Resolution Applied | Verification Status |
| :--- | :--- | :--- | :---: |
| **Localhost Proxy Interception (HTTP 503)** | Windows system environment variables `HTTP_PROXY` and `HTTPS_PROXY` were set to `http://10.10.78.22:3128`, but `NO_PROXY` was unset. Local requests to `127.0.0.1:8000` were forwarded to Squid, which rejected localhost connections with 503. | Configured `$env:NO_PROXY="localhost,127.0.0.1"` in startup scripts; added `--no-proxy-server` to Edge browser automation arguments; added `trust_env=False` to httpx test clients. | **RESOLVED & VERIFIED** |
| **Hover Event Triggering in Puppeteer** | Puppeteer `page.mouse.move()` did not trigger synthetic React canvas events if canvas was outside the initial viewport offset. | Replaced bounding box manual movement with `canvas.hover()`, which reliably scrolls to the canvas and dispatches native DOM mouseover/mousemove events. | **RESOLVED & VERIFIED** |
| **Ruff Code Style in Test Scripts** | New audit test scripts had uppercase variable `T` (N806) and unsorted import blocks (I001). | Renamed `T` to `t_field` and formatted imports via `ruff check --fix .` and `ruff format`. | **RESOLVED & VERIFIED** |

---

## 7. Remaining Known Limitations

1. **CPU Speed Reality for Small Meshes**: On single $32 \times 32$ meshes on CPU, direct sparse numerical solve with SuperLU is $\approx 11\text{--}13\times$ faster than evaluating the 5-member neural surrogate ensemble. This is not a defect; it is a fundamental property of sparse linear algebra on small matrices ($\mathcal{O}(N)$ bandwidth). The dashboard transparently communicates this in the header banner, capabilities disclosure, and comparison views.
2. **Fixed-Resolution Surrogate ($32 \times 32$)**: The frozen FNO-2D ensemble is trained for $32 \times 32$ discretizations. Arbitrary mesh resolutions ($4 \times 4$ to $128 \times 128$) are routed automatically to the reference FVM solver without crashing or corrupting data.
3. **Provisional Screening Thresholds**: Disagreement screening thresholds ($\tau_{\text{mean}} = 2.5\text{ K}, \tau_{\max} = 5.0\text{ K}$) remain provisional as calibrated in Milestone 06C on the 200-sample calibration set.

---

## 8. Final Acceptance Recommendation

### Verdict: **FULL ACCEPTANCE RECOMMENDED**

The Milestone 07 Interactive Scientific Digital Twin Dashboard and Milestone 07A Acceptance Verification demonstrate complete compliance with all physical, numerical, safety, and software engineering criteria:
- **Test Suite**: 194/194 automated Python tests passing with 0 errors.
- **Code Quality**: 100% clean Ruff check across `src/`, `tests/`, and `scripts/`.
- **Frontend Quality**: Zero TypeScript errors (`tsc -b`), production Vite build verified.
- **Visual QA**: 9 representative visual states captured and verified in Microsoft Edge.
- **Fail-Closed Integrity**: Proven zero leakage of rejected surrogate outputs.

Execution stops at the conclusion of Milestone 07A in accordance with project instructions. Do not proceed to Milestone 08.
