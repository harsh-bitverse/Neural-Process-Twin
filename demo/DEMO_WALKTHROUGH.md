# Neural Process Twin — Public Demonstration & Walkthrough Guide

This document outlines the step-by-step demonstration walkthrough for researchers, engineering evaluators, and reviewers.

---

## Interactive Walkthrough Steps

### Step 1: Automated Script Walkthrough
To execute the complete 8-step interactive demonstration from the command line:

```bash
uv run python demo/run_interactive_demo.py
```

This runs:
1. **Problem Configuration:** $64\times 64$ grid, circular thermal inclusion ($k=5.0\text{ W/(m}\cdot\text{K)}$ in $1.0\text{ W/(m}\cdot\text{K)}$ background), central Gaussian hotspot ($300\text{ W/m}^3$), and mixed Dirichlet-Neumann boundaries ($300\text{ K} \to 350\text{ K}$).
2. **Verified Numerical Solve:** Direct SuperLU FVM factorization with machine-precision relative residual ($3.54 \times 10^{-15}$).
3. **Experimental Neural Inference:** Forward pass of $A_3$ `RefinedFNO2D`.
4. **Physics Smoothing:** Post-hoc $A_4$ Selective Boundary Jacobi smoother executing on the outer 2-cell boundary ring ($<1\text{ ms}$).
5. **Accuracy Comparison:** Evaluates RMSE ($7.35\text{ K}$) and MaxAE ($18.42\text{ K}$).
6. **Boundary Error Inspection:** Demonstrates outer boundary ring error reduction from $16.74\text{ K} \to 14.77\text{ K}$.
7. **Physical Diagnostics:** Demonstrates heat flow imbalance ($90.6\%$).
8. **Scientific Governance Verdict:** Explicitly documents why the prediction carries `is_certified: false` because MaxAE exceeds the $10\text{ K}$ engineering tolerance ceiling.
9. **Visual Output:** Saves a 4-panel visual comparison to `artifacts/figures/demo_showcase.png`.

---

### Step 2: Interactive Web Dashboard Walkthrough

1. **Launch Platform:**
   ```bash
   uv run uvicorn neural_process_twin.api.app:create_app --factory --host 0.0.0.0 --port 8000
   ```
2. **Open Dashboard:** Navigate to [http://localhost:8000](http://localhost:8000).
3. **Preset Selection:**
   - In the sidebar preset selector, choose **Thermal Inclusion Hotspot**.
   - Observe the 2D conductivity preview (circular feature) and volumetric heat source.
4. **Mode 1: Verified Numerical Mode:**
   - Under Mode Selection, select **Verified Numerical Mode (Conservative FVM)**.
   - Click **Run Simulation**.
   - Observe the smooth temperature field rendered on the HTML5 canvas.
   - Inspect the diagnostic telemetry: Relative residual $< 10^{-14}$, exact physical energy balance $< 10^{-12}$, green "Certified Numerical Ground Truth" badge.
5. **Mode 2: Experimental Neural Mode:**
   - Toggle to **Experimental Neural Mode (A3 RefinedFNO)**.
   - Notice the prominent amber disclaimer banner: *"Experimental Neural Mode is an uncertified surrogate approximation."*
   - Toggle **Apply Selective Boundary Jacobi (A4)** on.
   - Click **Run Simulation**.
   - Observe the sub-50 ms inference latency and Pointwise Jacobi indicator.
6. **Mode 3: Dual-Mode Comparison:**
   - Switch to **Research Comparison Mode**.
   - The UI automatically renders both temperature fields side-by-side with an interactive difference heatmap ($\Delta T = T_{\text{neural}} - T_{\text{fvm}}$).
   - Hover over cells to see exact numerical vs neural values and error residuals.

---

### Step 3: REST API Demonstration via `curl`

#### 1. Verified Numerical Simulation
```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "numerical_fvm",
    "grid": {"nx": 64, "ny": 64, "lx": 1.0, "ly": 1.0},
    "conductivity": {"pattern_type": "uniform", "uniform_value": 2.5},
    "source": {"pattern_type": "uniform", "uniform_value": 100.0},
    "boundary_conditions": {
      "left": {"side": "left", "type": "dirichlet", "value": 300.0},
      "right": {"side": "right", "type": "dirichlet", "value": 350.0},
      "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
      "top": {"side": "top", "type": "neumann", "value": 0.0}
    }
  }'
```

#### 2. Experimental Neural Simulation (Optimized, Sub-50ms)
```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "experimental_neural_corrected",
    "compute_diagnostics": false,
    "grid": {"nx": 64, "ny": 64, "lx": 1.0, "ly": 1.0},
    "conductivity": {"pattern_type": "uniform", "uniform_value": 2.5},
    "source": {"pattern_type": "uniform", "uniform_value": 100.0},
    "boundary_conditions": {
      "left": {"side": "left", "type": "dirichlet", "value": 300.0},
      "right": {"side": "right", "type": "dirichlet", "value": 350.0},
      "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
      "top": {"side": "top", "type": "neumann", "value": 0.0}
    }
  }'
```

#### 3. Dual-Mode Comparison
```bash
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "experimental_neural_corrected",
    "grid": {"nx": 64, "ny": 64, "lx": 1.0, "ly": 1.0},
    "conductivity": {"pattern_type": "uniform", "uniform_value": 2.5},
    "source": {"pattern_type": "uniform", "uniform_value": 100.0},
    "boundary_conditions": {
      "left": {"side": "left", "type": "dirichlet", "value": 300.0},
      "right": {"side": "right", "type": "dirichlet", "value": 350.0},
      "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
      "top": {"side": "top", "type": "neumann", "value": 0.0}
    }
  }'
```
