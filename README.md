# Neural Process Twin

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![Tests Passing](https://img.shields.io/badge/tests-318%20passed-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/docker-multi--stage-blue.svg)](Dockerfile)
[![Release](https://img.shields.io/badge/release-v1.0.0--rc.1-orange.svg)](RELEASE_NOTES.md)

> **A Dual-Mode Scientific Digital Twin Platform for Physics-Consistent Thermal Simulation**  
> *Coupling Conservative Finite-Volume Solvers with Multi-Resolution Fourier Neural Operators*

---

## 1. Executive Overview

**Neural Process Twin** is an open-source scientific computing and machine-learning research platform designed to model two-dimensional steady-state heat conduction governed by the variable-conductivity elliptic PDE:

$$-\nabla \cdot \big(k(x, y) \nabla T(x, y)\big) = q(x, y)$$

In safety-critical physical systems (e.g. electronics packaging, power converters, battery cooling), uncertified deep-learning predictions pose serious operational risks. Neural Process Twin addresses this fundamental challenge by enforcing an **explicit dual-mode operating paradigm**:

1. **Verified Numerical Mode (`numerical_fvm`):**  
   Conservative cell-centered Finite Volume Method (FVM) solver with harmonic interface conductivities and direct sparse LU factorization (SuperLU via SciPy). Serves as the authoritative, certified ground truth with machine-precision relative algebraic residuals ($\|r\|_2 / \|b\|_2 \le 3.54 \times 10^{-14}$) and theoretical second-order convergence ($p = 2.000$) verified under the Method of Manufactured Solutions (MMS).
2. **Experimental Neural Mode (`experimental_neural` / `experimental_neural_corrected`):**  
   Multi-resolution Fourier Neural Operator ($A_3$ `RefinedFNO2D`) coupled with a local 3-layer convolutional residual block and an optional post-hoc $A_4$ Selective Boundary Jacobi physics smoother. Delivers an **8.76x end-to-end API speedup** (499.3 ms vs 4,373.7 ms) at $256\times 256$ resolution, accompanied unconditionally by transparent diagnostic disclaimers.
3. **Dual-Mode Comparison API (`/api/compare`):**  
   Executes both models concurrently and evaluates the spatial error field $\Delta T(x, y) = |T_{\text{neural}} - T_{\text{fvm}}|$, giving engineers instant visual error attribution.

---

## 2. Platform Architecture

```mermaid
flowchart TD
    subgraph Client Applications
        UI["React 19 Dashboard<br/>(TypeScript, HTML5 Canvas Rendering)"]
        Client["Scientific Python / REST API Client"]
    end

    subgraph Security & Governance Boundary
        Middleware["ResourceSafetyMiddleware<br/>(10 MB Payload Ceiling, UUID Tracing)"]
        Governor["ConcurrencyGovernor<br/>(Max 4 FVM Solves, Max 8 Neural Inferences)"]
    end

    subgraph Dual-Mode Service Engines
        FVM["Verified Numerical Mode<br/>(HeatConductionSolver2D, SuperLU)"]
        Neural["Experimental Neural Mode<br/>(RefinedFNO2D A3 / Selective Jacobi A4)"]
        Compare["Dual-Mode Comparator<br/>(Side-by-Side Spatial Difference)"]
    end

    subgraph Model Storage
        Checkpoints["Cryptographic Trust Boundary<br/>(SHA-256 Verified, weights_only=True)"]
    end

    Client Applications --> Middleware
    Middleware --> Governor
    Governor --> FVM
    Governor --> Neural
    Governor --> Compare
    Neural -.-> Checkpoints
```

---

## 3. Key Scientific Findings & Negative Results

Unlike research projects that report only favorable accuracy numbers, Neural Process Twin comprehensively documents critical negative findings across 12 milestones:

1. **Epistemic Uncertainty Collapse under Physical Shift:**  
   Standard epistemic uncertainty metrics (MC dropout, deep ensembles) collapse under physical distribution shifts (high conductivity contrasts $k_{\max}/k_{\min} \ge 15$). Networks report low uncertainty ($\sigma \le 3.0\text{ K}$) despite local temperature errors exceeding $48\text{ K}$ MaxAE.
2. **Failure of Automated Hybrid Routing:**  
   Auditing surrogate-accepted queries against standard engineering tolerances ($\text{MaxAE} \le 10\text{ K}$) revealed that **76% to 82% of accepted IID predictions violated the 10 K ceiling**. Automated hybrid promotion was permanently rejected in favor of explicit user mode selection.
3. **The $O(h^{-2})$ Error Certification Dilemma:**  
   Rigorous mathematical a posteriori $L_\infty$ error bounds derived from the discrete maximum principle blow up as $\|\mathbf{A}^{-1}\|_\infty = O(h^{-2}) = O(N)$. On a $256\times 256$ mesh, the theoretical bound overestimates actual error by **$19,783\times$** ($465,000\text{ K}$ vs $23.5\text{ K}$), rejecting 100% of queries. Tight barrier majorant solves cost 93.4% of the direct FVM solve time.
4. **Resolution Invariance vs Boundary Rings:**  
   Fourier spectral truncation ($k_{\max}=12$) concentrates 92% of peak errors within 3 boundary cells. Coupling a local 3-layer convolutional residual block with a 2-cell Selective Boundary Jacobi smoother systematically reduces boundary ring MaxAE while leaving interior cells untouched.

---

## 4. Frozen Multi-Resolution Benchmark Summary

Evaluated on the frozen independent benchmark suite across all canonical resolutions:

| Resolution | Grid Cells | FVM Solve [ms] | Neural Forward [ms] | End-to-End API Speedup | $A_3$ Mean RMSE [K] | $A_4$ Mean RMSE [K] | MaxAE $\le 10\text{ K}$ Compliance |
|------------|------------|----------------|---------------------|------------------------|---------------------|---------------------|------------------------------------|
| $32 \times 32$ | 1,024 | 52.1 ms | 63.2 ms | **2.46x** | 6.24 K | 6.13 K | 20.0% |
| $64 \times 64$ | 4,096 | 260.4 ms | 64.5 ms | **10.66x** | 6.35 K | 6.28 K | 16.0% |
| $128 \times 128$ | 16,384 | 845.2 ms | 66.8 ms | **20.69x** | 6.42 K | 6.38 K | 16.0% |
| $256 \times 256$ | 65,536 | 3,810.0 ms | 78.5 ms | **8.76x** | 6.49 K | 6.48 K | 16.0% |

*(Note: End-to-end API speedup compares `/api/simulate` in direct FVM mode vs experimental neural mode with `compute_diagnostics=False`.)*

---

## 5. Quick-Start Guide

### Prerequisites
- Python 3.12+
- Node.js 20+ and npm (for React dashboard)
- [`uv`](https://github.com/astral-sh/uv) (recommended Python package manager)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/harsh-bitverse/Neural-Process-Twin.git
cd "Neural Process Twin"

# Create virtual environment and install dependencies
uv sync

# Install frontend dependencies and build dashboard
cd frontend
npm ci
npm run build
cd ..
```

### 2. Verify Checkpoint Provenance
```bash
uv run python scripts/verify_deployment_artifacts.py
```
*Expected: 6/6 checkpoints verified via immutable SHA-256 digests.*

### 3. Launch Backend Server
```bash
uv run uvicorn neural_process_twin.api.app:create_app --factory --host 0.0.0.0 --port 8000
```
- Interactive API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Platform Readiness Probe: [http://localhost:8000/api/ready](http://localhost:8000/api/ready)
- Operational Metrics: [http://localhost:8000/api/metrics](http://localhost:8000/api/metrics)

---

## 6. Docker Container Deployment

A hardened multi-stage container packaging Node 22 (frontend builder) with Python 3.12-slim executes strictly under an unprivileged user (`appuser:10001`):

```bash
# Build and run with Docker Compose (4 CPUs, 4096 MB RAM)
docker compose up -d

# Verify container health
curl -f http://localhost:8000/api/ready
```

---

## 7. Python API Client Example

```python
import httpx

payload = {
    "mode": "numerical_fvm",  # Or 'experimental_neural', 'experimental_neural_corrected'
    "grid": {"nx": 64, "ny": 64, "lx": 1.0, "ly": 1.0},
    "conductivity": {
        "pattern_type": "inclusion",
        "inclusion_k_background": 1.0,
        "inclusion_k_feature": 5.0,
        "inclusion_center_x": 0.5,
        "inclusion_center_y": 0.5,
        "inclusion_radius": 0.2,
    },
    "source": {
        "pattern_type": "gaussian_spot",
        "spot_peak_value": 250.0,
        "spot_center_x": 0.5,
        "spot_center_y": 0.5,
        "spot_sigma": 0.15,
    },
    "boundary_conditions": {
        "left": {"side": "left", "type": "dirichlet", "value": 300.0},
        "right": {"side": "right", "type": "dirichlet", "value": 350.0},
        "bottom": {"side": "bottom", "type": "neumann", "value": 0.0},
        "top": {"side": "top", "type": "neumann", "value": 0.0},
    },
}

response = httpx.post("http://localhost:8000/api/simulate", json=payload, timeout=30.0)
data = response.json()

print(f"Status: {data['status']}")
print(f"Is Certified: {data['is_verified_numerical']}")
print(f"Peak Temperature: {data['temperature_max_k']:.2f} K")
print(f"Energy Imbalance: {data['diagnostics']['relative_flux_imbalance']:.2e}")
```

---

## 8. Reproducibility & Research Artifacts

Every scientific result and figure in this repository is 100% reproducible:

```bash
# Run complete test suite (318 passed tests)
uv run pytest -q

# Run frozen multi-resolution benchmark suite
uv run python scripts/benchmark_milestone_12_frozen_suite.py

# Generate all 11 publication-quality research figures (300 DPI)
uv run python scripts/generate_milestone_12_publication_figures.py
```

Generated research figures are archived in [`artifacts/figures/`](artifacts/figures/):
- `fig01_fvm_mms_convergence.png`: FVM Method of Manufactured Solutions convergence orders ($p=2.000$).
- `fig02_temperature_fields_comparison.png`: Side-by-side temperature comparison and difference field.
- `fig03_spatial_error_maps.png`: Spatial error distributions for $A_3$ vs $A_4$ Selective Jacobi.
- `fig04_multiresolution_rmse_scaling.png`: Zero-shot resolution invariance ($32\times 32 \to 256\times 256$).
- `fig05_maxae_distributions.png`: Localized error distributions vs 5 K and 10 K engineering tolerances.
- `fig06_iid_vs_ood_generalization.png`: Performance breakdown across physical shift regimes.
- `fig07_boundary_correction_effects.png`: Boundary ring error reduction mechanics.
- `fig08_latency_breakdown_speedups.png`: Latency decomposition and 8.76x API speedup.
- `fig09_accuracy_vs_latency_tradeoffs.png`: Accuracy-latency Pareto frontier.
- `fig10_empirical_uncertainty_gating_failure.png`: OOD confidence collapse and false acceptance quadrant.
- `fig11_rigorous_error_bounds_dilemma.png`: The $O(h^{-2})$ stability constant explosion.

---

## 9. Comprehensive Documentation

- **Final Technical Research Report:** [`reports/NEURAL_PROCESS_TWIN_FINAL_RESEARCH_REPORT.md`](reports/NEURAL_PROCESS_TWIN_FINAL_RESEARCH_REPORT.md)
- **Academic Research Paper Manuscript:** [`papers/neural_process_twin_research_paper.md`](papers/neural_process_twin_research_paper.md)
- **Milestone Audit Archive:** [`reports/README.md`](reports/README.md)
- **Release Notes & Changelog:** [`RELEASE_NOTES.md`](RELEASE_NOTES.md) | [`CHANGELOG.md`](CHANGELOG.md)
- **Known Limitations & Governance:** [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)

---

## 10. Repository Structure

```text
Neural Process Twin/
├── artifacts/
│   ├── checkpoints/           # Authoritative frozen model checkpoints
│   ├── figures/               # 11 publication research figures (300 DPI)
│   └── reports/               # Machine-readable evaluation benchmarks (JSON)
├── frontend/                  # React 19 + TypeScript dashboard
├── papers/                    # Formal research paper manuscript
├── reports/                   # Detailed milestone reports (M00-M12)
├── scripts/                   # Benchmarks, evaluations, and figure generators
├── src/neural_process_twin/
│   ├── api/                   # FastAPI backend, middleware, and governance
│   ├── data/                  # Continuous boundary encoding and datasets
│   ├── inference/             # Surrogate services, validator, contracts
│   ├── models/                # RefinedFNO2D, CNN baselines, factory
│   ├── solvers/               # Conservative FVM solver, assembly, Jacobi
│   ├── training/              # Physics-aware loss functions, checkpointing
│   └── uncertainty/           # Ensembles, conformal prediction, OOD metrics
└── tests/                     # 318 unit, scientific, and integration tests
```

---

## 11. Governance and License

Neural Process Twin is released as an open-source scientific research artifact under the [MIT License](LICENSE).
*(Note: As an academic research release, experimental neural outputs carry no warranty or safety certification. Verified numerical FVM mode must always be consulted for critical engineering applications.)*
