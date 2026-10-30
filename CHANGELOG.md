# Changelog

All notable changes to the **Neural Process Twin** scientific platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0-rc.1] - 2026-10-30

### Added
- **Dual-Mode Scientific Architecture (Milestone 09 & 10):**
  - Verified Numerical Mode (`numerical_fvm`) powered by conservative cell-centered Finite Volume Method (FVM), harmonic interface conductivities, and direct SuperLU factorization.
  - Experimental Neural Mode (`experimental_neural` / `experimental_neural_corrected`) powered by multi-resolution Fourier Neural Operator ($A_3$ `RefinedFNO2D`).
  - Dual-mode side-by-side comparison endpoint (`/api/compare`) computing exact ground-truth spatial error fields $\Delta T(x, y)$.
- **Multi-Resolution Refined Neural Operator ($A_3$) & Physics Smoothing ($A_4$) (Milestones 08F & 08G):**
  - Coupling of 4 spectral convolution layers with a local 3-layer convolutional residual block to mitigate high-frequency Fourier mode truncation.
  - Continuous 12-channel physical boundary encoding supporting zero-shot resolution transfer from $32\times 32$ to $256\times 256$.
  - Post-hoc $A_4$ Selective Boundary Jacobi smoother executing on the outer 2-cell boundary ring ($<15\text{ ms}$ on $256\times 256$).
- **Deployment Hardening & Concurrency Governance (Milestones 11 & 12):**
  - `ConcurrencyGovernor` with process-local bounded semaphores (max 4 FVM solves, max 8 neural passes) and 5.0-second queue timeout.
  - `ResourceSafetyMiddleware` enforcing 10 MB payload limits, correlation IDs (`X-Request-ID`), response timing headers (`X-Response-Time-Ms`), and traceback sanitization.
  - Cryptographic checkpoint trust verification using SHA-256 digests and restricted PyTorch unpickling (`weights_only=True`).
  - Asynchronous non-blocking application lifecycle pre-warming via `asyncio.to_thread`.
  - Live readiness probe at `/api/ready` and operational metrics telemetry at `/api/metrics`.
- **Packaging & Visual Showcase:**
  - Multi-stage production `Dockerfile` (Node 22 frontend build + Python 3.12-slim backend) running strictly as unprivileged user `appuser:10001`.
  - Declarative orchestration via `docker-compose.yml` with 4-CPU and 4096 MB RAM resource ceilings.
  - Comprehensive demonstration suite with 11 publication-quality research figures (300 DPI) and interactive demo scripts.
- **Support for Signed Heat Sources & Sinks (Milestone 12):**
  - Validated mathematical support for endothermic heat sinks ($q < 0$) and mixed signed sources with strict energy balance conservation.

### Fixed
- **Latency Attribution Discrepancy (Milestone 10):**
  - Discovered that neural API latency was inflated by redundant FVM sparse matrix assembly executed solely for physical telemetry.
  - Introduced optional diagnostic evaluation (`compute_diagnostics=False`), delivering an **8.76x API speedup** (499.3 ms vs 4,373.7 ms) at $256\times 256$.
- **Conflation of Linear Residual and Jacobi Indicator:**
  - Strictly separated discrete linear residual $\|b - AT\|_2$ [W] from Pointwise Jacobi Indicator [K] in schema and documentation.
- **Race Conditions in Model Singletons:**
  - Resolved concurrency hazards in lazy model initialization using process-wide double-checked locking (`_SINGLETON_LOCK`).

### Security & Governance
- **Permanent Rejection of Automated Hybrid Routing:**
  - Documented that epistemic uncertainty estimators collapse under physical distribution shift, admitting 11% false acceptances with peak errors exceeding 48 K MaxAE.
  - Refused automated surrogate promotion; established explicit user-selected dual-mode governance.
- **Resolution of $O(h^{-2})$ Error Certification Dilemma:**
  - Proved that rigorous PDE a posteriori bounds overestimate actual error by up to $19,783\times$ due to discrete maximum principle stability constant scaling.

---

## [0.8.0] - 2026-10-28
- Implemented multi-resolution Fourier Neural Operator training across $32\times 32$ and $64\times 64$ grids.
- Developed continuous distance-weighted boundary condition fields.
- Benchmarked single-pass ($U_1$) vs multi-pass MC dropout ($U_3$) uncertainty quantification.

## [0.6.0] - 2026-10-27
- Implemented initial hybrid inference engine ($R_1$) with 5-member deep ensemble.
- Added finite-sample split conformal prediction intervals.
- Integrated interactive React dashboard with HTML5 canvas rendering.

## [0.4.0] - 2026-10-25
- Developed differentiable PyTorch finite-volume loss formulation matching SciPy sparse assembly.
- Added scale-invariant dimensionless PDE residual loss and boundary flux penalties.

## [0.1.0] - 2026-10-24
- Initial repository scaffold, configuration validation, and test harness.
- Conservative cell-centered FVM reference solver with SuperLU sparse direct solve.
- Method of Manufactured Solutions (MMS) verification confirming asymptotic second-order spatial convergence ($p=2.000$).
