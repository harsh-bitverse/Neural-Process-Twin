# Neural Process Twin — Known Technical Limitations and Governance Boundaries

**Document Version:** 1.0.0-rc.1  
**Status:** Authoritative Governance Document  
**Date:** October 30, 2026

---

## 1. Safety and Certification Boundaries

### 1.1 Uncertified Status of Neural Surrogate Predictions
- **Limitation:** Predictions generated via Experimental Neural Mode (`experimental_neural` or `experimental_neural_corrected`) are **uncertified mathematical approximations**.
- **Governance Requirement:** Under zero circumstances may neural outputs be substituted for verified numerical solves in safety-critical thermal certification (e.g. nuclear, aerospace, medical, or fire-safety regulatory filings).
- **Mandatory Enforcement:** All API endpoints unconditionally emit `is_certified: false`, a non-null `certification_note`, and descriptive warnings whenever neural inference is executed.

### 1.2 Epistemic Insecurity under Physical Distribution Shift
- **Limitation:** Deep neural network confidence scores (including deep ensembles, heteroscedastic aleatoric heads, and spatial Monte Carlo dropout) fail to detect certain out-of-distribution physical anomalies.
- **Observed Failure:** Under high conductivity contrasts ($k_{\max}/k_{\min} \ge 15$), neural surrogates report low uncertainty ($\sigma \le 3.0\text{ K}$) despite actual local errors exceeding $48.0\text{ K}$ MaxAE.
- **Consequence:** Automated hybrid gating based solely on empirical uncertainty thresholds cannot guarantee safety; automated surrogate promotion is permanently prohibited in this codebase.

---

## 2. Mathematical Error-Certification Limitations

### 2.1 The $O(h^{-2})$ Discrete Stability Dilemma
- **Limitation:** A posteriori $L_\infty$ error bounds derived from the discrete maximum principle ($\|e\|_\infty \le \|\mathbf{A}^{-1}\|_\infty \|r\|_\infty$) cannot provide practical single-pass error certificates on fine meshes.
- **Scaling Dilemma:** Because the elliptic stability constant $\|\mathbf{A}^{-1}\|_\infty$ scales as $O(h^{-2}) = O(N)$, the rigorous bound overestimates true local errors by $300\times$ at $32\times 32$ and by $19,783\times$ at $256\times 256$.
- **Majorant Cost:** Exact barrier majorant solves ($\mathbf{A} w = |r|$) eliminate overestimation but cost $93.4\%$ of the direct FVM solve time, defeating the throughput advantage of neural inference.

---

## 3. Discretization and Resolution Constraints

### 3.1 Neural Surrogate Resolution Boundaries
- **Limitation:** The experimental surrogate $A_3$ `RefinedFNO2D` is verified strictly on four canonical square resolutions: $32\times 32, 64\times 64, 128\times 128, 256\times 256$.
- **Behavior:** Submitting non-canonical resolutions (e.g. $48\times 48$, $96\times 96$, or non-square aspect ratios) to neural mode fails closed with HTTP 400 (`UNSUPPORTED_RESOLUTION`).
- **Numerical Reference Flexibility:** The Verified Numerical FVM solver accepts arbitrary resolutions and arbitrary rectangular aspect ratios ($L_x \ne L_y, N_x \ne N_y$) up to $N_x N_y \le 65,536$.

### 3.2 Maximum Grid Dimensions
- **Limitation:** To prevent memory amplification attacks and CPU core exhaustion, all endpoints enforce $N_x, N_y \le 256$ and total control volume count $N_x N_y \le 65,536$.

---

## 4. Concurrency and Multi-Worker Deployment Boundaries

### 4.1 Process-Local Concurrency Governance
- **Limitation:** `ConcurrencyGovernor` relies on in-memory Python `threading.BoundedSemaphore` primitives. Semaphores and active counters are **strictly process-local**.
- **Multi-Worker Implication:** When deploying Uvicorn or Gunicorn with multiple worker processes (e.g. `--workers 4`), each worker process instantiates its own independent governor. Total concurrent numerical solves will scale to $W \times 4$.
- **Recommended Topology:** Production deployments should run a single worker process per container ($W = 1$) and scale horizontally via container orchestration replicas (Kubernetes Pods / AWS ECS tasks) with pinned CPU and memory quotas.

---

## 5. Summary of Operating Scope

| Capability | Verified Numerical Mode | Experimental Neural Mode |
|------------|-------------------------|--------------------------|
| **Mathematical Formulation** | Cell-Centered Finite Volume Method | Fourier Neural Operator + Local Residual Block |
| **Discrete Residual Norm** | Machine Precision ($\le 3.54 \times 10^{-14}$) | Approximate ($10^{-2} - 10^{-1}$) |
| **Physical Energy Conservation** | Guaranteed ($|\Delta Q| / Q \le 10^{-12}$) | Approximate (Flux imbalance $90\%-98\%$) |
| **Certified Status** | **`is_certified: true`** | **`is_certified: false`** |
| **Supported Grid Resolutions** | Arbitrary ($N_x, N_y \le 256$) | Canonical Only ($\{32, 64, 128, 256\}$) |
| **End-to-End Latency at $256\times 256$** | ~4,370 ms | ~499 ms (**8.76x speedup**) |
| **Intended Engineering Use** | Certified Design Sign-Off & Verification | Rapid Interactive Exploration & Parameter Sweeps |
