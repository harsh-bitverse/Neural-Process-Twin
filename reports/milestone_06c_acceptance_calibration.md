# Milestone 06C — Hybrid Acceptance Calibration and Safety Audit

**Neural Process Twin: Acceptance Policy Evaluation, Physics Gate Calibration, and Robustness Verification**  
**Date**: October 2026  
**Status**: Completed and Verified Audit Report  
**Author**: Computational Science & Scientific ML Engineering Team  
**Deliverable**: Comprehensive Acceptance Evaluation, Physics-Gate Calibration, Robustness Tests, and Empirical Benchmark  

---

## 1. Executive Summary

Milestone 06C conducts an exhaustive empirical audit of the **Hybrid Digital Twin Inference Engine** implemented in Milestone 06B. This audit evaluates whether the surrogate acceptance policy is dependable, examines the correlation between physical diagnostics and actual prediction errors, calibrates physics-based acceptance gates exclusively on an independent calibration partition, verifies fail-closed robustness under timeouts and solver exceptions, and benchmarks end-to-end execution paths.

### 1.1 Non-Negotiable Scientific Guardrails Respected

1. **Frozen Ensemble Preservation**: The five FNO-2D models trained in Milestone 04B and selected in Milestone 05 were preserved strictly untouched. Runtime SHA-256 hashes were actively validated against golden reference digests.
2. **Partition Integrity**: Physics-gate calibration and threshold sweeps were performed **exclusively** on the independent 200-sample calibration dataset (`artifacts/data/calibration_dataset.h5`). No threshold selection or tuning was performed on the IID or OOD test partitions.
3. **No Unearned Scientific Claims**: We report real measured CPU latencies and explicitly state that direct sparse numerical solves are faster than deep neural operator ensembles for single $32 \times 32$ meshes on CPU.
4. **Epistemic Humility**: We prove empirically that **low ensemble disagreement does not guarantee low prediction error**.

---

## 2. Phase 1 — Implementation Integrity Audit

A comprehensive code inspection was performed across [`contracts.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/inference/contracts.py), [`validator.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/inference/validator.py), [`screening.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/inference/screening.py), and [`engine.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/inference/engine.py).

### 2.1 Audit Findings and Defect Resolutions

| Audit Checkpoint | Initial Implementation (06B) | Defect Analysis & Scientific Risk | Resolution & Fix Implemented (06C) |
| :--- | :--- | :--- | :--- |
| **Checkpoint Integrity Verification** | Checkpoint SHA-256 digests were computed at runtime and recorded in provenance metadata, but never asserted against golden digests. | A modified, corrupted, or tampered checkpoint would be loaded silently without raising an error. | Added authoritative `FROZEN_ENSEMBLE_SHA256` dictionary. `HybridInferenceEngine.from_artifacts()` now verifies every checkpoint against golden digests, raising `ValueError` on any discrepancy. Tested in [`test_hybrid_robustness.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/tests/integration/test_hybrid_robustness.py). |
| **Timeout Execution Semantics** | Used `with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:`. | In Python, exiting a `with ThreadPoolExecutor` block calls `executor.shutdown(wait=True)`. When `future.result(timeout=...)` raises `TimeoutError`, the context manager **blocked until the worker thread completed**, completely defeating the timeout! | Replaced `with` block with explicit executor lifecycle management: on `TimeoutError`, calls `executor.shutdown(wait=False, cancel_futures=True)` and returns immediately without blocking. |
| **Worker Process Isolation** | Fallback solver executed exclusively in a worker thread. | Python threads cannot be forcefully killed or interrupted while executing C-extension linear algebra routines (`spsolve`). | Implemented optional process-isolated fallback (`fallback_isolation="process"` in `InferenceOptions`) using `multiprocessing` with explicit worker termination (`proc.terminate()`) upon timeout. |
| **Fail-Closed Guarantees** | Enforced across `INVALID_INPUT`, `FALLBACK_REQUIRED`, and `SOLVER_FAILED`. | Verified: `temperature` is strictly `None` in all failure paths; rejected surrogate predictions are never leaked. | Confirmed and covered by regression tests. |
| **Physical Units & Dimensions** | Evaluated across normalization, face-flux formulas, and diagnostics. | Verified: $k$ in $\text{W/(m K)}$, $q$ in $\text{W/m}^3$, $T$ in $\text{K}$, fluxes in $\text{W}$, face areas $A_x = \Delta y \cdot t, A_y = \Delta x \cdot t$. | Preserved general formulation across all modules. |

---

## 3. Phase 2 — Acceptance Policy Evaluation on Test Partitions

The frozen hybrid inference engine was evaluated across the 200 samples of `test_iid` and 200 samples of `test_ood` from `artifacts/data/scaled_dataset.h5`.

### 3.1 Partition Summary Statistics

| Partition | Total Samples | Accepted Samples | Fallback / Rejected | Acceptance Rate | Rejection Reasons Breakdown | Output Temperature Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`test_iid`** | 200 | 187 | 13 | **$93.5\%$** | `HIGH_ENSEMBLE_DISAGREEMENT`: 13 ($6.5\%$) | $0.67 \pm 0.47\text{ K}$ (Accepted) |
| **`test_ood`** | 200 | 0 | 200 | **$0.0\%$** | `HIGH_INPUT_CONTRAST`: 200 ($100.0\%$) | $0.00\text{ K}$ (100% Fallback Solved) |

```
Test Partition Acceptance Breakdown
================================================================================
test_iid (200 samples):
  [=======================================================================---] 93.5% Accepted (187)
  [===                                                                       ]  6.5% Fallback (13)
  
test_ood (200 samples):
  [                                                                          ]  0.0% Accepted (0)
  [==========================================================================] 100.0% Fallback (200)
================================================================================
```

### 3.2 Granular Comparison: Accepted vs. Rejected Subsets (`test_iid`)

To assess whether the rejection policy meaningfully isolates higher-error predictions, accepted and rejected sample distributions were evaluated separately:

| Metric | Accepted Subset ($N=187$) | Rejected Subset ($N=13$) | Difference / Separation Ratio |
| :--- | :---: | :---: | :---: |
| **Surrogate RMSE [K]** | **$0.67 \pm 0.47$** (p95: $1.67$) | **$1.44 \pm 0.81$** (p95: $2.55$) | **$2.15\times$ higher error in rejected** |
| **Surrogate Relative $L_2$ Error [\%]** | **$0.20 \pm 0.14\%$** (p95: $0.51\%$) | **$0.43 \pm 0.24\%$** (p95: $0.78\%$) | **$2.15\times$ higher in rejected** |
| **Surrogate MaxAE [K]** | **$2.29 \pm 1.59$** (p95: $5.64$) | **$4.87 \pm 2.65$** (p95: $8.29$) | **$2.13\times$ higher in rejected** |
| **Mean Ensemble Disagreement $s_T$ [K]** | **$0.48 \pm 0.16$** (p95: $0.78$) | **$1.15 \pm 0.12$** (p95: $1.34$) | **$2.40\times$ higher in rejected** |
| **Peak Cell Disagreement $s_T$ [K]** | **$1.81 \pm 0.65$** (p95: $2.97$) | **$3.78 \pm 0.40$** (p95: $4.27$) | **$2.09\times$ higher in rejected** |
| **Relative PDE Residual** | **$0.004 \pm 0.002$** | **$0.007 \pm 0.003$** | **$1.75\times$ higher in rejected** |
| **Absolute Flux Imbalance [W]** | **$52.1 \pm 57.0$** | **$93.9 \pm 88.5$** | **$1.80\times$ higher in rejected** |
| **End-to-End Latency [ms]** | **$55.4 \pm 12.8$** | **$66.8 \pm 9.4$** | $+11.4\text{ ms}$ (includes FVM solve) |

### 3.3 Critical Scientific Finding: Low Disagreement $\ne$ Low Error

> [!WARNING]
> **Safety Limitation Proven**: While rejected samples exhibit $2.1\times$ higher average error than accepted samples, **low ensemble disagreement does NOT guarantee low prediction error**.
> 
> In the accepted `test_iid` subset:
> - **15 out of 187 accepted samples ($8.0\%$) exhibited maximum absolute errors exceeding $5.0\text{ K}$ (up to $8.65\text{ K}$)**, despite having spatial mean disagreements below $0.85\text{ K}$ and peak cell disagreements below $3.5\text{ K}$.
> - Example — Sample 62: Mean Disagreement = $0.81\text{ K}$, Peak Disagreement = $3.37\text{ K}$ (both well below screening thresholds), yet $\mathrm{MaxAE} = 8.65\text{ K}$ and $\mathrm{RMSE} = 2.44\text{ K}$.
> 
> **Root Cause**: All five ensemble members share the same Fourier neural operator inductive bias and were trained on the same supervised objective (albeit with different random initialization seeds). When a test problem features a spatial frequency or boundary interaction that the network architecture systematically underestimates, **all ensemble members make correlated systematic bias errors**. Because the members agree with each other on the incorrect prediction, ensemble disagreement remains small ($s_T \approx 0.6 - 0.8\text{ K}$) while the true physical error is substantial ($6 - 8\text{ K}$).

---

## 4. Phase 3 — Calibration of Physics Gates

All correlation analyses and threshold sweeps were performed **strictly on the independent 200-sample calibration partition** (`artifacts/data/calibration_dataset.h5`), ensuring complete separation from test sets.

### 4.1 Correlation Analysis: Diagnostics vs. Ground-Truth Error

To evaluate which metrics are reliable indicators of surrogate error, we computed Spearman rank ($\rho$) and Pearson ($r$) correlations between all diagnostic candidates and ground-truth errors:

| Diagnostic Indicator | Spearman $\rho$ (MaxAE) | Pearson $r$ (MaxAE) | Spearman $\rho$ (RMSE) | Spearman $\rho$ (Rel $L_2$) | Diagnostic Utility Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Peak Ensemble Disagreement** ($\max s_T$) | **$+0.790$** | $+0.751$ | **$+0.806$** | **$+0.808$** | **Strongest single indicator** of error |
| **Mean Ensemble Disagreement** ($\bar{s}_T$) | **$+0.762$** | $+0.718$ | **$+0.792$** | **$+0.791$** | **Very strong indicator** of global error |
| **Discrete PDE Residual ($L_2$ Norm [W])** | $+0.454$ | $+0.412$ | $+0.458$ | $+0.463$ | **Moderate indicator**; correlates with local slope error |
| **Relative PDE Residual** | $+0.399$ | $+0.354$ | $+0.423$ | $+0.433$ | **Moderate indicator**; scale-invariant |
| **Absolute Flux Imbalance [W]** | $+0.185$ | $+0.124$ | $+0.176$ | $+0.178$ | **Weak indicator**; highly noisy on boundary stencil |
| **Relative Flux Imbalance** | **$+0.014$** | $+0.009$ | $+0.022$ | $+0.030$ | **Zero correlation**; completely uninformative |
| **Conductivity Contrast Ratio** | $+0.153$ | $+0.138$ | $+0.119$ | $+0.125$ | Weak within IID (crucial for OOD separation) |

```
Spearman Rank Correlation with True Temperature MaxAE (K)
================================================================================
Peak Disagreement:   [===================================================] +0.790
Mean Disagreement:   [===============================================]     +0.762
PDE Residual (L2):   [============================]                         +0.454
Relative Residual:   [========================]                             +0.399
Flux Imbalance (Abs):[============]                                         +0.185
Contrast Ratio:      [=========]                                            +0.153
Flux Imbalance (Rel):[]                                                     +0.014
================================================================================
```

### 4.2 Engineering Error Tolerances and Baseline Screener Performance

We defined four candidate engineering error tolerances on the calibration partition:
- **$\tau_{\mathrm{err}} = 10.0\text{ K}$**: Coarse industrial tolerance.
- **$\tau_{\mathrm{err}} = 5.0\text{ K}$**: Moderate process-monitoring tolerance.
- **$\tau_{\mathrm{err}} = 2.0\text{ K}$**: Precision digital twin tolerance.
- **$\tau_{\mathrm{err}} = 1.0\text{ K}$**: High-precision boundary tolerance.

| Tolerance $\tau_{\mathrm{err}}$ | Total True Violations in Calibration | Baseline Acceptance Rate | False Acceptances (Violations Missed) | False Acceptance Rate | Precision | Recall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10.0\text{ K}$** | $1$ ($0.5\%$) | $92.5\%$ | **$0$** | **$0.0\%$** | $20.0\%$ | **$100.0\%$** |
| **$5.0\text{ K}$** | $28$ ($14.0\%$) | $92.5\%$ | **$19$** | **$9.5\%$** | $60.0\%$ | $32.1\%$ |
| **$2.0\text{ K}$** | $102$ ($51.0\%$) | $92.5\%$ | **$88$** | **$44.0\%$** | $93.3\%$ | $13.7\%$ |
| **$1.0\text{ K}$** | $175$ ($87.5\%$) | $92.5\%$ | **$160$** | **$80.0\%$** | $100.0\%$ | $8.6\%$ |

### 4.3 Evaluation of Physics-Gated Policies

We swept candidate thresholds for relative PDE residuals and boundary flux imbalances on the calibration partition to evaluate whether adding physics gates reduces false acceptances for the $5.0\text{ K}$ tolerance:

#### 1. Relative PDE Residual Gates
- Observed IID relative PDE residuals in calibration are tightly distributed between $0.000$ and $0.012$ ($< 1.2\%$).
- Thresholds above $0.02$ ($2\%$) have **zero filtering effect** (acceptance remains $92.5\%$, false acceptances remain 19).
- Setting a strict threshold at $0.005$ reduces false acceptances from 19 to 12, but drops acceptance rate to $62.0\%$, forcing $38\%$ unnecessary solver fallbacks.

#### 2. Boundary Flux Imbalance Gates
- In Milestone 06B, we demonstrated that the half-cell stencil factor $\frac{2}{\Delta x} = 64\text{ m}^{-1}$ scales even sub-Kelvin boundary errors into tens of Watts of flux imbalance.
- The median surrogate flux imbalance in calibration is $38.5\text{ W}$.
- Sweeping strict flux thresholds:
  - Threshold $\le 10.0\text{ W}$: Acceptance rate collapses to **$8.0\%$** (92% fallbacks!), false acceptances drops from 19 to 1, but precision is only $14.7\%$.
  - Threshold $\le 20.0\text{ W}$: Acceptance rate is **$18.5\%$** ($81.5\%$ fallbacks).
  - Threshold $\le 50.0\text{ W}$: Acceptance rate is **$50.0\%$** ($50\%$ fallbacks).

### 4.4 Calibration Policy Conclusions

1. **Boundary Flux Imbalance is NOT Suitable as a Primary Filter**: Because global flux imbalance has virtually zero correlation with interior temperature error ($\rho = 0.014$ relative, $\rho = 0.185$ absolute) and exhibits extreme stencil sensitivity, enforcing a tight flux gate induces massive false rejections of high-quality surrogate predictions without meaningful safety gains.
2. **PDE Residual as Defense-in-Depth**: Relative PDE residual exhibits moderate correlation ($\rho = 0.40$). A conservative relative residual threshold of $\tau_{\mathrm{res}} \le 0.05$ safely catches severe divergence without penalizing nominal in-distribution predictions.
3. **Calibrated Threshold Specification**:
   - `provisional_mean_disagreement_threshold_k`: $0.9978\text{ K}$
   - `provisional_max_disagreement_threshold_k`: $4.4526\text{ K}$
   - `input_contrast_threshold`: $7.16$
   - `relative_pde_residual_threshold`: $0.05$ (defense-in-depth against algebraic inconsistency)
   - `boundary_flux_imbalance_threshold_w`: Disabled ($None$) by default to prevent severe false rejection.

---

## 5. Phase 4 — Timeout and Failure Robustness

Robustness tests were authored in [`tests/integration/test_hybrid_robustness.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/tests/integration/test_hybrid_robustness.py) to stress-test failure modes:

### 5.1 Verification Scenarios and Results

1. **Checkpoint Tampering Detection**:
   Created a dummy checkpoint with modified byte contents under a recognized directory name (`scaled_fno_supervised_seed42`). Initializing `HybridInferenceEngine.from_artifacts()` immediately raised:
   `ValueError: Checkpoint integrity violation for scaled_fno_supervised_seed42: expected SHA-256 37b0b1a4..., observed ...`.
2. **Non-Blocking Thread Timeout**:
   Configured solver timeout at $0.1\text{ s}$ with a mock solver sleeping for $2.0\text{ s}$. The call returned in $0.15\text{ s}$, returned `InferenceStatus.SOLVER_FAILED`, set `temperature = None`, and logged `SOLVER_TIMEOUT`. Calling thread execution did **not** hang.
3. **Repeated Timeout Resilience**:
   Triggered three consecutive solver timeouts. The engine absorbed all three failures with `temperature = None`, and the subsequent normal request executed cleanly and returned an accepted solution without state leakage.
4. **Solver Divergence & Non-Finite Handling**:
   Simulated a solver returning `NaN` temperatures and failing algebraic convergence. The engine caught the condition, set `status = SOLVER_FAILED`, `rejection = SOLVER_DIVERGENCE`, and strictly returned `temperature = None`.

---

## 6. Phase 5 — End-to-End Evaluation and Benchmarks

Benchmarking was conducted across identical 50-sample batches for four inference modes on both `test_iid` and `test_ood` partitions (Host CPU, Python 3.12, PyTorch 2.14.1+cpu):

### 6.1 Benchmark Results Table

| Inference Mode | Workload Partition | Mean Latency [ms] | Std Dev [ms] | Median [ms] | 95th Pct [ms] | Throughput [samp/s] | Acceptance Rate | Solver Fallback Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Direct FVM Solver** | `test_iid` | **$5.56$** | $\pm 1.19$ | $4.80$ | $7.28$ | **$179.8$** | — | — |
| **Direct FVM Solver** | `test_ood` | **$5.58$** | $\pm 1.16$ | $4.90$ | $7.07$ | **$179.1$** | — | — |
| **Surrogate-Only** | `test_iid` | **$62.27$** | $\pm 8.00$ | $63.60$ | $70.65$ | **$16.1$** | $100\%$ (unscreened) | $0\%$ |
| **Surrogate-Only** | `test_ood` | **$58.68$** | $\pm 9.71$ | $60.99$ | $73.22$ | **$17.0$** | $100\%$ (unscreened) | $0\%$ |
| **Hybrid Fail-Closed** | `test_iid` | **$64.47$** | $\pm 7.80$ | $65.53$ | $76.79$ | **$15.5$** | **$94.0\%$** | $0\%$ (aborts on reject) |
| **Hybrid Fail-Closed** | `test_ood` | **$63.80$** | $\pm 8.53$ | $64.99$ | $76.31$ | **$15.7$** | **$0.0\%$** | $0\%$ (aborts on reject) |
| **Hybrid with Fallback**| `test_iid` | **$65.16$** | $\pm 8.39$ | $65.79$ | $77.83$ | **$15.3$** | **$94.0\%$** | **$6.0\%$** |
| **Hybrid with Fallback**| `test_ood` | **$75.46$** | $\pm 8.11$ | $75.29$ | $86.50$ | **$13.3$** | **$0.0\%$** | **$100.0\%$** |

```
CPU Wall-Clock Latency Comparison (32x32 Grid, ms)
================================================================================
Direct FVM Solver (IID):       [==] 5.56 ms
Direct FVM Solver (OOD):       [==] 5.58 ms
Surrogate-Only (5x FNO IID):   [====================================] 62.27 ms
Hybrid Fail-Closed (IID):      [======================================] 64.47 ms
Hybrid with Fallback (IID):    [=======================================] 65.16 ms
Hybrid with Fallback (OOD):    [=============================================] 75.46 ms
================================================================================
```

### 6.2 Key Latency Takeaways

1. **Direct FVM Dominance on Single CPU**: Direct sparse linear solve via SuperLU ($N = 1024$) takes $\approx 5.5\text{ ms}$, operating at $180\text{ samples/s}$. The 5-member deep ensemble requires $\approx 62\text{ ms}$ on CPU ($16\text{ samples/s}$).
2. **Hybrid Overhead**: Running input validation, ensemble forward pass, diagnostics, and screening adds only $\approx 2.2\text{ ms}$ of overhead above the raw ensemble inference ($64.5\text{ ms}$ vs $62.3\text{ ms}$).
3. **Fallback Cost on OOD**: When an input is rejected (100% of OOD samples), executing the surrogate pipeline and then dispatching to FVM fallback requires $75.5\text{ ms}$ ($13.3\text{ samples/s}$). This represents a modest $10\text{ ms}$ overhead above the rejected surrogate execution, but **delivers $0.00\text{ K}$ ground-truth physical accuracy** instead of exposing $80 - 130\text{ K}$ surrogate errors.

---

## 7. Deliverables and Artifact Inventory

All required deliverables have been produced, verified, and placed in repository locations:

1. **Audit & Calibration Report**:
   - [`reports/milestone_06c_acceptance_calibration.md`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/reports/milestone_06c_acceptance_calibration.md)
2. **Machine-Readable Calibration Results**:
   - [`artifacts/reports/milestone_06c_acceptance_results.json`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/artifacts/reports/milestone_06c_acceptance_results.json) ($662.8\text{ KB}$, containing complete per-sample records, correlations, tolerance tables, and benchmark profiles).
3. **Evaluation & Calibration Script**:
   - [`scripts/evaluate_milestone_06c_acceptance.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/scripts/evaluate_milestone_06c_acceptance.py)
4. **Robustness & Concurrency Regression Tests**:
   - [`tests/integration/test_hybrid_robustness.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/tests/integration/test_hybrid_robustness.py)
5. **Updated Reports Index**:
   - [`reports/README.md`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/reports/README.md)

---

## 8. Summary of Empirically Supported Claims vs. Heuristics

To maintain scientific integrity, we explicitly demarcate what has been proven from what remains an experimental heuristic:

| Claim / Mechanism | Status | Supporting Empirical Evidence |
| :--- | :---: | :--- |
| **Fail-closed guarantees hold under all failure modes** | **Empirically Proven** | Verified in unit, integration, and robustness tests: `result.temperature is None` whenever validation aborts, solver times out, solver diverges, or solver raises an exception. |
| **Frozen checkpoint provenance is immutable** | **Empirically Proven** | Verified: SHA-256 digests of all 5 FNO checkpoints match Milestone 04B/05B records; anti-tamper assertions reject corrupted weights on startup. |
| **Domain contrast gate reliably catches extreme OOD** | **Empirically Proven** | Verified: 100% (200/200) of high-contrast OOD samples were rejected and routed to fallback reference solves. |
| **Low ensemble disagreement implies low prediction error** | **REFUTED** | Refuted by data: $8.0\%$ (15/187) of accepted IID samples had $\mathrm{MaxAE} > 5.0\text{ K}$ (up to $8.65\text{ K}$) despite having low ensemble disagreement ($s_T < 0.8\text{ K}$). Shared inductive bias produces correlated model errors. |
| **Boundary flux imbalance can serve as a precision error gate** | **REFUTED** | Refuted by data: Boundary flux imbalance has virtually zero correlation with actual temperature error ($\rho = 0.014$ relative, $\rho = 0.185$ absolute) and causes $92\%$ false rejection rates if tightly thresholded. |
| **Surrogate accelerates single-problem CPU inference** | **REFUTED** | Refuted by data: On single $32 \times 32$ meshes on CPU, direct sparse numerical solve is $11 - 13\times$ faster than the 5-member deep ensemble. |

---

## 9. Conclusion

Milestone 06C demonstrates that the hybrid inference engine is operationally robust, strictly fail-closed, and highly effective at routing out-of-distribution inputs to conservative reference solves. However, the audit also establishes the critical scientific reality that ensemble disagreement alone cannot guarantee sub-$5\text{ K}$ precision against correlated model bias, and boundary flux diagnostics cannot serve as a reliable substitute for epistemic uncertainty.

In accordance with project instructions, Milestone 06C is complete, and work stops here before Milestone 07.
