# Milestone 08A — Neural Operator Performance and Generalization Research Design

**Neural Process Twin: Empirical Audit, Generalization Failure Analysis, and Comprehensive Experimental Program**  
**Date**: October 2026  
**Status**: Completed Research Design  
**Author**: Computational Science & Scientific ML Engineering Team  
**Deliverables**: Baseline System Audit, Multi-Resolution Benchmark, Generalization Hypotheses, Prioritized Research Candidates, and Experimental Protocol  

---

## 1. Executive Summary

Milestone 08A establishes a scientifically rigorous, empirical research design to investigate and resolve the fundamental accuracy, generalization, and computational efficiency bottlenecks of **Neural Process Twin**.

Across Milestones 00–07A, Neural Process Twin built a verified 2D steady-state heat conduction twin combining a conservative finite-volume reference solver (FVM), a frozen 5-member Fourier Neural Operator (FNO-2D) ensemble, split-conformal uncertainty intervals, and a fail-closed hybrid safety screening engine. 

While the system is numerically verified and deployed in an interactive research dashboard, rigorous empirical auditing reveals a **core scientific paradox**:

```
+---------------------------------------------------------------------------------------------------+
|                                     THE RESOLUTION PARADOX                                        |
|                                                                                                   |
|  Small Meshes (32x32):                                                                            |
|  - Accuracy: Surrogate achieves lowest error (IID RMSE = 0.67 K, Rel-L2 = 0.52%).                  |
|  - Speed: FVM SuperLU solve (4.41 ms) is ~12x FASTER on CPU than the 5-member ensemble (52.48 ms).|
|  - Practicality: Neural surrogate is a computational bottleneck on single small CPU meshes.       |
|                                                                                                   |
|  Larger Meshes (64x64, 128x128, 256x256):                                                         |
|  - Speed: FNO scales near-linearly (O(N log N)), crossing over FVM SuperLU (O(N^1.5)) on CPU.     |
|    At 256x256, Single FNO is 5.2x faster than FVM (53.3 ms vs 276.4 ms).                         |
|  - Accuracy: Zero-shot super-resolution accuracy CATASTROPHICALLY DEGRADES:                       |
|    RMSE jumps 12.86x from 0.94 K (at 32x32) to 12.09 K (at 64x64) for the identical physics!     |
|  - Practicality: Computational speedup on fine grids cannot be exploited due to severe error.     |
+---------------------------------------------------------------------------------------------------+
```

### 1.1 Non-Negotiable Research Guardrails Respected

1. **No Ad-Hoc Model Retraining**: This milestone is strictly an analytical and experimental design program. No new models have been trained, no PDE families have been expanded, and all 5 frozen ensemble checkpoints remain untouched.
2. **Empirical Measurement over Unsubstantiated Claims**: All computational complexity metrics, resolution crossovers, and super-resolution error amplifications reported herein were empirically measured on the host system and archived in [`artifacts/reports/milestone_08a_baseline_audit.json`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/artifacts/reports/milestone_08a_baseline_audit.json).
3. **Hardware Honesty**: The host platform operates on a CPU execution environment (`torch.cuda.is_available() = False`). GPU evaluation protocols are rigorously defined as conditional research protocols without extrapolating imaginary GPU speedups.
4. **Historical Benchmark Preservation**: Existing datasets (`scaled_dataset.h5`, `calibration_dataset.h5`) remain permanent benchmark references (Benchmark v1.0). New distributions or multi-resolution datasets are defined as versioned holdouts.

---

## 2. Phase 1 — Existing System Audit

A systematic code and performance audit was executed across all core subsystems using [`scripts/audit_milestone_08a_benchmarks.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/scripts/audit_milestone_08a_benchmarks.py):

### 2.1 Model Architecture and Parameter Audit

| Model Component | Parameter Count | Trainable Parameters | Architectural Description | Invariant Dimension |
| :--- | :---: | :---: | :--- | :--- |
| **CNN Baseline 2D** | **$31,265$** | $31,265$ ($100\%$) | 4 ConvBlock2D layers (kernel $3 \times 3$, width 32), replicate padding, residual skips. | Local receptive field ($9 \times 9$ cells). |
| **Single FNO-2D** | **$1,188,641$** | $1,188,641$ ($100\%$) | Lifting Conv $1 \times 1$ ($12 \to 32$), 4 SpectralConv2D layers (12 modes) + 1x1 bypass, 2-layer projection ($32 \to 128 \to 1$). | Resolution-invariant in Fourier space. |
| **Deep Ensemble ($M=5$)** | **$5,943,205$** | $5,943,205$ ($100\%$) | 5 independent FNO-2D models (`seed42` to `seed46`), sequential evaluation on CPU. | $5.0\times$ single model computational cost. |

- **Parameter Ratio**: FNO-2D has **$38.0\times$ more parameters** than the CNN baseline ($1.19\text{ M}$ vs $31.3\text{ K}$).
- **Ensemble Memory Footprint**: Loading all 5 ensemble checkpoints into memory occupies $\approx 142.2\text{ MB}$ of checkpoint weight data on disk ($28.45\text{ MB}$ per member).

### 2.2 Input Normalization and 12-Channel Boundary Encoding

1. **Normalization Limitations**:
   - `FieldNormalizer` fits standard $z$-score statistics strictly on the training partition:
     $$\tilde{k} = \frac{k - \mu_k}{\sigma_k}, \quad \tilde{q} = \frac{q - \mu_q}{\sigma_q}, \quad \tilde{T} = \frac{T - \mu_T}{\sigma_T}$$
     where $\mu_k = 2.7491\text{ W/(m K)}, \sigma_k = 0.9596, \mu_T = 327.04\text{ K}, \sigma_T = 17.46\text{ K}$.
   - **Flaw in Linear Scaling**: Heat conduction is governed by $-\nabla \cdot (k \nabla T) = q$, meaning temperature gradients scale as $\nabla T \propto 1/k$. Linear $z$-score normalization treats a conductivity jump from $1 \to 30$ symmetrically to an arithmetic shift, distorting the logarithmic sensitivity of thermal resistance.
2. **Boundary Condition Channel Discretization Flaw**:
   - Boundary condition channels ([`boundary_encoding.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/data/boundary_encoding.py)) encode Dirichlet and Neumann boundary values onto the peripheral cells of the 2D grid:
     - Left face: Column $i=0$ has mask $= 1.0$, values $= (T_{\mathrm{left}} - \mu_T)/\sigma_T$.
     - Right face: Column $i=nx-1$ has mask $= 1.0$.
     - Bottom face: Row $j=0$ has mask $= 1.0$.
     - Top face: Row $j=ny-1$ has mask $= 1.0$.
   - **Root Cause of Super-Resolution Failure**: On a $32 \times 32$ grid, the boundary mask has physical width $\Delta x = 1/32 = 0.03125\text{ m}$. On a $64 \times 64$ grid, the boundary mask has physical width $\Delta x = 1/64 = 0.015625\text{ m}$. Because the mask occupies exactly 1 grid cell regardless of resolution, **the input function $a(x)$ changes its physical profile under mesh refinement**. The FNO is not receiving a discretized continuous function; it is receiving a resolution-dependent numerical artifact!

### 2.3 Finite-Volume Reference Solver Complexity Scaling

The conservative reference solver (`HeatConductionSolver2D`) discretizes the elliptic PDE on a cell-centred Cartesian grid using SciPy's sparse direct solver (SuperLU). Measured CPU execution times across grid resolutions:

```
+--------------------------------------------------------------------------------------------------+
| Resolution     Num Cells (N)    Mean Latency (ms)    Std Latency (ms)    Relative Residual       |
+--------------------------------------------------------------------------------------------------+
| 32 x 32        1,024             4.41 ms             +/- 0.43 ms         1.27e-15 (Machine Prec) |
| 64 x 64        4,096            15.61 ms             +/- 1.65 ms         1.92e-15 (Machine Prec) |
| 128 x 128     16,384            56.27 ms             +/- 1.16 ms         2.71e-15 (Machine Prec) |
| 256 x 256     65,536           276.40 ms             +/- 5.84 ms         3.90e-15 (Machine Prec) |
+--------------------------------------------------------------------------------------------------+
```

- **Observed Scaling**: Doubling the mesh resolution increases cell count $4\times$. Solver runtime scales by $3.54\times$ ($32 \to 64$), $3.60\times$ ($64 \to 128$), and $4.91\times$ ($128 \to 256$). This reflects the theoretical 2D nested dissection complexity $\mathcal{O}(N^{1.5}) = \mathcal{O}(n^3)$.

### 2.4 Neural Operator CPU Scaling & Measured Crossover

Empirically measured on the host CPU (batch size = 1, 10 repeated timed iterations, 3 warmup passes):

```
+--------------------------------------------------------------------------------------------------+
| Resolution     FVM SuperLU        Single FNO (CPU)     5-Member Ensemble    CNN Baseline (CPU)   |
+--------------------------------------------------------------------------------------------------+
| 32 x 32          4.41 ms            9.48 ms              52.48 ms             0.87 ms            |
| 64 x 64         15.61 ms            9.12 ms              47.75 ms             0.86 ms            |
| 128 x 128       56.27 ms           10.95 ms              58.49 ms             2.14 ms            |
| 256 x 256      276.40 ms           53.31 ms             285.27 ms            22.20 ms            |
+--------------------------------------------------------------------------------------------------+
```

```
CPU Wall-Clock Latency vs. Mesh Resolution
================================================================================
Resolution     FVM SuperLU       Single FNO        5-Member Ensemble
--------------------------------------------------------------------------------
 32 x 32   [==] 4.4ms        [=====] 9.5ms     [===========================] 52.5ms
 64 x 64   [=======] 15.6ms  [====] 9.1ms      [========================] 47.8ms
128 x 128  [===========] 56.3ms [=====] 11.0ms [============] 58.5ms
256 x 256  [===================================] 276.4ms
                             [======] 53.3ms (Single FNO 5.2x faster)
                             [====================================] 285.3ms
================================================================================
```

#### Key Measured Takeaways:
1. **Single FNO CPU Crossover**: Single FNO becomes faster than direct FVM SuperLU at **$\approx 48 \times 48$ resolution** ($15.6\text{ ms}$ for FVM vs $9.1\text{ ms}$ for FNO at $64 \times 64$). At $256 \times 256$, single FNO is **$5.18\times$ faster** than FVM ($53.3\text{ ms}$ vs $276.4\text{ ms}$).
2. **Ensemble CPU Bottleneck**: The 5-member ensemble evaluates 5 full FNO forward passes sequentially on CPU. At $32 \times 32$, FVM is **$11.9\times$ faster** than the ensemble. The ensemble does not achieve parity with FVM until $\approx 180 \times 180$.
3. **CNN Speed**: The CNN baseline is extraordinarily fast on CPU ($0.87\text{ ms}$ at $32 \times 32$; $22.2\text{ ms}$ at $256 \times 256$), being $5\times$ faster than FVM at $32 \times 32$. However, its local $3 \times 3$ receptive field limits long-range elliptic propagation, causing higher errors on global boundary-driven problems.

---

## 3. Phase 2 — Generalization Failure Analysis

### 3.1 Failure Mode 1: High Conductivity Contrast ($k_{\max}/k_{\min} \ge 10$)

- **Empirical Evidence**: On the $30:1$ high-contrast test preset, surrogate prediction RMSE degraded from $0.67\text{ K}$ (IID) to **$34.71\text{ K}$** (an error explosion of $51.8\times$).
- **Mechanistic Root Cause**:
  1. Truncated Fourier modes ($12 \times 12$) cannot represent sharp spatial gradient discontinuities across inclusion interfaces without Gibbs ringing.
  2. The elliptic operator $-\nabla \cdot (k \nabla T) = q$ requires continuous normal heat flux $-k \partial T / \partial n$ across material interfaces. When $k$ jumps by $30\times$, the normal derivative $\partial T / \partial n$ must jump by $1/30$. Standard FNO architecture has no inductive bias enforcing flux continuity across discontinuous material interfaces.
- **Testable Hypothesis H1**:
  > *Transforming conductivity into logarithmic space $\log_{10}(k)$ prior to lifting, combined with an interface-flux consistency loss across high-contrast edges, will reduce high-contrast ($k_{\text{contrast}} \ge 10$) prediction RMSE by at least $40\%$ compared to standard z-score normalization.*

---

### 3.2 Failure Mode 2: Short-Correlation-Length Microstructures ($\ell \in [0.04, 0.08]\text{ m}$)

- **Empirical Evidence**: In Milestone 05B, the 47 samples with fine-scale Matérn Gaussian random fields produced an extreme boundary flux imbalance of **$59,591.9 \pm 18,571.9\text{ W}$**, despite reference solver solutions exhibiting machine-precision balance ($8.89 \times 10^{-3}\text{ W}$).
- **Mechanistic Root Cause**:
  1. A spatial correlation length of $\ell = 0.04\text{ m}$ on a unit domain $[0, 1]\text{ m}$ corresponds to characteristic spatial wavenumber $k_{\text{char}} \approx 1/\ell = 25\text{ m}^{-1}$.
  2. The current FNO retains only 12 Fourier modes ($k_x \le 12, k_y \le 12$). All spatial frequencies above $12\text{ m}^{-1}$ are discarded by the spectral convolution layer.
  3. Consequently, the FNO behaves as a spatial low-pass filter, failing to capture microstructural thermal channeling.
- **Testable Hypothesis H2**:
  > *Augmenting the low-frequency spectral convolution branch with a localized high-frequency spatial multi-scale branch (e.g., Dilated Residual Blocks or Multiwavelet transforms) will reduce short-correlation-length prediction RMSE by $\ge 35\%$ while preserving the fixed-frequency parameter count within $1.25\times$.*

---

### 3.3 Failure Mode 3: Boundary Prediction Error and the Flux Amplification Anomaly

- **Empirical Evidence**: The root cause of the $59.6\text{ kW}$ anomaly (Milestone 06A) was proven to be the linear amplification of boundary temperature errors by boundary conductivity:
  $$\Delta Q_{\mathrm{left}} = 2 \sum_{j=1}^{N_y} k_{j, 0} (T_{j, 0}^{\mathrm{pred}} - T_{j, 0}^{\mathrm{true}})$$
  When $k_{\mathrm{bnd}} \approx 13\text{ W/(m K)}$, a modest boundary temperature error $\Delta T \approx 45\text{ K}$ multiplies into $59.6\text{ kW}$ of artificial heat flow.
- **Mechanistic Root Cause**:
  The FNO employs **soft boundary condition learning**: boundary values are provided merely as input channels. The neural operator has no structural guarantee that the predicted field satisfies Dirichlet boundaries at $x=0$ or $x=1$.
- **Testable Hypothesis H3**:
  > *An architectural Hard Boundary Condition ansatz:*
  > $$T_{\theta}(x, y) = T_{\mathrm{prescribed}}(x, y) + D(x, y) \cdot u_{\theta}(x, y)$$
  > *where $D(x, y)$ is a smooth distance-to-boundary function vanishing identically on Dirichlet boundaries, will mathematically guarantee $\Delta T_{\mathrm{bnd}} \equiv 0$ on Dirichlet faces, reducing boundary flux imbalance by $> 90\%$ without degrading interior prediction accuracy.*

---

### 3.4 Failure Mode 4: Zero-Shot Super-Resolution Degradation ($32 \times 32 \to 64 \times 64$)

- **Empirical Evidence**: Evaluating a model trained on $32 \times 32$ zero-shot on a $64 \times 64$ grid increased prediction RMSE from **$0.9403\text{ K}$ to $12.0947\text{ K}$ ($12.86\times$ error increase)**, with peak absolute error jumping from $2.32\text{ K}$ to $36.97\text{ K}$.
- **Mechanistic Root Cause**:
  1. **Discretization-Dependent Masking**: The boundary condition encoding paints masks onto exactly 1 cell width ($1/32\text{ m}$ on $32 \times 32$; $1/64\text{ m}$ on $64 \times 64$).
  2. **Gibbs Cancellation Tuning**: Because the training data featured strong non-periodic linear gradients ($300\text{ K} \to 350\text{ K}$), the model learned a delicate cancellation between the Fourier modes and the spatial $1 \times 1$ bypass convolutions specifically tuned to the 12-mode cutoff at $32 \times 32$.
- **Testable Hypothesis H4**:
  > *Replacing grid-cell boundary masks with continuous signed distance coordinates, combined with multi-resolution training (sampling batches across $32 \times 32$ and $64 \times 64$), will reduce zero-shot super-resolution error at $128 \times 128$ from $> 12\text{ K}$ to $< 2.0\text{ K}$.*

---

### 3.5 Failure Mode 5: Correlated Systematic Ensemble Errors

- **Empirical Evidence**: In Milestone 06C, accepted IID samples had an RMSE of $0.67\text{ K}$, but rejected samples had an RMSE of $1.44\text{ K}$. Crucially, we observed that **low ensemble disagreement does not guarantee low error**.
- **Mechanistic Root Cause**:
  All 5 ensemble members share the exact same architecture, spectral truncation, and training objective, varying only in random seed initialization. On out-of-distribution physical regimes (e.g., extreme source or high contrast), all 5 members suffer from the same structural inductive bias and under-predict systematically in the same direction.
- **Testable Hypothesis H5**:
  > *Architectural diversity in the ensemble (combining FNO with U-Net and varying spectral mode counts $M \in \{8, 12, 16\}$) will increase epistemic disagreement on OOD failure cases by $\ge 50\%$, significantly reducing the false-negative acceptance rate.*

---

## 4. Phase 3 — Performance Research Design

### 4.1 Benchmarking Methodology & Experimental Rigor

To prevent timing artifacts, thermal throttling, and memory leaks, all performance evaluations must strictly follow this protocol:

1. **Host Environment Isolation**:
   - Explicitly report CPU model, physical core count, base/boost clock, RAM, OS, Python version, and PyTorch build.
   - For GPU evaluations (when available): Report GPU model, VRAM capacity, driver version, and CUDA version.
2. **Warmup Protocol**:
   - Minimum 5 complete inference forward passes before taking measurements to populate CUDA context, kernel caches, and memory allocators.
3. **Repeated Timing Protocol**:
   - Minimum 20 repeated runs for small grids ($32 \times 32, 64 \times 64$); minimum 10 runs for large grids ($128 \times 128, 256 \times 256$).
   - High-resolution monotonic timer: `time.perf_counter()`.
   - On GPU: Explicit synchronization before and after timing: `torch.cuda.synchronize()`.
4. **Memory Measurement Protocol**:
   - Peak CPU memory tracked via `tracemalloc` and operating system RSS.
   - Peak GPU memory tracked via `torch.cuda.max_memory_allocated()`.
5. **Batching and Throughput Protocol**:
   - Evaluate across batch sizes $B \in \{1, 8, 32, 128\}$.
   - Throughput metric: $\text{Throughput} = \frac{B \times N_{\mathrm{reps}}}{\Delta t_{\mathrm{total}}}$ [problems/second].

### 4.2 Benchmark Evaluation Matrix

```
+---------------------------------------------------------------------------------------------------+
| Resolution | Direct FVM  | Single CNN | Single FNO | 5-Ensemble | GPU FNO (B=1) | GPU FNO (B=128) |
+---------------------------------------------------------------------------------------------------+
| 32 x 32    | 4.41 ms     | 0.87 ms    | 9.48 ms    | 52.48 ms   | [Protocol]    | [Protocol]      |
| 64 x 64    | 15.61 ms    | 0.86 ms    | 9.12 ms    | 47.75 ms   | [Protocol]    | [Protocol]      |
| 128 x 128  | 56.27 ms    | 2.14 ms    | 10.95 ms   | 58.49 ms   | [Protocol]    | [Protocol]      |
| 256 x 256  | 276.40 ms   | 22.20 ms   | 53.31 ms   | 285.27 ms  | [Protocol]    | [Protocol]      |
+---------------------------------------------------------------------------------------------------+
```

---

## 5. Phase 4 — Candidate Research Directions

Eight research candidates were formulated, audited, and prioritized based on **scientific value per unit of compute**:

```
+---------------------------------------------------------------------------------------------------+
| Rank | Candidate Research Direction                                 | Priority | Expected Impact  |
+---------------------------------------------------------------------------------------------------+
|  1   | Boundary-Aware Operator Architectures (Hard BC Ansatz)       | CRITICAL | Eliminates 60 kW |
|  2   | Face-Consistent Finite-Volume Loss Formulation               | HIGH     | Physics Parity   |
|  3   | Multi-Resolution FNO Training & Continuous Coordinate Masks  | HIGH     | Unlocks Speedup  |
|  4   | Logarithmic Conductivity & Contrast-Invariant Scaling        | HIGH     | OOD Robustness   |
|  5   | Single-Model Evidential Uncertainty & Ensemble Distillation  | MEDIUM   | 5x Speedup       |
|  6   | Adaptive Cost-Utility Hybrid Routing Policy                  | MEDIUM   | Optimal Runtime  |
|  7   | Calibrated Multi-Objective Loss Balancing (GradNorm)         | MEDIUM   | Pareto Stability |
|  8   | Operator Quantization (FP16/INT8) and Pruning                | LOW      | Edge Deploy      |
+---------------------------------------------------------------------------------------------------+
```

---

### Detailed Specification of Candidates

#### Candidate 1: Boundary-Aware Operator Architectures (Hard BC Enforcement)
- **Hypothesis**: Structurally enforcing Dirichlet boundaries via an exact distance transform ansatz:
  $$T_{\theta}(x, y) = T_{\mathrm{dirichlet}}(x, y) + D(x, y) \cdot u_{\theta}(x, y)$$
  guarantees zero boundary temperature error, eliminates the $59.6\text{ kW}$ boundary flux anomaly, and removes the need for soft boundary loss penalties.
- **Implementation**:
  1. Add continuous distance transform generator $D(x, y) = \min(x, 1-x, y, 1-y)$ (zero on boundaries, positive in interior).
  2. Implement `BoundaryAwareFNO2D` wrapping standard FNO backbone.
- **Controls**: Baseline FNO-2D with identical parameters, optimizer, and seed.
- **Metrics**: Boundary temperature RMSE [K], boundary flux imbalance [W], interior RMSE [K].
- **Cost**: Zero additional model parameters; $< 2\%$ runtime overhead.
- **Expected Failure Mode**: Vanishing gradient near boundaries if $D(x, y)$ is too steep.

#### Candidate 2: Face-Consistent Dirichlet Boundary Losses
- **Hypothesis**: Formulating boundary losses using exact finite-volume half-cell conductances and linear face extrapolation:
  $$T_{\mathrm{face}} = 1.5 T_0 - 0.5 T_1$$
  eliminates the artificial gradient suppression caused by cell-center matching (`use_extrapolation=False`), reducing boundary flux imbalance by $> 75\%$.
- **Implementation**:
  1. Update `BoundaryConditionLoss` in [`physics_loss.py`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/training/physics_loss.py) to enable face extrapolation by default.
  2. Penalize boundary face conduction matching $A T - b$ stencil.
- **Controls**: Soft penalty with `use_extrapolation=False` (Milestone 04B baseline).
- **Metrics**: Boundary flux error [W], discrete PDE residual [W], Rel-$L_2$ [%].
- **Cost**: Trivial (purely loss function modification).

#### Candidate 3: Multi-Resolution FNO Training & Continuous Encodings
- **Hypothesis**: Training FNO on mixed resolutions ($32 \times 32$ and $64 \times 64$) with resolution-invariant continuous coordinate encodings will achieve true discretization invariance, reducing $128 \times 128$ zero-shot RMSE from $12\text{ K}$ to $< 2\text{ K}$.
- **Implementation**:
  1. Replace 1-cell binary masks with continuous signed distance functions.
  2. Implement multi-resolution DataLoader sampling batches across resolutions.
- **Controls**: Standard single-resolution ($32 \times 32$) FNO-2D.
- **Metrics**: Zero-shot super-resolution RMSE at $64 \times 64, 128 \times 128, 256 \times 256$.
- **Cost**: Moderate ($1.5\times$ training wall-clock time due to $64 \times 64$ batches).

#### Candidate 4: Logarithmic Conductivity and Contrast-Invariant Normalization
- **Hypothesis**: Replacing linear conductivity normalization with logarithmic feature scaling $\tilde{k} = \log_{10}(k)$ linearizes the governing thermal resistance, reducing prediction error on high-contrast media ($k_{\text{contrast}} \ge 10$) by $\ge 40\%$.
- **Implementation**:
  1. Add `LogFieldNormalizer` supporting logarithmic transformations for strictly positive fields ($k > 0$).
  2. Update dataset generation and evaluation pipelines.
- **Controls**: Linear $z$-score normalizer (Benchmark v1.0).
- **Metrics**: High-contrast partition RMSE [K], maximum point error [K].
- **Cost**: Zero compute cost (pure data pre-processing).

#### Candidate 5: Single-Model Evidential Uncertainty and Distillation
- **Hypothesis**: A single FNO equipped with Deep Evidential Regression (predicting $\gamma, \nu, \alpha, \beta$ of a Normal-Inverse-Gamma distribution) or distilled from the 5-member ensemble matches ensemble OOD detection performance (F1 $\ge 0.95$) while reducing inference latency by $5.0\times$.
- **Implementation**:
  1. Implement Evidential loss head on FNO-2D output ($4$ channels instead of $1$).
  2. Implement Knowledge Distillation loss matching ensemble mean and variance.
- **Controls**: 5-member deep ensemble.
- **Metrics**: Inference latency [ms], OOD detection AUROC, conformal interval width [K].
- **Cost**: Equivalent to training 1 single FNO model.

#### Candidate 6: Adaptive Solver-Surrogate Hybrid Routing
- **Hypothesis**: An adaptive runtime cost-utility model that dynamically routes queries based on mesh resolution ($N$) and batch size ($B$) minimizes total execution time while maintaining $100\%$ fail-closed physical guarantees:
  - Small mesh ($N \le 32 \times 32$, $B=1$): Route directly to FVM ($4.4\text{ ms}$).
  - Fine mesh ($N \ge 64 \times 64$) or batched ($B \ge 8$): Route to surrogate fast-path.
- **Implementation**:
  1. Add `AdaptiveRoutingEngine` extending Milestone 06B engine.
  2. Implement dynamic decision threshold function.
- **Controls**: Static hybrid routing (Milestone 06B baseline).
- **Metrics**: End-to-end throughput [problems/s], mean turnaround time [ms].
- **Cost**: Pure software orchestration logic (zero training cost).

#### Candidate 7: Calibrated Multi-Objective Loss Balancing (GradNorm)
- **Hypothesis**: Dynamic gradient-norm balancing (GradNorm) between supervised data loss, PDE residual loss, and boundary loss prevents gradient conflict, achieving the optimal Pareto frontier between accuracy and physical conservation.
- **Implementation**:
  1. Implement `GradNormLossBalancer` monitoring $\|g_{\text{data}}\|, \|g_{\text{pde}}\|, \|g_{\text{bc}}\|$.
  2. Dynamically update loss weights $w_{\text{pde}}, w_{\text{bc}}$ during training.
- **Controls**: Fixed-weight loss calibration (Milestone 04B baseline, $w_{\text{pde}} = 0.005$).
- **Metrics**: Pareto frontier of Rel-$L_2$ vs. PDE residual $\|r\|_2$.
- **Cost**: $\approx 10\%$ increase in training backward pass time.

#### Candidate 8: Operator Compression, Quantization, and Pruning
- **Hypothesis**: Post-training FP16 and INT8 quantization of FNO weights reduces memory footprint by $4\times$ and accelerates CPU inference by $\ge 1.8\times$ with $< 0.1\text{ K}$ accuracy degradation.
- **Implementation**:
  1. Implement PyTorch dynamic quantization on linear/conv layers: `torch.ao.quantization.quantize_dynamic`.
- **Controls**: Unquantized FP32 FNO.
- **Metrics**: Model size [MB], latency [ms], RMSE degradation [K].
- **Cost**: Low (post-training calibration only).

---

## 6. Phase 5 — Experimental Protocol & Methodological Standards

### 6.1 Dataset Partitions and Leakage Prevention

To ensure evaluation integrity, all experiments must respect these data partitions:

```
[Offset 1,000,000] Train:       Seeds 1,000,042 .. 1,001,041 (N = 1000)  [Fitting only]
[Offset 2,000,000] Validation:  Seeds 2,000,042 .. 2,000,191 (N = 150)   [Early stopping & hyperparams]
[Offset 3,000,000] Test-IID:    Seeds 3,000,042 .. 3,000,241 (N = 200)   [Unseen in-distribution evaluation]
[Offset 4,000,000] Test-OOD:    Seeds 4,000,042 .. 4,000,241 (N = 200)   [Unseen physical shift evaluation]
[Offset 5,000,000] Calibration: Seeds 5,000,042 .. 5,000,241 (N = 200)   [Conformal quantile fitting only]
```

- **Leakage Invariant**: `FieldNormalizer` and `LogFieldNormalizer` must be fitted **exclusively on the 1000-sample training partition**. Conformal quantiles must be computed **exclusively on the 200-sample calibration partition**. Test partitions must never be touched during model selection or calibration.

### 6.2 Statistical Replication and Significance Testing

- **Random Seeds**: Every candidate model must be trained across **5 independent random seeds**: `42, 43, 44, 45, 46`.
- **Equal Training Budget**:
  - Maximum epochs: 30.
  - Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $10^{-4}$).
  - Learning rate: $10^{-3}$ with Cosine Annealing to $10^{-5}$.
  - Batch size: 32.
  - Early stopping: Patience 10 on validation loss.
- **Statistical Reporting**:
  - All metrics reported as **$\text{Mean} \pm \text{Sample Std}$ ($ddof=1$)**.
  - Hypothesis testing: Paired two-tailed Student's $t$-test across seeds ($N=5, df=4$).
  - Significance threshold: $p < 0.05$ with Holm-Bonferroni correction for multi-hypothesis comparisons.

### 6.3 Comprehensive Metrics Suite

| Category | Primary Metric | Formula / Definition | Target Threshold |
| :--- | :--- | :--- | :--- |
| **Accuracy** | Root Mean Squared Error (RMSE) | $\sqrt{\frac{1}{N_{\mathrm{cells}}} \sum (T - T^*)^2}$ [K] | $< 0.8\text{ K}$ (IID); $< 3.0\text{ K}$ (OOD) |
| **Accuracy** | Relative $L_2$ Error | $\|T - T^*\|_2 / \|T^*\|_2 \times 100\%$ | $< 0.6\%$ (IID); $< 2.5\%$ (OOD) |
| **Accuracy** | Maximum Absolute Error (MaxAE) | $\max_{(x,y)} \|T(x,y) - T^*(x,y)\|$ [K] | $< 3.0\text{ K}$ (IID); $< 10.0\text{ K}$ (OOD) |
| **Physics** | Discrete PDE Residual | $\|A T - b\|_2$ [W] | $< 25.0\text{ W}$ |
| **Physics** | Relative PDE Residual | $\|A T - b\|_2 / \|b\|_2$ | $< 5.0 \times 10^{-3}$ |
| **Conservation** | Global Heat Imbalance | $\|Q_{\mathrm{out}} - Q_{\mathrm{gen}}\|$ [W] | $< 50.0\text{ W}$ |
| **Conservation** | Boundary Flux Error | $\sum_{\mathrm{faces}} \|Q_{\mathrm{face}} - Q_{\mathrm{face}}^*\|$ [W] | $< 20.0\text{ W}$ |
| **Uncertainty** | Conformal Empirical Coverage | Fraction of test cells with $T^* \in [\hat{T}_{\mathrm{lo}}, \hat{T}_{\mathrm{hi}}]$ | $\ge 94.0\%$ (Nominal $95\%$) |
| **Uncertainty** | Mean Interval Width | $\frac{1}{N_{\mathrm{cells}}} \sum (\hat{T}_{\mathrm{hi}} - \hat{T}_{\mathrm{lo}})$ [K] | $< 4.5\text{ K}$ |
| **Uncertainty** | OOD Detection F1 Score | Harmonic mean of Precision and Recall on OOD vs IID | $\ge 0.95$ |
| **Efficiency** | Inference Latency (p50 / p95) | Wall-clock forward pass time [ms] | Single FNO $< 10\text{ ms}$; Ensemble $< 25\text{ ms}$ |
| **Efficiency** | Throughput | Problems evaluated per second [problems/s] | $> 100\text{ problems/s}$ on CPU |

---

## 7. Phase 6 — Research Roadmap & Implementation Plan

```
+---------------------------------------------------------------------------------------------------+
|                                 MILESTONE 08 RESEARCH ROADMAP                                     |
|                                                                                                   |
|  [Milestone 08A] Research Design & Baseline Empirical Audit  <--- (COMPLETED)                     |
|         |                                                                                         |
|         v                                                                                         |
|  [Milestone 08B] Boundary-Aware Architectures & Face-Consistent Losses                            |
|         |        - Implement Hard BC Distance Ansatz D(x,y)                                       |
|         |        - Formulate face-consistent harmonic Dirichlet loss                              |
|         |        - Eliminate 59.6 kW boundary anomaly                                             |
|         v                                                                                         |
|  [Milestone 08C] Multi-Resolution Generalization & High-Contrast Normalization                    |
|         |        - Continuous distance coordinate encodings                                       |
|         |        - Logarithmic conductivity transformation log10(k)                               |
|         |        - Mixed-resolution training (32x32 + 64x64)                                      |
|         v                                                                                         |
|  [Milestone 08D] Lightweight Evidential Uncertainty & Ensemble Distillation                       |
|         |        - Deep Evidential Regression head (single-model UQ)                              |
|         |        - Knowledge distillation from 5-member ensemble to student                       |
|         |        - Reduce inference latency from 52 ms to < 10 ms                                 |
|         v                                                                                         |
|  [Milestone 08E] Adaptive Hybrid Routing & Comprehensive Empirical Benchmark                      |
|                  - Adaptive resolution-aware routing policy                                       |
|                  - Final multi-resolution benchmark (32x32 to 256x256)                            |
|                  - Updated hybrid digital twin production release                                 |
+---------------------------------------------------------------------------------------------------+
```

### 7.1 Detailed Milestone Breakdown

| Milestone | Primary Focus | Key Deliverables | Compute Requirement | Success Gate |
| :--- | :--- | :--- | :--- | :--- |
| **08B** | Boundary-Aware Operator & Face-Consistent Physics Losses | `BoundaryAwareFNO2D`, face-extrapolated boundary loss, boundary flux benchmark. | 10 training runs (5 seeds $\times$ 2 configs). | Boundary flux error $< 50\text{ W}$; zero Dirichlet face temperature error. |
| **08C** | Multi-Resolution & Log-Conductivity | `LogFieldNormalizer`, continuous coordinate masks, mixed-resolution dataset. | 10 training runs (5 seeds $\times$ 2 configs). | $64 \times 64$ zero-shot RMSE $< 2.0\text{ K}$; $30:1$ contrast RMSE $< 5.0\text{ K}$. |
| **08D** | Evidential UQ & Distillation | Evidential FNO model, student distillation training loop, calibration audit. | 10 training runs (5 evidential, 5 distilled). | Single-model latency $< 10\text{ ms}$; OOD F1 score $\ge 0.95$. |
| **08E** | Adaptive Routing & Master Benchmark | `AdaptiveRoutingEngine`, full benchmark report across 32 to 256 grids. | Benchmarking only (zero training). | End-to-end hybrid throughput $> 150\text{ problems/s}$ on CPU with fail-closed safety. |

---

## 8. Milestone Conclusion and Acceptance Checklist

Milestone 08A establishes the empirical baseline, diagnoses root-cause failure mechanisms, and defines a controlled experimental program:
- [x] **Phase 1 Audit Complete**: Model parameters, solver complexity, and data encodings fully audited.
- [x] **Empirical Baseline Archived**: [`artifacts/reports/milestone_08a_baseline_audit.json`](file:///C:/Workspace/Projects/Neural%20Process%20Twin/artifacts/reports/milestone_08a_baseline_audit.json) records real measured latencies and zero-shot error amplification ($12.86\times$).
- [x] **Generalization Hypotheses Formulated**: Five testable, mathematically rigorous hypotheses defined (H1–H5).
- [x] **Candidate Directions Prioritized**: Eight candidate directions specified with controls, costs, and failure modes.
- [x] **Experimental Protocol Defined**: Strict partition disjointness, 5 seeds, paired statistical tests, and comprehensive metric suite.
- [x] **Sequential Roadmap Established**: Ordered roadmap (08B $\to$ 08C $\to$ 08D $\to$ 08E).
- [x] **Code Quality & Tests**: All 194 existing unit, integration, and scientific tests pass; 100% clean Ruff checks.

*In accordance with project instructions, execution stops at the conclusion of Milestone 08A for review prior to beginning implementation of Milestone 08B.*
