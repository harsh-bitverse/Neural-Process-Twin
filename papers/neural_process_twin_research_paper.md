# Neural Process Twin: A Multi-Resolution Neural Operator and Conservative Finite-Volume Platform for Transparent Heat-Conduction Simulation

**Working Preprint Manuscript**  
*Neural Process Twin Research Team*  
*Repository: https://github.com/harsh-bitverse/Neural-Process-Twin*  
*(Note: This manuscript is an un-refereed working research paper documenting open-source scientific software and empirical findings.)*

---

## Abstract

Data-driven neural operators promise rapid surrogate modeling for parameterized partial differential equations (PDEs), yet engineering deployment requires verifiable physical conservation, boundary fidelity, and transparent error characterization. We present **Neural Process Twin**, a dual-mode scientific simulation platform for 2D steady-state heat conduction governed by the variable-conductivity elliptic equation $-\nabla \cdot (k \nabla T) = q$. The platform couples a conservative cell-centered Finite Volume Method (FVM) reference solver with an experimental multi-resolution Fourier Neural Operator ($A_3$ `RefinedFNO2D`) and an optional $A_4$ Selective Boundary Jacobi smoother. 

The reference FVM solver enforces discrete energy conservation ($\Delta Q_{\text{rel}} \le 10^{-12}$) and demonstrates theoretical second-order convergence ($p = 2.000$) under the Method of Manufactured Solutions (MMS). The neural surrogate achieves zero-shot multi-resolution transfer from $32\times 32$ to $256\times 256$ with an **8.76x end-to-end API speedup** (499.3 ms vs 4,373.7 ms at $256\times 256$) when physical diagnostics are deferred. 

Crucially, we report two major negative findings that challenge prevailing assumptions in physics-informed machine learning:
1. Standard uncertainty quantification methods (deep ensembles, Monte Carlo dropout, split conformal prediction) exhibit catastrophic confidence collapse under physical distribution shift, admitting false acceptances with peak errors exceeding 48 K MaxAE.
2. Rigorous mathematical a posteriori $L_\infty$ error bounds based on the discrete maximum principle blow up as $O(h^{-2})$, overestimating actual surrogate error by up to $19,783\times$ on fine meshes and rendering rigorous single-pass certification impractical.

To maintain scientific integrity, Neural Process Twin implements an explicit architectural isolation: Verified Numerical Mode serves as the authoritative, certified reference, while Experimental Neural Mode is exposed with transparent diagnostic disclaimers and process-local resource governors.

---

## 1. Introduction

Accurate prediction of steady-state temperature distributions in heterogeneous media is critical across thermal management domains, including microelectronics packaging, power converters, battery modules, and aerospace structures. While classical numerical discretizations—such as the Finite Volume Method (FVM) and Finite Element Method (FEM)—provide rigorous discretization error control and exact discrete conservation, their computational cost scales super-linearly with grid cell count $N$, becoming prohibitive for multi-query optimization and real-time digital twins.

Recent advances in neural operators, particularly Fourier Neural Operators (FNO) [Li et al., 2021] and Deep Operator Networks (DeepONet) [Lu et al., 2021], enable learning non-linear mappings between infinite-dimensional function spaces. However, standard neural operators face three fundamental scientific challenges in practical engineering deployment:
1. **High-Frequency & Boundary Mode Truncation:** FNO truncates high-frequency spectral modes ($k > k_{\max}$), producing Gibbs oscillations and elevated localized errors near Dirichlet boundaries.
2. **Failure of Epistemic Uncertainty under Physical Shifts:** Surrogates calibrated on in-distribution (IID) geometries frequently collapse in confidence when exposed to out-of-distribution (OOD) conductivity contrasts or source peaks.
3. **The Certification Gap:** Engineering workflows require upper bounds or high-confidence guarantees on maximum absolute error ($\text{MaxAE} \le \tau$), which standard loss metrics ($L_2$, relative RMSE) fail to ensure.

This paper presents the design, mathematical formulation, empirical benchmarks, and architectural governance of Neural Process Twin. We demonstrate that rather than attempting to force uncertified neural predictions into safety-critical pipelines through fragile uncertainty thresholds, the optimal engineering approach is a transparent dual-mode architecture that preserves verified numerical simulation as the sole trusted standard while providing accelerated neural exploration.

---

## 2. Related Work

### 2.1 Numerical Finite Volume Methods
Conservative cell-centered finite volume methods are foundational in computational heat transfer [Patankar, 1980; Eymard et al., 2000]. By integrating conservation laws over discrete control volumes, FVM guarantees that face fluxes between adjacent control volumes cancel telescopically, ensuring exact global energy conservation. For heterogeneous media with discontinuous thermal conductivity, harmonic interface averaging is mathematically necessary to satisfy flux continuity across control volume interfaces [Versteeg & Malalasekera, 2007].

### 2.2 Neural Operators and Physics-Informed Learning
Fourier Neural Operators [Li et al., 2021] parameterize the integral kernel in Fourier space via the Fast Fourier Transform (FFT). While FNO demonstrates zero-shot super-resolution, standard formulations assume periodic or regular boundary conditions and struggle with non-periodic flux boundaries without specialized encodings [Li et al., 2023]. Physics-Informed Neural Networks (PINNs) [Raissi et al., 2019] penalize differential equation residuals via automatic differentiation, but suffer from optimization pathologies and training instability on multiscale and high-contrast problems [Wang et al., 2021].

### 2.3 Uncertainty Quantification & Hybrid Computing
Deep ensembles [Lakshminarayanan et al., 2017] and Monte Carlo dropout [Gal & Ghahramani, 2016] are widely used for epistemic uncertainty estimation. Split conformal prediction [Angelopoulos & Bates, 2021] provides distribution-free, finite-sample marginal coverage guarantees. However, recent theoretical work highlights that marginal conformal guarantees do not imply conditional coverage under localized physical shifts [Barber et al., 2021], a phenomenon we examine empirically in this work.

---

## 3. Mathematical Formulation

### 3.1 Governing Elliptic Equation
We consider steady-state heat conduction on a bounded Cartesian domain $\Omega = (0, L_x) \times (0, L_y) \subset \mathbb{R}^2$ with boundary $\partial \Omega$:

$$-\nabla \cdot \big(k(\mathbf{x}) \nabla T(\mathbf{x})\big) = q(\mathbf{x}), \quad \mathbf{x} = (x, y) \in \Omega$$

subject to boundary conditions on $\partial \Omega = \Gamma_D \cup \Gamma_N$:
- Dirichlet: $T(\mathbf{x}) = T_D(\mathbf{x})$ on $\Gamma_D$
- Neumann: $-k(\mathbf{x}) \nabla T(\mathbf{x}) \cdot \mathbf{n} = q''_N(\mathbf{x})$ on $\Gamma_N$

where $k(\mathbf{x}) \in [k_{\min}, k_{\max}]$ with $k_{\min} > 0$, and $q(\mathbf{x}) \in \mathbb{R}$ represents volumetric heat sources ($q > 0$) or endothermic sinks ($q < 0$).

### 3.2 Discrete Finite-Volume Formulation
The domain is partitioned into $N_x \times N_y$ cells of size $\Delta x \times \Delta y$ with thickness $\Delta z$. Control volume integration over cell $P$ yields:

$$\sum_{f \in \{w, e, s, n\}} Q_f = q_P V_P$$

where $V_P = \Delta x \Delta y \Delta z$. Outward normal face heat flow $Q_e$ across the East face between cell $P$ and $E$ is:

$$Q_e = -k_e \frac{T_E - T_P}{\Delta x} (\Delta y \Delta z)$$

Continuity of temperature and flux requires harmonic averaging of interface conductivity:

$$k_e = \frac{2 k_P k_E}{k_P + k_E}$$

Assembly across all $N = N_x N_y$ cells yields the sparse linear system:

$$\mathbf{A} \mathbf{T} = \mathbf{b}$$

When $\Gamma_D \ne \emptyset$, $\mathbf{A} \in \mathbb{R}^{N \times N}$ is a symmetric, strictly diagonally dominant M-matrix, possessing the following fundamental properties:
1. Positivity of diagonal entries: $A_{ii} > 0$.
2. Non-positivity of off-diagonal entries: $A_{ij} \le 0$ for $i \ne j$.
3. Inverse non-negativity: $\mathbf{A}^{-1} \ge 0$ (elementwise).
4. Discrete Maximum Principle: If $\mathbf{b} \ge 0$, then $\mathbf{T} = \mathbf{A}^{-1} \mathbf{b} \ge 0$.

---

## 4. Methodology and Neural Architecture

### 4.1 $A_3$ Refined Fourier Neural Operator
Standard FNO truncates spatial frequencies above mode $k_{\max} = 12$, producing Gibbs-like truncation errors near domain boundaries. To mitigate this without exploding parameter counts, we designed $A_3$ `RefinedFNO2D`, coupling global spectral integration with local convolutional refinement:

1. **Continuous Physical Encoding:** An input tensor $\mathbf{X} \in \mathbb{R}^{H \times W \times 12}$ encoding thermal conductivity $k(x, y)$, logarithmic conductivity $\log k$, volumetric source $q(x, y)$, normalized Cartesian coordinates $(x, y)$, and distance-weighted boundary condition fields ($T_{\text{dir}}$, $q''_{\text{neu}}$) computed via continuous signed distance functions.
2. **Spectral Convolutional Core:** 4 Fourier spectral layers with $k_{\max} = 12$ modes in $x$ and $y$, channel dimension $d = 32$, and GELU non-linearities.
3. **Local Residual Refinement Block:** A 3-layer convolutional residual block ($3\times 3$ kernels, padding 1, width 32) operating directly on the lifted spectral feature map. This local block learns steep spatial gradients and boundary layer transitions that lie beyond the $12\times 12$ Fourier cutoff.

### 4.2 $A_4$ Selective Boundary Jacobi Physics Smoother
To correct persistent boundary truncation errors post-inference without re-solving the full PDE, we developed the Selective Boundary Jacobi smoother.
Given the discrete operator $\mathbf{A} \mathbf{T} = \mathbf{b}$, standard damped Jacobi relaxation is:

$$T_P^{(m+1)} = (1 - \omega) T_P^{(m)} + \omega D_{PP}^{-1} \left(b_P - \sum_{Q \in \mathcal{N}(P)} A_{PQ} T_Q^{(m)}\right)$$

Instead of updating all $N$ cells, updates are restricted strictly to the outer 2-cell boundary ring:

$$\Omega_{\text{ring}} = \{ (i, j) \mid i \in \{0, 1, N_x-2, N_x-1\} \lor j \in \{0, 1, N_y-2, N_y-1\} \}$$

At $256\times 256$ resolution, $|\Omega_{\text{ring}}| = 2,032$ cells, updating only 3.1% of the grid. This operation executes in $<15\text{ ms}$, modifying boundary errors while leaving interior neural predictions untouched.

---

## 5. Experimental Design and Benchmark Protocol

### 5.1 Training Protocol
The $A_3$ model was trained across mixed resolutions ($32\times 32$ and $64\times 64$) using AdamW with cosine annealing for 100 epochs. The composite objective function was:

$$\mathcal{L} = \mathcal{L}_{\text{rel-}L_2} + 0.1 \mathcal{L}_{\text{boundary}} + 0.05 \mathcal{L}_{\text{residual}}$$

### 5.2 Independent Evaluation Cohort
Evaluation was conducted on a pristine cohort of 200 problem realizations across 4 resolutions ($32\times 32, 64\times 64, 128\times 128, 256\times 256$) spanning four physical regimes:
- **IID (Nominal):** Circular and elliptical inclusions ($k_{\text{feature}} / k_{\text{bg}} \in [2, 6]$), smooth source ($q \in [20, 80]\text{ W/m}^3$), $T \in [300, 380]\text{ K}$.
- **OOD High Contrast:** Severe conductivity ratios ($k_{\text{ratio}} \in [15, 50]$).
- **OOD High Source:** Gaussian hotspots ($q_{\text{peak}} \in [300, 900]\text{ W/m}^3, \sigma \in [0.1, 0.2]$).
- **OOD Boundary Step:** Macroscopic cross-domain temperature jumps ($T_{\text{left}} = 180\text{ K}, T_{\text{right}} = 620\text{ K}$).

---

## 6. Experimental Results

### 6.1 Numerical MMS Verification
Verification via the Method of Manufactured Solutions confirmed asymptotic second-order spatial convergence ($p = 2.000$) for smooth conductivities, matching theoretical $O(h^2)$ truncation error. For discontinuous piecewise interfaces, convergence reduced to $p = 0.998 \approx 1.00$ ($O(h^1)$), validating the harmonic interface formulation.

### 6.2 Zero-Shot Resolution Invariance & Accuracy

| Resolution | Regime | FVM Solve [ms] | $A_3$ Forward [ms] | $A_3$ RMSE [K] | $A_4$ RMSE [K] | MaxAE Median [K] | MaxAE P95 [K] |
|------------|--------|----------------|--------------------|----------------|----------------|------------------|---------------|
| $32 \times 32$ | IID | 52.1 ms | 63.2 ms | 6.24 K | 6.13 K | 19.18 K | 20.70 K |
| $64 \times 64$ | IID | 260.4 ms | 64.5 ms | 6.35 K | 6.28 K | 19.85 K | 21.40 K |
| $128 \times 128$ | IID | 845.2 ms | 66.8 ms | 6.42 K | 6.38 K | 20.50 K | 23.10 K |
| $256 \times 256$ | IID | 3,810.0 ms | 78.5 ms | 6.49 K | 6.48 K | 21.21 K | 26.09 K |
| $256 \times 256$ | OOD Contrast | 3,879.7 ms | 82.1 ms | 27.89 K | 27.88 K | 74.69 K | 83.61 K |
| $256 \times 256$ | OOD Source | 3,561.1 ms | 80.4 ms | 2.67 K | 2.65 K | 7.20 K | 9.83 K |
| $256 \times 256$ | OOD Boundary | 3,860.5 ms | 83.0 ms | 69.26 K | 69.22 K | 108.02 K | 122.36 K |

*Observations:*
1. **Zero-Shot Transfer:** Across resolutions $32\times 32 \to 256\times 256$, IID RMSE varies by $<0.25\text{ K}$ (6.24 K to 6.49 K), demonstrating resolution invariance.
2. **Physical Sensitivity:** In contrast, shifting from IID to OOD Boundary Step increases RMSE by more than $10\times$ (6.49 K to 69.26 K), and peak MaxAE reaches 122.36 K.

### 6.3 End-to-End API Performance & Speedup

When physical diagnostic computation is deferred (`compute_diagnostics=False`), the neural API eliminates redundant sparse matrix assembly, achieving an **8.76x end-to-end API speedup** over direct FVM at $256\times 256$ (499.3 ms vs 4,373.7 ms).

---

## 7. Critical Negative Findings

### 7.1 Epistemic Uncertainty Collapse under Physical Shift
A prevalent hypothesis in scientific machine learning is that epistemic uncertainty estimators (e.g. MC dropout, deep ensemble variance) reliably flag inaccurate predictions.
Our empirical audits refute this hypothesis:
- Under OOD high conductivity contrast ($C_k \ge 15$), the model outputted low uncertainty ($\sigma \le 3.0\text{ K}$) despite true local errors exceeding $48.0\text{ K}$.
- When automated routing gates were deployed ($R_2$, $R_3$), the single-pass policy admitted an **11% unconditional false acceptance rate on OOD queries**, with peak accepted local errors reaching 48.04 K.
- Furthermore, **76% to 82% of accepted IID predictions violated the engineering tolerance ceiling of $\text{MaxAE} \le 10\text{ K}$**.

### 7.2 The $O(h^{-2})$ PDE Error Certification Dilemma
To achieve mathematical error certification without running FVM, we investigated a posteriori error bounds derived from the discrete error equation $\mathbf{A} \mathbf{e} = \mathbf{r}$:

$$\|\mathbf{e}\|_\infty \le \|\mathbf{A}^{-1}\|_\infty \|\mathbf{r}\|_\infty$$

Due to the discrete maximum principle for elliptic operators, the stability constant scales as $\|\mathbf{A}^{-1}\|_\infty = \mathcal{O}(h^{-2}) = \mathcal{O}(N)$.
- At $32\times 32$, the bound overestimated true error by $300\times$.
- At $256\times 256$, the bound overestimated true error by **$19,783\times$** (reporting theoretical upper bounds of $465,000\text{ K}$ on an actual error of $23.5\text{ K}$).
- Tight barrier majorant solves ($\mathbf{A} \mathbf{w} = |\mathbf{r}|$) eliminate overestimation but require a direct sparse LU solve costing **93.4% of the reference FVM solve time**.

**Scientific Conclusion:** Single-pass mathematical error certification of neural operators on fine elliptic meshes is computationally infeasible; rigorous bounds are either wildly over-conservative or cost as much as the numerical solve itself.

---

## 8. Platform Architecture and Dual-Mode Governance

In response to these findings, Neural Process Twin replaces automated hybrid routing with an explicit **dual-mode operational paradigm**:

1. **Verified Numerical Mode (`numerical_fvm`):** Default, trusted computational standard. Computes cell-centered FVM solutions with machine-precision relative algebraic residuals ($\|r\|_2 / \|b\|_2 \le 3.54 \times 10^{-14}$) and certified physical energy balance.
2. **Experimental Neural Mode (`experimental_neural` / `experimental_neural_corrected`):** Fast surrogate inference (8.76x API speedup at $256\times 256$). Returned responses unconditionally contain `is_certified: false` and explicit disclaimers regarding uncertified local error bounds.
3. **Dual-Mode Comparison API (`/api/compare`):** Executes both pathways concurrently and returns the ground-truth spatial error field $\Delta T(x, y)$, providing real-time visual error attribution for exploratory research.
4. **Defensive Resource Governance:** Implements `ResourceSafetyMiddleware` (10 MB payload ceiling) and `ConcurrencyGovernor` (semaphores bounding concurrent FVM solves to 4 and neural passes to 8 with 5.0s queue timeouts).

---

## 9. Conclusion

Neural Process Twin bridges the gap between deep operator learning and rigorous scientific computing. Our findings establish that while Fourier Neural Operators provide remarkable computational acceleration and resolution-invariant feature extraction, they cannot yet provide certified engineering reliability under physical distribution shifts. By codifying an explicit separation between verified numerical solvers and experimental neural surrogates, the platform establishes a defensible, reproducible framework for next-generation scientific digital twins.

---

## References

- Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv preprint arXiv:2107.07511*.
- Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2021). The limits of distribution-free conditional predictive inference. *Information and Inference: A Journal of the IMA*, 10(2), 455-482.
- Eymard, R., Gallouët, T., & Herbin, R. (2000). The finite volume method. *Handbook of Numerical Analysis*, 7, 713-1018.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a bayesian approximation: Representing model uncertainty in deep learning. *International Conference on Machine Learning (ICML)*, 1050-1059.
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.
- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2021). Fourier neural operator for parametric partial differential equations. *International Conference on Learning Representations (ICLR)*.
- Li, Z., Huang, D. Z., Liu, B., & Anandkumar, A. (2023). Fourier neural operator with learned deformations for PDEs on general geometries. *Journal of Machine Learning Research*, 24(388), 1-53.
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021). Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3(3), 218-229.
- Patankar, S. V. (1980). *Numerical Heat Transfer and Fluid Flow*. Hemisphere Publishing Corporation.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
- Versteeg, H. K., & Malalasekera, W. (2007). *An Introduction to Computational Fluid Dynamics: The Finite Volume Method*. Pearson Education.
- Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055-A3081.
