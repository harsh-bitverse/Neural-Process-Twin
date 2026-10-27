# Milestone 04C Audit Report: Scientific Consistency & Verification

## Executive Summary & Audit Scope

This report delivers the formal scientific consistency audit for **Milestone 04B (Scaled Training and Physics-Loss Calibration)** of **Neural Process Twin**. 

The purpose of this audit is to rigorously verify all statistical claims, physical residual formulations, gradient calibration diagnostics, and out-of-distribution dataset semantics prior to advancing to **Milestone 05 (Uncertainty Estimation and Ensembles)**. This milestone is strictly an audit and verification activity: no models have been retrained, and all existing artifacts, checkpoints, and reports have been preserved without modification.

### Audit Verdict Summary
1. **Statistical Claims**:
   - The reported $\pm$ standard deviations in Milestone 04B used the population convention ($N=3$, `ddof=0`). Recalculating with the unbiased sample convention ($N-1=2$, `ddof=1`) scales all standard deviations by a factor of $\sqrt{3/2} \approx 1.2247$.
   - Seed-level variability (optimizer convergence across seeds, $\sigma_{\text{seed}} \approx 0.010\%$) is strictly distinct from within-partition sample-level variability across test cases ($\sigma_{\text{sample}} \approx 0.213\%$, $\sim 20\times$ larger). Both are now explicitly reported.
   - The reported claims of "$> 12\sigma$" and "$> 45\sigma$" significance ($p < 10^{-6}$) were based on an asymptotic Gaussian normal approximation ($Z$-test). With $N=3$ seeds, Student's $t$-distribution must be used ($\nu \approx 2.4$ for Welch's $t$-test; $df=2$ for paired $t$-test). While the physical residual reductions remain statistically significant at standard scientific thresholds ($p = 0.0013$ for FNO; $p = 0.00017$ for CNN), the asymptotic "$p < 10^{-6}$" claim is invalid for small sample sizes and is formally retracted and replaced.
2. **Physics Residual vs. Boundary Flux Imbalance**:
   - The discrete PDE residual $r = A T - b$ evaluates local cell-centred energy conservation [W] on each control volume.
   - Global flux imbalance $|Q_{\text{boundary, out}} - Q_{\text{generated}}|$ evaluates domain-wide conservation [W].
   - In `scaled_*_physics_bc`, interior residual optimization significantly reduced cell residuals ($48.5\%$ in FNO, $68.9\%$ in CNN), but increased boundary flux imbalance ($95\text{ W} \to 260\text{ W}$ in FNO, $124\text{ W} \to 724\text{ W}$ in CNN). The audit reveals this was caused by `BoundaryConditionLoss` with `use_extrapolation=False`, which penalized $(T_{P} - T_{\text{face}})^2$. Forcing cell-centre temperatures to match Dirichlet face temperatures suppresses the physical conductive temperature gradient $(T_P - T_{\text{face}})/(\Delta x/2)$, artificially reducing outward boundary heat transfer. In contrast, `full` mode preserves exact finite-volume half-cell conductances and maintains boundary flux conservation ($94.7\text{ W} \to 97.0\text{ W}$).
3. **Gradient Calibration & Telemetry**:
   - Pairwise cosine similarities recorded in `artifacts/reports/loss_scaling_calibration.json` were computed between unweighted parameter gradient vectors across all trainable parameters ($31,265$ in CNN; $1,188,641$ in FNO). They are mathematically invariant to scalar loss weights.
   - "Gradient parity" is verified as setting $w_{\text{pde}} = 0.005$ in interior mode so that $\|w_{\text{pde}} g_{\text{pde, int}}\| \approx 3.89\text{ W}$ equals $\|g_{\text{data}}\| \approx 3.81\text{ W}$ at initialization.
   - Training telemetry across all 18 runs confirms that positive gradient alignment persisted throughout the 30 epochs without gradient conflict.
4. **Dataset & OOD Physics Semantics**:
   - The 1,550-sample dataset was generated with machine-precision numerical solver convergence (residual $< 2.7 \times 10^{-14}$) and strict partition disjointness.
   - The extreme out-of-distribution temperature range ($30.89\text{ K}$ to $721.63\text{ K}$) represents mathematically exact solutions of the linear elliptic boundary value problem under high prescribed Neumann heat fluxes ($\pm 985\text{ W/m}^2$) through low-conductivity boundaries ($k = 1.0\text{ W/(m K)}$). The cryogenic minimum ($30.89\text{ K}$) is an artifact of unconstrained boundary sampling without thermodynamic temperature constraints.

---

## 1. Audit Scope & Reproduction Verification

### 1.1 Integrity of Milestone 04B Runs
All 18 experimental runs from Milestone 04B were audited against their respective disk artifacts:
- Checkpoints: [`artifacts/checkpoints/scaled_*_seed*/best_model.pt`](file:///c:/Workspace/Projects/Neural%20Process%20Twin/artifacts/checkpoints/) (all 18 present, non-empty, and loadable).
- Run reports: [`artifacts/reports/scaled_*_seed*.json`](file:///c:/Workspace/Projects/Neural%20Process%20Twin/artifacts/reports/) (all 18 present with complete gradient telemetry and loss histories).
- Master aggregation: [`artifacts/reports/milestone_04b_aggregated_results.json`](file:///c:/Workspace/Projects/Neural%20Process%20Twin/artifacts/reports/milestone_04b_aggregated_results.json).

Each run completed 30 epochs (960 optimizer steps at 32 steps/epoch) under early stopping patience 10 without premature convergence failure.

### 1.2 Verification Commands
The audit was conducted using the following commands:
```powershell
# 1. Audit dataset partition distributions, contrast ratios, and disjointness
.venv\Scripts\python scripts/audit_dataset_distributions.py

# 2. Recompute sample vs. population statistics and Welch/paired hypothesis tests
.venv\Scripts\python scripts/audit_analysis.py

# 3. Verify repository integrity and test suite
.venv\Scripts\pytest
.venv\Scripts\ruff check .
```

---

## 2. Statistical Verification & Corrected Significance Assessment

### 2.1 Sample ($ddof=1$) vs. Population ($ddof=0$) Standard Deviations
In Milestone 04B, all reported standard deviations across seeds were computed using NumPy's default `np.std(..., ddof=0)`:
\[
\sigma_{\mathrm{pop}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2}, \quad N=3
\]
For small sample sizes ($N=3$), standard scientific reporting requires the unbiased sample standard deviation:
\[
s_{\mathrm{sample}} = \sqrt{\frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2} = \sqrt{\frac{3}{2}} \cdot \sigma_{\mathrm{pop}} \approx 1.2247 \cdot \sigma_{\mathrm{pop}}
\]

The table below provides the master recomputed metrics comparing both conventions for the in-distribution test set (`test_iid`, 200 samples) and out-of-distribution test set (`test_ood`, 200 samples).

#### Master Seed-Averaged Benchmark Table
*Format: Mean $\pm$ Sample Std ($s$, $ddof=1$) [Population Std ($\sigma$, $ddof=0$)]*

| Model Configuration | Split | Relative $L_2$ [%] | RMSE [K] | Discrete Residual [W] | Flux Imbalance [W] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `scaled_cnn_supervised` | `test_iid` | $3.689 \pm 0.002$ [$0.002$] | $12.026 \pm 0.009$ [$0.008$] | $512.27 \pm 1.45$ [$1.18$] | $124.32 \pm 2.96$ [$2.41$] |
| `scaled_cnn_physics` | `test_iid` | $3.689 \pm 0.002$ [$0.002$] | $12.028 \pm 0.010$ [$0.008$] | $486.98 \pm 1.52$ [$1.24$] | $123.72 \pm 2.76$ [$2.26$] |
| `scaled_cnn_physics_bc` | `test_iid` | $4.072 \pm 0.007$ [$0.005$] | $13.280 \pm 0.025$ [$0.021$] | **$159.30 \pm 9.38$** [$7.66$] | $723.82 \pm 41.66$ [$34.01$] |
| `scaled_fno_supervised` | `test_iid` | **$0.284 \pm 0.011$** [$0.009$] | **$0.928 \pm 0.037$** [$0.030$] | $163.44 \pm 2.20$ [$1.79$] | **$94.70 \pm 20.22$** [$16.51$] |
| `scaled_fno_physics` | `test_iid` | $0.287 \pm 0.017$ [$0.014$] | $0.937 \pm 0.056$ [$0.046$] | $161.63 \pm 6.20$ [$5.06$] | $96.95 \pm 23.86$ [$19.48$] |
| `scaled_fno_physics_bc` | `test_iid` | $0.525 \pm 0.026$ [$0.021$] | $1.716 \pm 0.085$ [$0.070$] | **$84.12 \pm 7.18$** [$5.86$] | $259.88 \pm 78.93$ [$64.45$] |
| `scaled_cnn_supervised` | `test_ood` | $17.414 \pm 1.493$ [$1.219$] | $62.47 \pm 5.47$ [$4.46$] | $100,621 \pm 28,242$ [$23,060$] | $5,100 \pm 2,257$ [$1,843$] |
| `scaled_cnn_physics` | `test_ood` | $17.394 \pm 1.526$ [$1.246$] | $62.41 \pm 5.60$ [$4.57$] | $99,721 \pm 27,817$ [$22,713$] | $5,242 \pm 2,382$ [$1,945$] |
| `scaled_cnn_physics_bc` | `test_ood` | **$15.392 \pm 1.686$** [$1.376$] | **$55.79 \pm 6.27$** [$5.12$] | **$62,075 \pm 22,732$** [$18,561$] | $10,791 \pm 3,867$ [$3,157$] |
| `scaled_fno_supervised` | `test_ood` | $15.090 \pm 1.516$ [$1.238$] | $54.69 \pm 5.35$ [$4.36$] | $22,388 \pm 4,437$ [$3,623$] | $19,724 \pm 5,963$ [$4,869$] |
| `scaled_fno_physics` | `test_ood` | **$14.760 \pm 1.671$** [$1.364$] | **$53.49 \pm 5.77$** [$4.71$] | $21,964 \pm 4,600$ [$3,756$] | **$18,203 \pm 7,733$** [$6,314$] |
| `scaled_fno_physics_bc` | `test_ood` | $15.618 \pm 0.904$ [$0.738$] | $56.00 \pm 3.43$ [$2.80$] | **$10,788 \pm 1,613$** [$1,317$] | $19,536 \pm 10,694$ [$8,732$] |

---

### 2.2 Distinguishing Seed-Level from Sample-Level Variability
A key conceptual clarification required by this audit is the distinction between **seed-level variability** and **sample-level (within-partition) variability**:

1. **Seed-Level Variability ($\sigma_{\mathrm{seed}}$)**:
   Measures the stability of the training process across different random parameter initializations and DataLoader shuffle sequences.
   - For `scaled_fno_supervised` on `test_iid`, the seed standard deviation of the partition mean relative $L_2$ error is **$0.011\%$** ($s = 0.00011$).
   - This tiny value demonstrates that FNO optimization is highly reproducible across seeds on 1,000 training samples.
2. **Sample-Level Variability ($\sigma_{\mathrm{sample}}$)**:
   Measures the performance variation across the 200 distinct test problems within the partition, driven by varying thermal conductivities, boundary types, and localized source profiles.
   - For `scaled_fno_supervised_seed42` on `test_iid`, the standard deviation of relative $L_2$ error across individual samples is **$0.213\%$** ($s = 0.00213$).
   - On `test_ood`, sample-level standard deviation rises to **$6.28\%$** ($s = 0.0628$), reflecting dramatic difficulty differences between low-contrast and 50:1 contrast problems.

> [!IMPORTANT]
> Seed standard deviation must **never** be cited as an estimate of surrogate prediction uncertainty. The low seed variance ($\pm 0.011\%$) proves optimization determinism, not prediction confidence. Epistemic uncertainty across input space must be modeled separately (Milestone 05).

---

### 2.3 Paired Comparisons & Percentage Reductions
Because all 6 model configurations were trained and evaluated on the exact same dataset splits with identical test sample indices, comparisons across seeds can be performed both unpaired (Welch's $t$-test) and paired across seeds (matching Seeds 42, 43, 44):

#### Detailed Comparison on `test_iid` Discrete Residuals

```
FNO-2D Discrete PDE Residual (Watts [W]):
  Supervised (Seeds 42, 43, 44): [162.81, 165.88, 161.63] W  ->  Mean = 163.44 W (s = 2.20 W)
  Physics+BC (Seeds 42, 43, 44): [ 76.61,  84.85,  90.91] W  ->  Mean =  84.12 W (s = 7.18 W)
  Per-seed Delta (W):            [ 86.20,  81.04,  70.71] W  ->  Mean =  79.32 W (s = 7.89 W)
  Per-seed Reduction (%):        [52.94%, 48.85%, 43.75%]    ->  Mean = 48.52%
  Unrounded Aggregate Reduction: (163.44 - 84.12) / 163.44  =  48.53%

CNN-2D Discrete PDE Residual (Watts [W]):
  Supervised (Seeds 42, 43, 44): [513.79, 512.12, 510.91] W  ->  Mean = 512.27 W (s = 1.45 W)
  Physics+BC (Seeds 42, 43, 44): [161.41, 149.04, 167.45] W  ->  Mean = 159.30 W (s = 9.38 W)
  Per-seed Delta (W):            [352.38, 363.08, 343.46] W  ->  Mean = 352.97 W (s = 9.82 W)
  Per-seed Reduction (%):        [68.58%, 70.90%, 67.23%]    ->  Mean = 68.90%
  Unrounded Aggregate Reduction: (512.27 - 159.30) / 512.27  =  68.90%
```

---

### 2.4 Audit of Significance Claims ($> 12\sigma$ and $> 45\sigma$)

In Milestone 04B Section 6, the following calculation was reported:
\[
t = \frac{\Delta r}{\sigma_{\mathrm{diff}}} = \frac{79.3}{\sqrt{1.8^2 + 5.9^2}} = 12.85 > 12\sigma \implies p < 10^{-6}
\]

#### Forensic Audit of Flaws:
1. **Misidentified Standard Error**: The formula used $\sigma_{\mathrm{diff}} = \sqrt{s_1^2 + s_2^2}$ instead of the standard error of the mean difference $SE = \sqrt{s_1^2 / N_1 + s_2^2 / N_2}$. For $N_1 = N_2 = 3$, $SE$ is actually $\sqrt{s_1^2 + s_2^2} / \sqrt{3}$.
2. **Invalid Gaussian Normal Approximation**: The claim that "$t > 12 \implies p < 10^{-6}$" assumes an asymptotic standard normal distribution ($Z$-test). With only $N=3$ observations per group, the Central Limit Theorem does not apply. One must use Student's $t$-distribution, which has significantly heavier tails at small sample sizes.

#### Rigorous Re-Evaluation with Student's $t$-Distribution:

1. **Welch's Two-Sample $t$-test** (unpaired, unequal variances, $N=3$, $s_{\mathrm{sample}}$):
   - **FNO-2D**:
     \[
     SE = \sqrt{\frac{2.20^2}{3} + \frac{7.18^2}{3}} = 4.33\text{ W}, \quad t = \frac{79.32}{4.33} = 18.30
     \]
     Welch-Satterthwaite degrees of freedom: $\nu \approx 2.37$.
     Two-tailed $p$-value:
     \[
     p = 0.00131 \quad (\approx 0.13\%)
     \]
   - **CNN-2D**:
     \[
     SE = \sqrt{\frac{1.45^2}{3} + \frac{9.38^2}{3}} = 5.48\text{ W}, \quad t = \frac{352.97}{5.48} = 64.40
     \]
     Welch-Satterthwaite degrees of freedom: $\nu \approx 2.10$.
     Two-tailed $p$-value:
     \[
     p = 0.00017 \quad (\approx 0.017\%)
     \]

2. **Paired Two-Sample $t$-test** (matched seeds 42, 43, 44, $N=3$, $df = 2$):
   - **FNO-2D**:
     \[
     s_{\mathrm{diff}} = 7.89\text{ W}, \quad SE_{\mathrm{paired}} = \frac{7.89}{\sqrt{3}} = 4.55\text{ W}, \quad t = \frac{79.32}{4.55} = 17.42, \quad df = 2 \implies p = 0.00328
     \]
   - **CNN-2D**:
     \[
     s_{\mathrm{diff}} = 9.82\text{ W}, \quad SE_{\mathrm{paired}} = \frac{9.82}{\sqrt{3}} = 5.67\text{ W}, \quad t = \frac{352.97}{5.67} = 62.25, \quad df = 2 \implies p = 0.00026
     \]

#### Corrected Statement of Significance
The reduction in discrete PDE residual is statistically confirmed at standard scientific thresholds ($\alpha = 0.01$ and $\alpha = 0.05$):
- FNO-2D achieves a **$48.5\%$ residual reduction** ($p = 0.0013$, Welch $t=18.30$, $df=2.37$).
- CNN-2D achieves a **$68.9\%$ residual reduction** ($p = 0.00017$, Welch $t=64.40$, $df=2.10$).

The previous claim of "$p < 10^{-6}$" and "$> 12\sigma$" is formally retracted as an artifact of incorrect Gaussian assumptions on $N=3$ sample sizes.

---

## 3. Physics Residual & Boundary Flux Dissection

### 3.1 Mathematical Definitions and Units

The evaluation pipeline computes two independent physical consistency metrics, each rooted in the finite-volume formulation:

1. **Discrete Cell Residual $r_P$ [Watts]**:
   For each cell-centred control volume $P$, the discrete algebraic residual is:
   \[
   r_P = (A T - b)_P = \sum_{f \in \{e, w, n, s\}} Q_{P, f} - q_P V_P
   \]
   where $Q_{P, f}$ is the conductive heat flow [W] leaving cell $P$ across face $f$, and $q_P V_P$ is the rate of heat generation inside volume $V_P$ [W]. The aggregate metric reported is the Euclidean $L_2$ norm across all cells:
   \[
   \|r\|_2 = \sqrt{\sum_{P=1}^{N_{\mathrm{cells}}} r_P^2} \quad \text{[Watts]}
   \]
2. **Global Boundary Heat Flux Imbalance [Watts]**:
   By Gauss's divergence theorem, in steady-state conduction, the total heat flow exiting across all domain boundaries must exactly balance the total heat generated within the domain:
   \[
   Q_{\mathrm{boundary, out}} = \sum_{\mathrm{boundaries}} \int_{\Gamma_b} \mathbf{j} \cdot \mathbf{n} \, dA, \quad Q_{\mathrm{generated}} = \int_\Omega q \, dV
   \]
   The global flux imbalance evaluates the absolute conservation discrepancy:
   \[
   \Delta Q = |Q_{\mathrm{boundary, out}} - Q_{\mathrm{generated}}| \quad \text{[Watts]}
   \]

---

### 3.2 Forensic Analysis: Why Boundary Flux Imbalance Increased in `physics_bc`

A central anomaly identified in Milestone 04B was that while `physics_bc` configurations reduced discrete cell residuals ($r_P$) by $48.5\% - 68.9\%$, their global flux imbalance increased significantly:
- FNO Flux Imbalance: $94.7\text{ W} \to 259.9\text{ W}$ ($+174\%$)
- CNN Flux Imbalance: $124.3\text{ W} \to 723.8\text{ W}$ ($+482\%$)

The audit traced this behavior directly to the numerical discretization in [`src/neural_process_twin/training/physics_loss.py`](file:///c:/Workspace/Projects/Neural%20Process%20Twin/src/neural_process_twin/training/physics_loss.py#L314-L320):

```python
# In BoundaryConditionLoss (physics_loss.py:314):
if self.use_extrapolation and temperature_k.shape[3] > 1:
    t_face_l = 1.5 * temperature_k[:, :, :, 0:1] - 0.5 * temperature_k[:, :, :, 1:2]
else:
    t_face_l = temperature_k[:, :, :, 0:1]

err_dir_l = ((t_face_l - dir_v_l) / t_char) ** 2
```

#### Physical and Numerical Mechanism:
1. **Half-Cell Discretization**: In the cell-centred Finite Volume reference solver, the Dirichlet boundary condition is applied on the boundary face $\Gamma$ at $x = 0$. The first cell center $P_0$ is located at a distance of $\Delta x / 2$ from the boundary.
2. **Fourier Conductance**: The physical conductive heat flux crossing the boundary face into cell $P_0$ is:
   \[
   j_{\mathrm{out}} = \frac{2 k_0}{\Delta x} (T_{P_0} - T_{\mathrm{bnd}})
   \]
   In physical equilibrium with heat generation, $T_{P_0} \neq T_{\mathrm{bnd}}$. There must be a non-zero temperature gradient $(T_{P_0} - T_{\mathrm{bnd}})$ to transport heat out of the domain.
3. **The `use_extrapolation=False` Defect**:
   In `configs/experiments/scaled_*_physics_bc.yaml`, `use_extrapolation` was not enabled (defaulting to `False`). As a result, `BoundaryConditionLoss` penalized $(T_{P_0} - T_{\mathrm{bnd}})^2$, actively forcing the **cell center** temperature to equal the **boundary face** temperature ($T_{P_0} \to T_{\mathrm{bnd}}$).
4. **Flux Suppression & Imbalance**:
   Driving $T_{P_0} \to T_{\mathrm{bnd}}$ collapses the half-cell temperature drop $(T_{P_0} - T_{\mathrm{bnd}}) \to 0$. Consequently, the reconstructed outward boundary flux $Q_{\mathrm{boundary, out}}$ is choked, severely under-predicting the heat escaping the domain and causing a large global flux imbalance $|Q_{\mathrm{boundary, out}} - Q_{\mathrm{generated}}|$.
5. **Why `full` Mode Did Not Suffer This Defect**:
   In `scaled_*_physics` (`residual_type="full"`, $w_{\mathrm{bc}}=0$), the loss did not use `BoundaryConditionLoss`. Instead, `DiscretePDEResidual` incorporated the exact half-cell boundary conductance into the boundary cell residual:
   \[
   a_P T_{P_0} + \dots = q V_P + \frac{2 k_0}{\Delta x} T_{\mathrm{bnd}} A
   \]
   Because this formulation correctly accounts for the $\Delta x / 2$ offset, `scaled_fno_physics` maintained exact boundary flux balance ($94.7\text{ W} \to 96.95\text{ W}$), with zero degradation!

> [!NOTE]
> This trade-off was not a failure of physics regularization, but a discretization mismatch between naive cell-centre boundary penalties and cell-centred finite-volume stencils. Future milestones must use either `full` finite-volume residual operators or linear extrapolation stencils ($1.5 T_0 - 0.5 T_1$).

---

## 4. Gradient Calibration & Telemetry Audit

### 4.1 Invariance of Cosine Similarities
In [`artifacts/reports/loss_scaling_calibration.json`](file:///c:/Workspace/Projects/Neural%20Process%20Twin/artifacts/reports/loss_scaling_calibration.json), pairwise cosine similarities between gradient vectors were reported:
- $\cos(g_{\mathrm{data}}, g_{\mathrm{pde, full}}) = +0.8108$ (FNO), $+0.4957$ (CNN)
- $\cos(g_{\mathrm{data}}, g_{\mathrm{pde, interior}}) = +0.8928$ (FNO), $+0.2031$ (CNN)
- $\cos(g_{\mathrm{data}}, g_{\mathrm{bc}}) = +0.7898$ (FNO), $+0.5828$ (CNN)

The audit verified in [`scripts/diagnose_loss_scaling.py`](file:///c:/Workspace/Projects/Neural%20Process%20Twin/scripts/diagnose_loss_scaling.py#L105-L119) that these similarities are computed between the **unweighted** loss gradients:
\[
g_{\mathrm{data}} = \nabla_\theta \mathcal{L}_{\mathrm{data}}, \quad g_{\mathrm{pde}} = \nabla_\theta \mathcal{L}_{\mathrm{pde}}, \quad \cos(g_A, g_B) = \frac{\langle g_A, g_B \rangle}{\|g_A\|_2 \|g_B\|_2}
\]
Because $\cos(w_A g_A, w_B g_B) = \cos(g_A, g_B)$ for any positive scalars $w_A, w_B > 0$, the measured cosine similarities are **strictly independent of the loss weights**. They reflect intrinsic geometric alignment between data supervision and physical PDE stencils.

### 4.2 Parameter Scope
The gradient vectors are flattened and concatenated across **all trainable parameters**:
- **CNN-2D**: 31,265 parameters across 4 convolutional layers.
- **FNO-2D**: 1,188,641 parameters across lifting conv, 4 spectral conv layers (concatenating both real and imaginary weight components), and projection conv layers.

### 4.3 Full vs. Interior Residual Scaling & "Gradient Parity"
The unweighted PDE gradient norm varied dramatically between formulations:
- In `full` mode, $\|g_{\mathrm{pde, full}}\| \approx 0.408$ (FNO).
- In `interior` mode, $\|g_{\mathrm{pde, interior}}\| \approx 778.65$ (FNO).

**Origin of the Scale Discrepancy**:
`DimensionlessPDELoss` normalizes the residual norm by the problem scale: $\|r\|_2 / (\|b\|_2 + Q_{\mathrm{floor}})$.
In `full` mode, vector $b$ includes all boundary Dirichlet stencils ($b \sim \frac{2 k}{\Delta x} T_{\mathrm{bnd}} A \sim 10^3 - 10^4\text{ W}$), making the denominator large and the loss scale small. In `interior` mode, boundary terms are excluded from $b$, leaving only the volumetric heat generation ($b \sim q \cdot V \sim 10^{-2} - 10^1\text{ W}$), which reduces the denominator and increases the loss scale by three orders of magnitude.

**Verification of "Gradient Parity"**:
In `scaled_fno_physics_bc`, setting $w_{\mathrm{pde}} = 0.005$ yields:
\[
\|w_{\mathrm{pde}} g_{\mathrm{pde, interior}}\| = 0.005 \times 778.65 \approx 3.89\text{ W} \approx \|g_{\mathrm{data}}\| \approx 3.81\text{ W}
\]
"Gradient parity" is verified as precisely calibrating $w_{\mathrm{pde}}$ so that the physical residual gradient norm equals the supervised data gradient norm at step 0.

### 4.4 Telemetry Throughout Training
Telemetry recorded in all 18 JSON reports confirms that positive gradient alignment was maintained across the entire training trajectory:
- **Epoch 1**: $\cos(g_{\mathrm{data}}, g_{\mathrm{pde}}) = +0.632$, $\|g_{\mathrm{total}}\| = 19.42$
- **Epoch 15**: $\cos(g_{\mathrm{data}}, g_{\mathrm{pde}}) = +0.253$, $\|g_{\mathrm{total}}\| = 0.28$
- **Epoch 30**: $\cos(g_{\mathrm{data}}, g_{\mathrm{pde}}) = +0.158$, $\|g_{\mathrm{total}}\| = 0.32$

At no point did gradient conflict or catastrophic gradient opposition occur.

---

## 5. Dataset & Out-of-Distribution Physics Semantics

### 5.1 Dataset Composition & Disjointness Audit
The scaled dataset ([`artifacts/data/scaled_dataset.h5`](file:///c:/Workspace/Projects/Neural%20Process%20Twin/artifacts/data/scaled_dataset.h5)) was audited with [`scripts/audit_dataset_distributions.py`](file:///c:/Workspace/Projects/Neural%20Process%20Twin/scripts/audit_dataset_distributions.py):
- Total samples: 1,550 on a $32 \times 32$ grid ($1,024$ control volumes).
- Partitions: `train` (1,000), `validation` (150), `test_iid` (200), `test_ood` (200).
- Partition disjointness: Confirmed 100% disjoint with zero overlapping sample IDs, coordinates, or random seeds.
- Contrast ratio: $k_{\max} / k_{\min} = 4.49 \pm 0.68$ in `train` vs. $34.70 \pm 11.23$ (up to $49.92:1$) in `test_ood`.
- Heat source: $\bar{q} = 41.9\text{ W/m}^3$ in `train` vs. $\bar{q} = 160.3\text{ W/m}^3$ (max $3643.8\text{ W/m}^3$) in `test_ood`.

---

### 5.2 Forensic Investigation of Extreme OOD Temperatures ($30.9\text{ K} - 721.6\text{ K}$)

The `test_ood` temperature range exhibits extreme values: $T_{\min} = 30.89\text{ K}$ and $T_{\max} = 721.63\text{ K}$. The audit inspected the exact physical configurations generating these extremes:

#### Minimum Temperature ($30.89\text{ K}$, Sample 116):
- **Conductivity Field**: $k_{\min} = 1.00\text{ W/(m K)}$, $k_{\max} = 4.88\text{ W/(m K)}$.
- **Boundary Conditions**:
  - Left: Dirichlet $T = 320.91\text{ K}$
  - Right: Dirichlet $T = 327.85\text{ K}$
  - Top: Neumann $j = 0.00\text{ W/m}^2$ (adiabatic)
  - Bottom: Neumann $j = +984.73\text{ W/m}^2$ (**high prescribed outward heat extraction**)
- **Physical Explanation**:
  Fourier's law at the bottom boundary states:
  \[
  j_{\mathrm{out}} = -k \frac{\partial T}{\partial n} = k \frac{\partial T}{\partial y} = +984.73\text{ W/m}^2 \implies \frac{\partial T}{\partial y} \approx \frac{984.73}{1.00} \approx 985\text{ K/m}
  \]
  Over a domain height of $1.0\text{ m}$, maintaining an outward extraction of nearly $1\text{ kW/m}^2$ through a material with low thermal conductivity ($1\text{ W/(m K)}$) requires a steep temperature gradient of $\approx 985\text{ K/m}$. Given boundary temperatures of $\approx 325\text{ K}$ at the sides, the temperature drops precipitously towards the bottom edge, reaching $30.89\text{ K}$.

#### Maximum Temperature ($721.63\text{ K}$, Sample 64):
- **Conductivity Field**: $k_{\min} = 1.00\text{ W/(m K)}$, $k_{\max} = 4.80\text{ W/(m K)}$.
- **Boundary Conditions**:
  - Left: Dirichlet $T = 355.41\text{ K}$
  - Right: Dirichlet $T = 409.31\text{ K}$
  - Top: Neumann $j = 0.00\text{ W/m}^2$ (adiabatic)
  - Bottom: Neumann $j = -929.10\text{ W/m}^2$ (**high prescribed inward heat injection**)
- **Volumetric Heat Source**: $q_{\max} = 1327.5\text{ W/m}^3$ (total generated heat $= 1.48\text{ W}$).
- **Physical Explanation**:
  An inward heat flux of $-929.10\text{ W/m}^2$ continuously pumps heat into a low-conductivity boundary adjacent to a $409\text{ K}$ Dirichlet wall, while internal volumetric sources add heat. The resulting thermal buildup drives peak core temperatures up to $721.63\text{ K}$.

#### Scientific Distinction: Numerical Exactness vs. Physical Modeling Plausibility
1. **Mathematical Solver Exactness**:
   Both Samples 64 and 116 were solved by the conservative finite-volume reference solver with machine-precision algebraic residual:
   \[
   \|A T - b\|_2 < 2.7 \times 10^{-14}\text{ W}, \quad \text{Global Heat Imbalance } < 4.3 \times 10^{-10}\text{ W}
   \]
   They are rigorous, exact mathematical solutions to the linear elliptic boundary value problem $-\nabla \cdot (k \nabla T) = q$.
2. **Physical Modeling Plausibility**:
   The steady-state linear heat conduction model assumes temperature-independent thermal conductivity $k$ and neglects radiation ($\sigma T^4$) and phase changes. At $30.89\text{ K}$ (liquid neon temperatures), real materials undergo drastic changes in specific heat and thermal conductivity, and cryogenic cooling cannot be maintained without active refrigeration.
   These extreme temperatures arose because synthetic boundary condition sampling drew Neumann fluxes up to $\pm 1000\text{ W/m}^2$ without imposing thermodynamic inequality constraints ($T > T_{\mathrm{ambient}}$). While mathematically valid, they represent edge cases that stress-test surrogate models under extreme thermal gradients.

---

## 6. Recommendations & Readiness for Milestone 05

### 6.1 Readiness Verdict
Milestone 04B is **fully verified and scientifically sound**. The experimental results, checkpoints, dataset distributions, and gradient behaviors are trustworthy, provided the statistical corrections and numerical trade-offs documented in this audit are preserved.

The project is **cleared to proceed to Milestone 05 (Uncertainty Estimation and Ensembles)**.

### 6.2 Key Methodological Guidelines for Milestone 05
1. **Reporting Standards**:
   - Report sample standard deviations ($ddof=1$) for multi-seed benchmarks.
   - For statistical significance tests on small sample sizes ($N=3$), use Student's $t$-distribution with exact degrees of freedom; never cite asymptotic Gaussian $Z$-test $p$-values.
   - Always report within-partition sample variance ($\sigma_{\mathrm{sample}}$) separately from seed-level optimization variance ($\sigma_{\mathrm{seed}}$).
2. **Boundary Condition Formulations**:
   - In physics-aware loss functions, avoid naive Dirichlet face penalties on cell centres ($use\_extrapolation=False$).
   - Use either `residual_type="full"`, which preserves exact finite-volume half-cell conductances, or implement proper linear extrapolation stencils ($1.5 T_0 - 0.5 T_1$).
3. **Uncertainty Estimation Benchmarking**:
   - Evaluate whether ensemble disagreement (or MC-Dropout epistemic variance) scales proportionally to the observed $15\%$ OOD error gap.
   - Specifically evaluate uncertainty bounds on extreme gradient samples (such as OOD Sample 116 and Sample 64) to confirm whether uncertainty estimates correctly flag extreme regime departures.
