# Scientific Specification: Two-Dimensional Steady-State Heat Conduction

## 1. Governing Physical Equation

The physical problem addressed in Phase 1 of **Neural Process Twin** is linear, steady-state thermal conduction in a two-dimensional domain with heterogeneous thermal conductivity and volumetric heat generation.

The strong form of the governing partial differential equation (PDE) is:

\[
-\nabla \cdot \left( k(x, y) \nabla T(x, y) \right) = q(x, y) \quad \text{for } (x, y) \in \Omega
\]

Expanding the divergence and gradient operators in 2D Cartesian coordinates:

\[
-\frac{\partial}{\partial x}\left( k(x, y) \frac{\partial T(x, y)}{\partial x} \right) - \frac{\partial}{\partial y}\left( k(x, y) \frac{\partial T(x, y)}{\partial y} \right) = q(x, y)
\]

where Fourier's Law relates the conductive heat flux density vector $\mathbf{j}(x, y)$ to the temperature gradient:

\[
\mathbf{j}(x, y) = -k(x, y) \nabla T(x, y) = -k(x, y) \begin{bmatrix} \frac{\partial T}{\partial x} \\ \frac{\partial T}{\partial y} \end{bmatrix}
\]

Hence, the governing equation represents steady-state conservation of thermal energy:

\[
\nabla \cdot \mathbf{j}(x, y) = q(x, y)
\]

---

## 2. Physical Variables, Dimensions, and Units

All physical quantities in Neural Process Twin are specified in SI units unless explicitly non-dimensionalized:

| Symbol | Quantity | Dimension | SI Unit | Physical Constraints |
| :--- | :--- | :--- | :--- | :--- |
| $(x, y)$ | Spatial coordinates | $[L]$ | $\text{m}$ (meter) | $(x, y) \in [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$ |
| $L_z$ | Out-of-plane model thickness | $[L]$ | $\text{m}$ (meter) | $L_z > 0$, defaults to $1.0\text{ m}$ (unit depth) |
| $T(x, y)$ | Temperature field | $[\Theta]$ | $\text{K}$ (Kelvin) | Strictly positive absolute temperature ($T > 0$) |
| $k(x, y)$ | Thermal conductivity field | $[M L T^{-3} \Theta^{-1}]$ | $\text{W}/(\text{m}\cdot\text{K})$ | Strictly positive: $0 < k_{\min} \le k(x, y) \le k_{\max} < \infty$ |
| $q(x, y)$ | Volumetric heat generation | $[M L^{-1} T^{-3}]$ | $\text{W}/\text{m}^3$ | Finite: $q \ge 0$ (sources) or $q < 0$ (sinks) |
| $\mathbf{j}(x, y)$ | Conductive heat flux density | $[M T^{-3}]$ | $\text{W}/\text{m}^2$ | Vector pointing down the thermal gradient |
| $g_{\mathrm{out}}$ | Outward boundary normal flux | $[M T^{-3}]$ | $\text{W}/\text{m}^2$ | $g_{\mathrm{out}} = \mathbf{j} \cdot \mathbf{n} = -k \frac{\partial T}{\partial n}$ |

---

## 3. Initial Assumptions and Scope

The baseline physical formulation is constrained by the following explicit assumptions:

1. **Two-Dimensional Geometry**: The spatial domain is a planar rectangle $\Omega = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}] \subset \mathbb{R}^2$ with boundary $\partial \Omega = \Gamma_D \cup \Gamma_N$, where $\Gamma_D \cap \Gamma_N = \emptyset$.
2. **Thickness Convention**: Calculations assume out-of-plane thickness $L_z$ (defaulting to unit depth $L_z = 1.0\text{ m}$). Face areas are $A_x = \Delta y L_z$ and $A_y = \Delta x L_z$; cell control volume is $V_P = \Delta x \Delta y L_z$.
3. **Steady-State Regime**: Thermal transients have decayed ($\partial T / \partial t = 0$). Heat capacity and mass density do not enter the steady-state equation.
4. **Isotropic Material Response**: Conductivity $k(x, y)$ is spatially heterogeneous but isotropic at each point (scalar field, not a full tensor).
5. **No Internal Advection or Radiation**: Heat transport inside $\Omega$ is purely conductive. Radiative and convective exchanges occur strictly on the boundary.
6. **Linear Material Response**: Conductivity $k$ does not depend on temperature $T$ ($k = k(x,y)$; non-linear temperature-dependent conductivity $k(T)$ is deferred to future phases).

---

## 4. Thermal Conductivity Distributions

The simulator models heterogeneous media by generating parameterized spatial conductivity distributions $k(x, y)$. To guarantee well-posedness, uniform ellipticity must hold:

\[
\exists \, k_{\min}, k_{\max} \in (0, \infty) \quad \text{such that} \quad k_{\min} \le k(x, y) \le k_{\max} \quad \forall (x, y) \in \Omega
\]

Under this condition, the Lax-Milgram theorem guarantees the existence of a unique weak solution $T \in H^1(\Omega)$ for non-empty Dirichlet boundary conditions.

### Supported Distribution Families:
1. **Constant Conductivity**: Homogeneous reference media ($k(x, y) = k_0$).
2. **Piecewise Constant Inclusions**: Sharp material interfaces representing composite layers, thermal barrier coatings, or embedded inclusions.
3. **Smooth Continuous Heterogeneity**: Spatially varying smooth fields (e.g. sinusoidal or Gaussian random fields with continuous derivatives).

---

## 5. Boundary Conditions

The domain boundary $\partial \Omega$ is decomposed into Dirichlet ($\Gamma_D$) and Neumann ($\Gamma_N$) segments:

1. **Dirichlet Boundary Conditions (Prescribed Temperature)**:
   \[
   T(x, y) = T_b(x, y) \quad \text{on } \Gamma_D
   \]
   Prescribes temperature along isothermal contacts or heat sinks.

2. **Neumann Boundary Conditions (Prescribed Outward Conductive Heat Flux)**:
   \[
   g_{\mathrm{out}}(x, y) = \mathbf{j}(x, y) \cdot \hat{\mathbf{n}} = -k(x, y) \left( \nabla T(x, y) \cdot \hat{\mathbf{n}} \right) \quad \text{on } \Gamma_N
   \]
   where $\hat{\mathbf{n}}$ is the outward unit normal vector.
   - **Positive flux ($g_{\mathrm{out}} > 0$)**: Conductive heat leaves the domain across the boundary.
   - **Negative flux ($g_{\mathrm{out}} < 0$)**: Conductive heat enters the domain from the boundary.
   - **Adiabatic wall ($g_{\mathrm{out}} = 0$)**: Insulated boundary with zero normal conductive flux.

3. **Well-Posedness and Pure-Neumann Singularity**:
   - If $|\Gamma_D| > 0$, the Poincaré-Friedrichs inequality guarantees coercivity, yielding a unique solution $\mathbf{T}$. The assembled sparse stiffness matrix $A$ is symmetric positive-definite (SPD).
   - If $|\Gamma_D| = 0$ (pure Neumann boundary problem), the solution is determined only up to an arbitrary additive constant (nullspace dimension 1, spanned by the constant vector $\mathbf{1}$). Solvability requires exact global energy balance:
     \[
     \iint_\Omega q(x, y) \, dx dy = \oint_{\partial \Omega} g_{\mathrm{out}} \, ds
     \]
   The solver detects pure Neumann boundaries and raises `PureNeumannSingularError` unless an explicit reference temperature (gauge constraint) is pinned at a designated cell.

---

## 6. Numerical Reference Solver Implementation

### 6.1 Cell-Centred Finite-Volume Discretization

The domain $\Omega = [x_{\min}, x_{\max}] \times [y_{\min}, y_{\max}]$ is discretized into a uniform Cartesian grid of $nx \times ny$ control volumes (cells):
- Cell dimensions: $\Delta x = (x_{\max} - x_{\min}) / nx$, $\Delta y = (y_{\max} - y_{\min}) / ny$.
- Cell-centre coordinates:
  \[
  x_i = x_{\min} + (i + 0.5) \Delta x, \quad i \in \{0, \dots, nx - 1\}
  \]
  \[
  y_j = y_{\min} + (j + 0.5) \Delta y, \quad j \in \{0, \dots, ny - 1\}
  \]
- 2D array indexing convention: `array[j, i]` where row index $j$ maps to $y_j$ and column index $i$ maps to $x_i$, with shape `(ny, nx)`. Linearized index $p = j \cdot nx + i$.

Integrating $-\nabla \cdot (k \nabla T) = q$ over control volume $V_P = \Delta x \Delta y L_z$ and applying the divergence theorem:

\[
\oint_{\partial V_P} (\mathbf{j} \cdot \hat{\mathbf{n}}) \, dA = \iiint_{V_P} q \, dV \implies Q_e + Q_w + Q_n + Q_s = q_P V_P
\]

where $Q_f$ represents the net conductive heat flow rate leaving control volume $P$ across face $f$.

### 6.2 Internal Face Conductance and Harmonic Conductivity Averaging

For internal face $e$ between cell $P = (i, j)$ and East neighbor $E = (i+1, j)$:
- Distance from $P$ to face $e$ is $\Delta x / 2$ with conductivity $k_P$.
- Distance from face $e$ to $E$ is $\Delta x / 2$ with conductivity $k_E$.
- Heat flux continuity across face $e$ requires thermal resistances in series:
  \[
  R_{\text{total}} = R_{P \to e} + R_{e \to E} = \frac{\Delta x / 2}{k_P A_x} + \frac{\Delta x / 2}{k_E A_x} = \frac{\Delta x}{A_x} \left( \frac{k_P + k_E}{2 k_P k_E} \right)
  \]
  Equating to $R_{\text{total}} = \frac{\Delta x}{k_e A_x}$ yields the **harmonic mean**:
  \[
  k_e = \frac{2 k_P k_E}{k_P + k_E}
  \]
- The internal face thermal conductance is:
  \[
  G_e = \frac{k_e A_x}{\Delta x} = \frac{k_e \Delta y L_z}{\Delta x}
  \]
  Similarly, for horizontal internal face $n$ between $(i, j)$ and $(i, j+1)$:
  \[
  G_n = \frac{k_n A_y}{\Delta y} = \frac{k_n \Delta x L_z}{\Delta y}, \quad k_n = \frac{2 k_P k_N}{k_P + k_N}
  \]

#### Limitations of Harmonic Averaging:
- Harmonic averaging is exact for 1D flux continuity perpendicular to planar material interfaces aligned with the Cartesian grid lines.
- For diagonal or curved interfaces that cut through cells at arbitrary angles, staircase Cartesian grids introduce geometric discretization error; immersed boundary or unstructured methods are required for non-aligned interfaces.

### 6.3 Boundary Conductances and Stencil Closure

- **Dirichlet Boundary**:
  Face temperature is prescribed at the domain boundary: $T_b$. The distance from the boundary face to the nearest cell centre is $\Delta x / 2$ (for East/West) or $\Delta y / 2$ (for North/South).
  - Left boundary ($i = 0$): $G_{\text{left}, j} = \frac{k[j, 0] A_x}{\Delta x / 2} = \frac{2 k[j, 0] \Delta y L_z}{\Delta x}$.
  - Heat flow leaving cell $P$: $Q_{\text{left}} = G_{\text{left}, j} (T_{0, j} - T_{\text{left}, j})$.
  - Stencil contribution: $A_{P, P} \mathrel{+}= G_{\text{left}, j}$, $b_P \mathrel{+}= G_{\text{left}, j} T_{\text{left}, j}$.
- **Neumann Boundary**:
  Outward heat flux is prescribed: $g_{\mathrm{out}}$ [W/m²].
  - Heat flow leaving cell $P$: $Q_b = g_{\mathrm{out}} A_{\text{face}}$.
  - Stencil contribution: $b_P \mathrel{+}= - g_{\mathrm{out}} A_{\text{face}}$. No contribution to matrix diagonal $A_{P, P}$.

### 6.4 Sparse Linear System Assembly

The discrete system forms a sparse linear matrix equation:
\[
A \mathbf{T} = \mathbf{b}
\]
where $A \in \mathbb{R}^{N \times N}$ is assembled in Compressed Sparse Row (CSR) format via SciPy (`scipy.sparse.csr_matrix`), and solved via direct sparse LU factorization (`scipy.sparse.linalg.spsolve`).

---

## 7. Verification via Method of Manufactured Solutions (MMS)

### 7.1 Analytical Case 1: Constant Conductivity MMS
- Domain: $\Omega = [0, 1] \times [0, 1]$, $k_0 = 1.5\text{ W/(m K)}$, $T_0 = 300\text{ K}$, $A = 25\text{ K}$.
- Manufactured solution:
  \[
  T^*(x, y) = T_0 + A \sin\left(\frac{\pi (x - x_{\min})}{L_x}\right) \sin\left(\frac{\pi (y - y_{\min})}{L_y}\right)
  \]
- Exact source:
  \[
  q^*(x, y) = -k_0 \nabla^2 T^* = k_0 \pi^2 \left( \frac{1}{L_x^2} + \frac{1}{L_y^2} \right) A \sin\left(\frac{\pi (x - x_{\min})}{L_x}\right) \sin\left(\frac{\pi (y - y_{\min})}{L_y}\right)
  \]
- Boundary conditions: Homogeneous Dirichlet $T_b = T_0 = 300\text{ K}$ on all four boundaries.

### 7.2 Analytical Case 2: Spatially Varying Conductivity MMS
- Domain: $\Omega = [0, 1] \times [0, 1]$, $k(x, y) = k_0 + k_1 \cos\left(\frac{\pi x}{L_x}\right) \cos\left(\frac{\pi y}{L_y}\right)$ ($k_0 = 2.0, k_1 = 0.5$).
- Exact source:
  \[
  q^*(x, y) = -\nabla \cdot (k \nabla T^*) = - (\nabla k \cdot \nabla T^* + k \nabla^2 T^*)
  \]
  \[
  = k(x, y) \pi^2 \left( \frac{1}{L_x^2} + \frac{1}{L_y^2} \right) A \sin\left(\frac{\pi x}{L_x}\right) \sin\left(\frac{\pi y}{L_y}\right) + A k_1 \frac{\pi^2}{4} \left( \frac{1}{L_x^2} + \frac{1}{L_y^2} \right) \sin\left(\frac{2\pi x}{L_x}\right) \sin\left(\frac{2\pi y}{L_y}\right)
  \]

### 7.3 Measured Grid Refinement Convergence

Grid refinement studies conducted in Milestone 01 produced the following exact empirical results:

| Grid ($nx \times ny$) | Constant $k$ Rel $L_2$ Error | Constant $k$ Obs. Order $p$ | Variable $k$ Rel $L_2$ Error | Variable $k$ Obs. Order $p$ |
| :---: | :---: | :---: | :---: | :---: |
| $16 \times 16$ | $1.2969 \times 10^{-4}$ | — | $1.3030 \times 10^{-4}$ | — |
| $32 \times 32$ | $3.2379 \times 10^{-5}$ | **2.0020** | $3.2528 \times 10^{-5}$ | **2.0021** |
| $64 \times 64$ | $8.0919 \times 10^{-6}$ | **2.0005** | $8.1292 \times 10^{-6}$ | **2.0005** |
| $128 \times 128$ | $2.0228 \times 10^{-6}$ | **2.0001** | $2.0321 \times 10^{-6}$ | **2.0001** |

The observed order of accuracy converges asymptotically to $p = 2.0001$, rigorously verifying second-order spatial accuracy ($\mathcal{O}(h^2)$) for both homogeneous and heterogeneous media.

---

## 8. Known Numerical Limitations

1. **Orthogonal Grid Alignment**: The finite-volume scheme assumes a regular Cartesian grid aligned with the coordinate axes. Curved domains or non-orthogonal physical geometries require coordinate transformations or unstructured meshes.
2. **Interface Alignment**: Harmonic averaging is strictly second-order when material discontinuities align with cell faces. Slanted or curved material interfaces cutting across cells require volume-fraction weighted homogenization.
3. **Pure Neumann Conditioning**: Pure Neumann boundary problems are mathematically singular without a reference temperature. Users must specify at least one Dirichlet boundary or provide a gauge pin.

---

## 9. Synthetic Dataset Generation, Ingestion, and OOD Protocol

### 9.1 Gaussian Random Field (GRF) Formulation
Stationary isotropic Gaussian random fields $\xi(\mathbf{x}) \sim \mathcal{GP}(0, C)$ on $\Omega$ are synthesized using the continuous 2D Fourier spectral representation:
\[
\xi(\mathbf{x}) = \int_{\mathbb{R}^2} \sqrt{S(\mathbf{k})} e^{i \mathbf{k} \cdot \mathbf{x}} d\tilde{W}(\mathbf{k})
\]
where $S(\mathbf{k})$ is the 2D power spectral density (PSD) associated with covariance kernel $C(r)$ by the Wiener-Khinchin theorem:
- **Gaussian (Squared Exponential) Kernel**:
  \[
  C(r) = \sigma^2 \exp\left(-\frac{r^2}{2\ell^2}\right) \iff S(\mathbf{k}) = 2\pi \sigma^2 \ell^2 \exp\left(-\frac{\|\mathbf{k}\|^2 \ell^2}{2}\right)
  \]
- **Matérn Kernel ($\nu > 0$)**:
  \[
  C(r) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}r}{\ell}\right) \iff S(\mathbf{k}) = 4\pi \sigma^2 \nu \left(\frac{2\nu}{\ell^2}\right)^\nu \left(\frac{2\nu}{\ell^2} + \|\mathbf{k}\|^2\right)^{-(\nu + 1)}
  \]
Discretized on a uniform $N_y \times N_x$ Cartesian grid via 2D inverse FFT in $\mathcal{O}(N \log N)$ complexity, eliminating the prohibitive $\mathcal{O}(N^3)$ cost of decomposing dense $N \times N$ spatial covariance matrices.

### 9.2 Positivity-Preserving Transformations
Strict physical ellipticity requires $k(x, y) > 0$. Two transformations are supported:
1. **Sigmoidal Mapping**:
   \[
   k(x, y) = k_{\min} + (k_{\max} - k_{\min}) \cdot \frac{1}{1 + \exp(-\xi(x, y))}
   \]
   Guarantees $k(x, y) \in (k_{\min}, k_{\max})$ strictly everywhere.
2. **Log-Normal Mapping**:
   \[
   k(x, y) = \exp(\mu + \sigma \xi(x, y))
   \]
While non-linear mapping alters the Gaussian marginal distribution, it preserves spatial smoothness and the characteristic correlation length $\ell$.

### 9.3 In-Distribution (IID) vs. Out-of-Distribution (OOD) Definitions
Neural Process Twin defines an explicit physical separation between in-distribution benchmarks and out-of-distribution generalization tests:

| Property | In-Distribution (`train`, `val`, `test_iid`) | Out-of-Distribution (`test_ood`) | Physical Purpose |
| :--- | :--- | :--- | :--- |
| **Conductivity Regime** | Smooth GRF ($\ell \in [0.15, 0.35]\text{ m}$) | Sharp composite inclusions or fine GRF ($\ell \le 0.08\text{ m}$) | Tests high-frequency & discontinuous coefficient handling |
| **Contrast Ratio** | Low/Moderate ($k_{\max} / k_{\min} \in [2, 10]$) | Extreme ($k_{\max} / k_{\min} \in [20, 100]$) | Tests ill-conditioned interfacial gradient representation |
| **Volumetric Source** | Moderate Gaussian peaks ($q \le 500\text{ W/m}^3$) | Intense localized peaks ($q \in [500, 2000]\text{ W/m}^3$) | Tests localized steep gradient extrapolation |
| **Boundary Temp** | Standard range ($T_b \in [290, 360]\text{ K}$) | Extended range ($T_b \in [280, 420]\text{ K}$) | Tests thermal boundary condition generalization |

### 9.4 Quality Control & Conservation Diagnostics
Every candidate problem instance must satisfy three mandatory verification checks:
1. **Algebraic Residual**: Normalized sparse matrix residual $\|A T - b\|_2 / \|b\|_2 \le 10^{-6}$.
2. **Physical Heat Balance**: Global energy imbalance $|Q_{\mathrm{imbalance}}| / \max(|Q_{\mathrm{gen}}|, |Q_{\mathrm{boundary}}|, 1) \le 10^{-3}$.
3. **Thermal Positivity**: $\min(T(x, y)) > 0\text{ K}$.

Under `fail_fast=True`, any violation immediately halts synthesis. Under `fail_fast=False`, the instance is logged to an audit trail with detailed failure metrics and retried with a deterministic seed offset.

### 9.5 Safe PyTorch Streaming and Normalization
- **Leakage-Free Normalization**: `FieldNormalizer` statistics (mean, std, min, max) are fitted strictly on the `train` partition and serialized into the HDF5 archive.
- **Multi-Worker Safety**: To eliminate OS multiprocessing deadlock or pickled-handle errors, file handles are opened lazily per worker PID on first read and omitted from process serialization dictionaries via `__getstate__`.

---

## 10. Surrogate Modeling and Evaluation Framework

### 10.1 Deterministic Spatial Boundary Condition Representation
To ensure surrogate models uniquely distinguish boundary value problems at inference time without discarding corner information, boundary conditions are encoded into 8 spatial channels separated by normal orientation:
- **x-normal boundaries** (Left at $i=0$, Right at $i=nx-1$):
  - `bc_dirichlet_mask_x`: $1.0$ on Left/Right if Dirichlet, $0.0$ elsewhere.
  - `bc_dirichlet_val_x`: Prescribed Dirichlet temperature (normalized) on Left/Right.
  - `bc_neumann_mask_x`: $1.0$ on Left/Right if Neumann, $0.0$ elsewhere.
  - `bc_neumann_val_x`: Prescribed outward Neumann heat flux $j_{\text{out}}$ (normalized) on Left/Right.
- **y-normal boundaries** (Bottom at $j=0$, Top at $j=ny-1$):
  - `bc_dirichlet_mask_y`: $1.0$ on Bottom/Top if Dirichlet, $0.0$ elsewhere.
  - `bc_dirichlet_val_y`: Prescribed Dirichlet temperature (normalized) on Bottom/Top.
  - `bc_neumann_mask_y`: $1.0$ on Bottom/Top if Neumann, $0.0$ elsewhere.
  - `bc_neumann_val_y`: Prescribed outward Neumann heat flux $j_{\text{out}}$ (normalized) on Bottom/Top.

Combining these 8 boundary channels with normalized conductivity $k(x, y)$, heat source $q(x, y)$, and normalized spatial coordinates $x \in [0, 1], y \in [0, 1]$ yields the **canonical 12-channel model input tensor**:
\[
\mathbf{X} \in \mathbb{R}^{12 \times N_y \times N_x}
\]
This separation eliminates the corner ambiguity where adjacent boundary faces meet and allows exact, reversible reconstruction via `decode_boundary_conditions`.

### 10.2 Fourier Neural Operator (FNO-2D) Formulation
The Fourier Neural Operator learns a continuous mapping between infinite-dimensional function spaces:
\[
\mathcal{G}_\theta : \mathcal{A}(\Omega; \mathbb{R}^{12}) \longrightarrow \mathcal{U}(\Omega; \mathbb{R})
\]
Architecture:
1. **Lifting Projection**: $P : \mathbb{R}^{12} \to \mathbb{R}^{d_v}$ parameterized by a $1 \times 1$ convolution lifting spatial channels to latent channel width $d_v = 32$.
2. **Fourier Layers**: A sequence of $L$ layers computing:
   \[
   v_{l+1}(x) = \sigma\left( W_l v_l(x) + (\mathcal{K}_l v_l)(x) \right)
   \]
   where $W_l \in \mathbb{R}^{d_v \times d_v}$ is a local linear transform ($1 \times 1$ convolution), $\sigma$ is the GELU activation function, and $\mathcal{K}_l$ is a non-local kernel integral operator parameterized in Fourier space:
   \[
   (\mathcal{K}_l v_l)(x) = \mathcal{F}^{-1}\left( R_l \cdot (\mathcal{F} v_l) \right)(x)
   \]
   where $R_l \in \mathbb{C}^{d_v \times d_v \times K_y \times K_x}$ represents learned complex-valued Fourier mode multipliers.
3. **Conjugate Symmetry & Real FFT**:
   Because inputs are real-valued, the 2D real FFT (`torch.fft.rfft2`) satisfies Hermitian conjugate symmetry:
   $\hat{v}(-k_x, -k_y) = \hat{v}^*(k_x, k_y)$.
   Retained modes are truncated to $k_x \in [0, K_x - 1]$ and $k_y \in [0, K_y - 1] \cup [N_y - K_y, N_y - 1]$. Passing the exact spatial shape $s=(N_y, N_x)$ to `torch.fft.irfft2` guarantees exact dimensional reconstruction across even and odd grid resolutions.
4. **Projection Decoder**: $Q : \mathbb{R}^{d_v} \to \mathbb{R}^{128} \to \mathbb{R}^1$ parameterizes the mapping from latent representation back to the target scalar temperature field.

### 10.3 Convolutional Neural Network Baseline Formulation
The baseline CNN (`CNNBaseline2D`) provides a localized spatial surrogate:
- **Replicate Padding**: Standard zero padding treats the exterior of the domain as a fictitious zero-temperature reservoir, causing artificial temperature dips along domain perimeters. Setting padding mode to `replicate` projects perimeter boundary values continuously into the halo regions.
- **Residual Blocks**: Residual skip connections allow training deep convolutional backbones without gradient vanishing.

### 10.4 Scientific Evaluation Metrics in Kelvin
All physical metrics operate strictly on unnormalized fields in physical units:
- **Relative $L_2$ Error**:
  \[
  \text{Rel-}L_2 = \frac{\|T_{\text{pred}} - T_{\text{ref}}\|_2}{\|T_{\text{ref}}\|_2 + \epsilon}
  \]
- **Root Mean Square Error (RMSE)** [K]:
  \[
  \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (T_{\text{pred}, i} - T_{\text{ref}, i})^2}
  \]
- **Mean Absolute Error (MAE)** [K]:
  \[
  \text{MAE} = \frac{1}{N} \sum_{i=1}^N |T_{\text{pred}, i} - T_{\text{ref}, i}|
  \]
- **Maximum Absolute Error ($L_\infty$)** [K]:
  \[
  L_\infty = \max_i |T_{\text{pred}, i} - T_{\text{ref}, i}|
  \]

### 10.5 Discrete Physical Residual Diagnostics
To evaluate whether predicted temperature fields satisfy physical conservation laws independently of ground-truth labels:
1. **Discrete PDE Residual**:
   Evaluates the algebraic residual $r = A T_{\text{pred}} - b$ [W] using the conservative finite-volume stencil $A$ and load vector $b$ assembled from the reference solver.
2. **Global Boundary Flux Balance**:
   Evaluates total outward boundary heat flow $Q_{\text{boundary}} = \sum_{f \in \partial \Omega} j_{\text{out}, f} A_f$ vs. total volumetric heat generation $Q_{\text{gen}} = \sum_P q_P V_P$, reporting imbalance $|Q_{\text{boundary}} - Q_{\text{gen}}|$ in Watts [W].

---

## 11. Differentiable Physics-Aware Losses and Loss Formulations

### 11.1 Differentiable Conservative Finite-Volume Operator
To backpropagate physical PDE residual penalties directly into neural network parameters, the conservative finite-volume stencil of the reference solver is implemented as a fully vectorized, differentiable PyTorch operator (`DiscretePDEResidual`):

\[
r_P = (A T - b)_P = \sum_{f \in \mathcal{F}(P)} Q_f - q_P V_P \quad [\text{W}]
\]

1. **Internal Conductive Fluxes**:
   For internal faces between control volumes $P$ and neighbor $N$, harmonic conductivity averaging is computed differentiably:
   \[
   k_f = \frac{2 k_P k_N}{k_P + k_N + \epsilon}
   \]
   Conductive heat flow from $P$ to $N$:
   \[
   Q_{PN} = k_f \frac{A_f}{d_{PN}} (T_P - T_N)
   \]
   where $A_x = \Delta y L_z, d_x = \Delta x$ for East/West faces, and $A_y = \Delta x L_z, d_y = \Delta y$ for North/South faces.

2. **Boundary Flux Contributions**:
   - **Dirichlet Boundary**: Half-cell conductance $G_b = \frac{2 k_P A_b}{\Delta}$. Adds $G_b T_P$ to $(A T)_P$ and $G_b T_b$ to $b_P$.
   - **Neumann Boundary**: Prescribed outward flux $g_{\text{out}}$ [W/m²]. Adds $-g_{\text{out}} A_b$ to $b_P$.

This formulation matches the SciPy sparse linear system assembly $A T - b$ down to machine precision ($< 10^{-10}$ W in float64).

### 11.2 Dimensionless PDE Residual Loss
Raw residual $\|r\|_2$ is measured in Watts and scales with both spatial mesh refinement ($1/\sqrt{N}$) and the magnitude of thermal loads. To establish a scale-invariant, dimensionless optimization metric across arbitrary domains and parameter ranges, `DimensionlessPDELoss` normalizes the residual by the characteristic problem heat flow scale:

\[
\mathcal{L}_{\mathrm{PDE}} = \frac{\|r\|_2}{\|b\|_2 + Q_{\mathrm{floor}}}
\]

where $Q_{\mathrm{floor}} > 0$ (default: $1.0\text{ W}$) is a strictly positive denominator safeguard ensuring numerical stability and preventing infinite gradients when $b \approx 0$ (e.g. adiabatic domains with zero heat generation).

### 11.3 Explicit Boundary Condition Loss
`BoundaryConditionLoss` penalizes discrepancies between model predictions on perimeter cells and prescribed physical boundary values:
1. **Dirichlet Faces**:
   \[
   \mathcal{L}_{\mathrm{BC, Dir}} = \frac{1}{|\partial \Omega_D|} \sum_{f \in \partial \Omega_D} \left( \frac{T_{\mathrm{pred}, f} - T_{b, f}}{\Delta T_{\mathrm{char}}} \right)^2
   \]
   where $\Delta T_{\mathrm{char}} = \sigma_T$ (or $50.0\text{ K}$) non-dimensionalizes temperature errors.
2. **Neumann Faces**:
   \[
   \mathcal{L}_{\mathrm{BC, Neu}} = \frac{1}{|\partial \Omega_N|} \sum_{f \in \partial \Omega_N} \left( \frac{g_{\mathrm{pred}, f} - g_{b, f}}{g_{\mathrm{char}}} \right)^2
   \]
   where $g_{\mathrm{pred}} = -k_P \frac{\partial T}{\partial n}$ and $g_{\mathrm{char}} = 1000.0\text{ W/m}^2$.

### 11.4 Residual Domain and Spatial Orthogonality
To eliminate duplicate penalization when combining PDE residual loss with boundary condition loss:
- `residual_type="full"`: Evaluates $(A T - b)_P$ across all $N_y \times N_x$ cells, including boundary control volumes.
- `residual_type="interior"`: Evaluates $(A T - b)_P$ strictly on interior control volumes ($1 \le i \le N_x-2, 1 \le j \le N_y-2$). On these cells, all 4 bounding faces are internal faces; boundary values do not enter. This renders $\mathcal{L}_{\mathrm{PDE, int}}$ and $\mathcal{L}_{\mathrm{BC}}$ mathematically and spatially orthogonal.

### 11.5 Compound Objective Function
Surrogate training coordinates:
\[
\mathcal{L} = w_{\mathrm{data}} \mathcal{L}_{\mathrm{data}} + w_{\mathrm{pde}} \mathcal{L}_{\mathrm{pde}} + w_{\mathrm{bc}} \mathcal{L}_{\mathrm{bc}}
\]
All component losses are recorded and logged independently, preventing hidden loss obfuscation and enabling transparent multi-objective ablation studies.


