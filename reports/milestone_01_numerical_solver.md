# Milestone 01 Implementation Report: Numerical Reference Solver & Scientific Verification

## 1. Executive Summary

Milestone 01 for **Neural Process Twin** has been implemented and verified. A conservative, cell-centred finite-volume reference solver for 2D steady-state heat conduction:
\[
-\nabla \cdot (k(x, y) \nabla T(x, y)) = q(x, y)
\]
was constructed on uniform Cartesian grids with SciPy sparse matrix representations (`scipy.sparse.csr_matrix` and `scipy.sparse.linalg.spsolve`).

Every numerical convention, boundary condition treatment, harmonic interface conductivity averaging, and energy conservation diagnostic was derived from first principles and independently verified against analytical solutions and manufactured benchmarks.

No neural networks, surrogate models, or synthetic ML training pipelines were fabricated in this milestone.

---

## 2. Inventory of Modified and Created Files

| File / Path | Status | Purpose |
| :--- | :--- | :--- |
| `src/neural_process_twin/solvers/grid.py` | **Created** | Cell-centred 2D uniform Cartesian grid representation, metric properties ($V_P, A_x, A_y$), and field evaluation. |
| `src/neural_process_twin/solvers/boundary.py` | **Created** | Strongly typed `BoundarySide`, `BoundaryType`, `BoundaryCondition`, and `BoundaryConditions` with face coordinate evaluation. |
| `src/neural_process_twin/solvers/conductivity.py` | **Created** | Exact series resistance harmonic interface averaging `harmonic_average` and face conductance functions. |
| `src/neural_process_twin/solvers/assembly.py` | **Created** | Conservative finite-volume sparse stiffness matrix $A$ and RHS load vector $\mathbf{b}$ assembly, with pure-Neumann singularity detection. |
| `src/neural_process_twin/solvers/diagnostics.py` | **Created** | Independent physical energy balance diagnostics $\Delta Q = |Q_{\text{boundary, out}} - Q_{\text{generated}}|$ and algebraic residual norm metrics. |
| `src/neural_process_twin/solvers/solver.py` | **Created** | High-level `HeatConductionSolver2D` class, `solve_steady_heat_2d` function, and structured `SolverResult`. |
| `src/neural_process_twin/solvers/convergence.py` | **Created** | Discrete relative $L_2$ and $L_\infty$ error norms, convergence rate estimation, and Roache (1994, 1998) ASME Grid Convergence Index (GCI). |
| `src/neural_process_twin/solvers/__init__.py` | **Updated** | Exported full reference solver API. |
| `src/neural_process_twin/physics/manufactured.py` | **Created** | Analytical manufactured solutions (`SinusoidalMMSConstantK`, `SinusoidalMMSVariableK`, `LinearConduction1D`). |
| `src/neural_process_twin/physics/__init__.py` | **Updated** | Exported manufactured solution classes. |
| `src/neural_process_twin/utils/config.py` | **Updated** | Added `grid_type` to `GridConfig` (defaulting to `"cell_centered"`) with validation. |
| `configs/data.yaml` | **Updated** | Documented `grid_type: "cell_centered"` and control volume resolution semantics. |
| `tests/scientific/test_manufactured_solution.py` | **Created** | Method of Manufactured Solutions convergence study and GCI verification. |
| `tests/scientific/test_physical_verification.py` | **Created** | 10 independent physical verification checks (conservation, Dirichlet, Neumann, symmetry, interface conductivity, positivity, residuals). |
| `tests/scientific/test_placeholder.py` | **Removed** | Replaced placeholder with active scientific test suites. |
| `tests/scientific/README.md` | **Updated** | Documented scientific test suites, formulas, and execution commands. |
| `docs/scientific_specification.md` | **Updated** | Added complete FVM derivations, harmonic averaging, MMS formulas, convergence tables, and limitations. |
| `docs/architecture.md` | **Updated** | Documented implemented solver APIs and data structures. |
| `docs/experiment_protocol.md` | **Updated** | Added references to implemented energy balance diagnostics and solver residual baselines. |
| `README.md` | **Updated** | Updated milestone roadmap (Milestone 01 Completed) and repository structure tree. |

---

## 3. Method of Manufactured Solutions (MMS) Verification Results

Two independent manufactured solution test cases were evaluated across a sequence of four refined Cartesian grids ($N \in \{16, 32, 64, 128\}$):

### Case 1: Constant Conductivity MMS ($k_0 = 1.5\text{ W/(m K)}$)
\[
T^*(x, y) = 300 + 25 \sin(\pi x) \sin(\pi y), \quad q^*(x, y) = -k_0 \nabla^2 T^*
\]

| Grid Resolution | Relative $L_2$ Error | Observed Rate $p$ | $L_\infty$ Error [K] | Observed Rate $p$ |
| :---: | :---: | :---: | :---: | :---: |
| $16 \times 16$ | $1.2969 \times 10^{-4}$ | — | $4.8694 \times 10^{-3}$ | — |
| $32 \times 32$ | $3.2379 \times 10^{-5}$ | **2.0020** | $1.2188 \times 10^{-3}$ | **1.9983** |
| $64 \times 64$ | $8.0919 \times 10^{-6}$ | **2.0005** | $3.0480 \times 10^{-4}$ | **1.9996** |
| $128 \times 128$ | $2.0228 \times 10^{-6}$ | **2.0001** | $7.6208 \times 10^{-5}$ | **1.9999** |

### Case 2: Spatially Varying Conductivity MMS ($k(x, y) = 2.0 + 0.5 \cos(\pi x) \cos(\pi y)$)
\[
q^*(x, y) = -\nabla \cdot (k \nabla T^*) = - (\nabla k \cdot \nabla T^* + k \nabla^2 T^*)
\]

| Grid Resolution | Relative $L_2$ Error | Observed Rate $p$ |
| :---: | :---: | :---: |
| $16 \times 16$ | $1.3030 \times 10^{-4}$ | — |
| $32 \times 32$ | $3.2528 \times 10^{-5}$ | **2.0021** |
| $64 \times 64$ | $8.1292 \times 10^{-6}$ | **2.0005** |
| $128 \times 128$ | $2.0321 \times 10^{-6}$ | **2.0001** |

Both test cases exhibit monotonic error reduction by a factor of 4.000 per grid doubling, converging asymptotically to theoretical second-order accuracy ($\mathcal{O}(h^2)$) with $p = 2.0001$.

### Grid Convergence Index (GCI) Analysis ($r = 2.0, F_s = 1.25$)
- Apparent order of convergence: $p = 2.0001$
- Relative fine-grid error $\epsilon_{21}$: $2.02 \times 10^{-6}$
- Fine grid GCI ($\text{GCI}_{21}$): $0.00084\%$
- Asymptotic range ratio $\text{GCI}_{32} / (r^p \text{GCI}_{21})$: **1.0000** (confirms solutions lie deep within the asymptotic convergence regime).

---

## 4. Independent Physical Verification Results

All 10 required independent physical verification tests passed:

1. **Constant Solution**: In a domain with $q = 0$ and homogeneous Dirichlet boundaries $T = 325.0\text{ K}$, the computed field reproduces $325.0\text{ K}$ everywhere with max error $< 10^{-12}\text{ K}$.
2. **Linear Temperature Field**: Constant $k = 1.5$, zero source, Dirichlet $T_{\text{left}} = 300\text{ K}$, $T_{\text{right}} = 450\text{ K}$, and insulated adiabatic Neumann walls reproduce the 1D linear profile to machine precision ($< 10^{-11}\text{ K}$).
3. **Spatial Symmetry**: A symmetric domain with central Gaussian heat source and identical Dirichlet walls produces horizontal, vertical, and diagonal symmetry with error $< 10^{-12}\text{ K}$.
4. **Sparse Matrix Residual**: Across non-uniform conductivity problems, the linear residual $\|A \mathbf{T} - \mathbf{b}\|_2$ is $< 10^{-10}\text{ W}$ and relative residual is $< 10^{-11}$.
5. **Conductivity Positivity**: Zero or negative conductivities are rejected by `ValueError` enforcing uniform ellipticity.
6. **Grid Geometry Validation**: Inverted bounds ($x_{\min} \ge x_{\max}$), degenerate resolutions ($nx < 2$), and negative thicknesses are rejected.
7. **Dirichlet & Neumann Boundary Conditions**: Imposing positive outward flux ($g_{\mathrm{out}} = +50\text{ W/m}^2$, cooling) and negative outward flux ($g_{\mathrm{out}} = -50\text{ W/m}^2$, heating) matches analytical profiles to $< 10^{-11}\text{ K}$.
8. **Pure-Neumann Singularity Detection**: Completely Neumann boundaries without a gauge reference raise `PureNeumannSingularError`. Providing a gauge pin (`gauge_pin=(0, 0), gauge_temperature=293.15`) resolves the singularity and yields the exact uniform temperature to machine precision.
9. **Harmonic Interface Conductivity Averaging**: Across a discontinuous composite interface ($k_1 = 1.0, k_2 = 10.0$), harmonic averaging yields $k_{\text{interface}} = 1.8182\text{ W/(m K)}$ and reproduces the exact series thermal resistance and exact heat flux ($181.818\text{ W/m}^2$) to $< 10^{-10}\text{ W/m}^2$. Conversely, arithmetic averaging ($k_{\text{arith}} = 5.5$) produces an unphysical error $> 5\text{ W/m}^2$.
10. **Global Heat Balance Conservation**: Independent physical flux reconstruction verifies that total boundary heat leaving equals total volumetric heat generated:
    \[
    \frac{|Q_{\text{boundary, out}} - Q_{\text{generated}}|}{\max(|Q_{\text{generated}}|, 10^{-12})} < 10^{-12}
    \]

---

## 5. Verification Commands and Status

| Command | Status | Result Summary |
| :--- | :--- | :--- |
| `python scripts/verify_setup.py` | **Exit 0** | Runtime, dependencies, package imports, and default configs verified. |
| `pytest -v` | **Exit 0** | **43 passed in 0.56s** (3 integration, 3 MMS scientific, 10 physical scientific, 18 unit config, 9 unit package). |
| `ruff check .` | **Exit 0** | `All checks passed!` |
| `ruff format --check .` | **Exit 0** | `35 files already formatted`. |

---

## 6. Unresolved Issues

None. All numerical, physical, and software criteria for Milestone 01 are satisfied without workarounds or artificial assumptions.

---

## 7. Recommended Next Step: Session 02

Proceed to **Session 02: Synthetic Dataset Generation & Ingestion Pipelines**:
1. Implement parameterized field generators in `neural_process_twin.data`:
   - Gaussian Random Fields (GRF) with Matérn/Gaussian covariance for smooth heterogeneous conductivity.
   - Piecewise-constant composite inclusions with sharp contrast jumps.
   - Multi-peak Gaussian volumetric heat source generators.
2. Build batch dataset synthesis pipeline orchestrating the verified `HeatConductionSolver2D` to generate high-fidelity reference solution ensembles.
3. Implement archival storage in HDF5 format with dataset partitioning into Train, Validation, Test-IID, and Test-OOD splits.
4. Implement PyTorch `Dataset` and `DataLoader` abstractions with channel coordinate augmentation.
