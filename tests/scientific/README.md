# Scientific Verification Test Suite

This directory contains scientific verification tests for Neural Process Twin's numerical reference solver for 2D steady-state heat conduction:
\[
-\nabla \cdot (k(x, y) \nabla T(x, y)) = q(x, y)
\]

Implemented in **Milestone 01**.

---

## 1. Test Suite Overview

### 1.1 Method of Manufactured Solutions (`test_manufactured_solution.py`)
- **Constant Conductivity MMS (`test_mms_constant_conductivity_convergence`)**:
  - Analytical solution: $T^*(x, y) = T_0 + A \sin(\pi (x - x_{\min}) / L_x) \sin(\pi (y - y_{\min}) / L_y)$
  - Source: $q^*(x, y) = k_0 \pi^2 (1/L_x^2 + 1/L_y^2) A \sin(\dots) \sin(\dots)$
  - Refinement sequence: $N \in \{16, 32, 64, 128\}$
  - Asserts second-order convergence: $1.95 \le p \le 2.05$.
- **Variable Conductivity MMS (`test_mms_variable_conductivity_convergence`)**:
  - Non-uniform conductivity: $k(x, y) = k_0 + k_1 \cos(\pi (x - x_{\min}) / L_x) \cos(\pi (y - y_{\min}) / L_y)$
  - Fully coupled source: $q^*(x, y) = -\nabla \cdot (k \nabla T^*) = - (\nabla k \cdot \nabla T^* + k \nabla^2 T^*)$
  - Asserts second-order convergence: $1.95 \le p \le 2.05$.
- **Grid Convergence Index (`test_grid_convergence_index_mms`)**:
  - Implements Roache (1994, 1998) ASME V&V standard across grids $N \in \{32, 64, 128\}$.
  - Verifies asymptotic range condition $\text{GCI}_{32} / (r^p \text{GCI}_{21}) \approx 1.0$.

### 1.2 Physical Verification Checks (`test_physical_verification.py`)
1. **Constant Solution (`test_constant_solution`)**: Homogeneous Dirichlet $T=T_0$, $q=0 \implies T(x, y) = T_0$ to machine precision ($< 10^{-12}\text{ K}$).
2. **Linear Conduction (`test_linear_temperature_field`)**: Compatible Dirichlet boundaries with adiabatic top/bottom walls reproduce exact 1D linear profile to machine precision ($< 10^{-11}\text{ K}$).
3. **Spatial Symmetry (`test_spatial_symmetry`)**: Symmetric domain, symmetric Gaussian source, and symmetric Dirichlet boundaries produce horizontal, vertical, and diagonal symmetry within machine precision ($< 10^{-12}\text{ K}$).
4. **Sparse Matrix Residual (`test_sparse_matrix_residual`)**: Computed solution strictly satisfies assembled linear system $\|A \mathbf{T} - \mathbf{b}\|_2 < 10^{-10}\text{ W}$ and relative residual $< 10^{-11}$.
5. **Conductivity Positivity (`test_conductivity_positivity_validation`)**: Enforces strict ellipticity $k > 0$; rejects non-positive or non-finite values with informative `ValueError`.
6. **Grid Geometry Validation (`test_grid_geometry_validation`)**: Validates domain bounds ($x_{\min} < x_{\max}$), cell counts ($nx, ny \ge 2$), and thickness ($L_z > 0$).
7. **Boundary Conditions (`test_boundary_conditions_dirichlet_and_neumann`)**: Independent tests for Dirichlet and Neumann boundaries, covering both positive (heat leaving) and negative (heat entering) outward fluxes.
8. **Pure Neumann Singularity (`test_pure_neumann_singularity_and_gauge`)**: Detects undetermined additive temperature constant when all boundaries are Neumann; raises `PureNeumannSingularError`. Verifies unique solve when gauge pin is supplied.
9. **Interface Conductivity Averaging (`test_interface_harmonic_conductivity_averaging`)**: Demonstrates that harmonic interface averaging preserves exact series resistance and continuous heat flux across material discontinuities, contrasting with erroneous arithmetic averaging.
10. **Global Heat Balance (`test_global_physical_heat_balance`)**: Independently calculates boundary heat fluxes and verifies global energy conservation $|Q_{\text{boundary, out}} - Q_{\text{generated}}| < 10^{-10}\text{ W}$ (relative imbalance $< 10^{-12}$).

---

## 2. Running Scientific Tests

```powershell
# Run all scientific tests with verbose output
pytest tests/scientific/ -v

# Run only manufactured solution convergence tests
pytest tests/scientific/test_manufactured_solution.py -v

# Run only physical invariant verification tests
pytest tests/scientific/test_physical_verification.py -v
```
