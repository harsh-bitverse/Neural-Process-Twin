# Scientific Experimentation and Benchmarking Protocol

## 1. Purpose and Scope

This protocol establishes the scientific discipline and quality standards for conducting and reporting computational experiments within the **Neural Process Twin** framework.

Scientific machine learning surrogate models must be evaluated under standards that exceed standard computer vision benchmarks. In engineering digital twins, surrogate predictions guide safety-critical decisions; therefore, models must be scrutinized for both statistical fidelity and physical consistency.

---

## 2. Determinism and Random Seed Management

1. **Explicit Seed Seeding**:
   - Every experiment configuration must declare an explicit integer `seed` in both data synthesis and model training blocks.
   - The random seed must be set simultaneously across all random number generators:
     - Python standard library `random.seed(seed)`
     - NumPy `numpy.random.seed(seed)`
     - PyTorch CPU and CUDA generators `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`
2. **Deterministic Operations**:
   - Where exact bitwise reproducibility is demanded, PyTorch deterministic flags must be activated:
     `torch.backends.cudnn.deterministic = True` and `torch.use_deterministic_algorithms(True)`.
   - Any hardware-specific non-determinism (e.g., non-deterministic atomic operations on certain GPU kernels) must be logged explicitly.
3. **Multi-Seed Robustness**:
   - Core baseline comparisons must be evaluated across a minimum of 3 distinct random seeds (e.g., seeds `42, 137, 2026`).
   - Results must report mean $\pm$ standard deviation across seeds.

---

## 3. Configuration Snapshotting and Experiment Provenance

Every experiment run must automatically archive a complete provenance record in `experiments/<experiment_id>/`:

1. **Resolved Configuration**: A frozen copy of the fully resolved YAML configuration (base configuration merged with all experiment overrides).
2. **Git Versioning**: The exact Git commit hash, current branch, and status of any uncommitted modifications (`git diff` summary).
3. **Environment Manifest**: Complete software environment record, including:
   - Operating system and kernel version.
   - Python version and interpreter path.
   - Versions of PyTorch, NumPy, SciPy, and all loaded dependencies (`uv pip freeze` or `pip freeze`).
4. **Hardware Specifications**:
   - Host CPU model, core count, and system memory.
   - GPU model, compute capability, VRAM capacity, CUDA driver version, and cuDNN version.

---

## 4. Strict Data Partitioning and Leakage Prevention

1. **Partition Hygiene**:
   - Data generation must divide samples into four mutually exclusive splits:
     - **Training Set (Train)**: Used solely for parameter gradient updates.
     - **Validation Set (Val)**: Used for hyperparameter tuning, model selection, and early stopping.
     - **In-Distribution Test Set (Test-IID)**: Drawn from the exact parameter distribution as training, evaluated only after training is frozen.
     - **Out-of-Distribution Test Set (Test-OOD)**: Parameter sets featuring unseen conductivity contrasts, sharp geometric inclusions, or localized thermal sources.
2. **Zero Information Leakage**:
   - Feature normalization statistics (e.g., mean and standard deviation of fields, min-max bounds) must be computed exclusively on the Training Set. Under no circumstances may statistics be calculated over the entire dataset or test splits.
   - Generative parameter seeds for synthetic fields must be disjoint between splits.

---

## 5. Dual-Metric Evaluation Framework

Surrogate models must be evaluated under a dual-metric regime balancing statistical accuracy with physical validity:

### 5.1 Statistical Field Accuracy Metrics
Evaluated over all spatial grid points $\mathbf{x}_{i, j} \in \Omega$:

1. **Relative $L_2$ Error**:
   \[
   \epsilon_{L_2} = \frac{\|T_{\text{pred}} - T_{\text{ref}}\|_2}{\|T_{\text{ref}}\|_2} = \frac{\sqrt{\sum_{i, j} (T_{\text{pred}}(i, j) - T_{\text{ref}}(i, j))^2}}{\sqrt{\sum_{i, j} T_{\text{ref}}(i, j)^2}}
   \]
2. **Mean Absolute Error (MAE)**:
   \[
   \epsilon_{\text{MAE}} = \frac{1}{N_x N_y} \sum_{i, j} |T_{\text{pred}}(i, j) - T_{\text{ref}}(i, j)|
   \]
3. **Maximum Pointwise Error ($L_\infty$ Error)**:
   \[
   \epsilon_{L_\infty} = \max_{i, j} |T_{\text{pred}}(i, j) - T_{\text{ref}}(i, j)|
   \]
   High $L_\infty$ errors indicate localized physical failure modes (e.g., near boundaries or sharp conductivity steps) that are masked by global $L_2$ averages.

### 5.2 Physical Consistency Metrics
Evaluated without requiring ground-truth reference solutions:

1. **Discrete Interior PDE Residual Norm**:
   \[
   \mathcal{R}_{\text{interior}} = \| -\nabla_h \cdot (k \nabla_h T_{\text{pred}}) - q \|_2
   \]
   Computed using the identical conservative finite-volume stencil as the reference numerical solver (`neural_process_twin.solvers.assembly.assemble_steady_heat_system`).
2. **Global Boundary Heat Flux Conservation Balance**:
   In steady-state, net heat conducted across the boundary must equal total volumetric heat generated:
   \[
   \Delta Q = \left| \oint_{\partial \Omega} -k \frac{\partial T_{\text{pred}}}{\partial n} \, d\Gamma - \iint_\Omega q \, dx dy \right|
   \]
   Relative flux imbalance $\Delta Q / \max(\iint_\Omega q \, dx dy, 10^{-12})$ is evaluated directly via `neural_process_twin.solvers.diagnostics.compute_heat_balance`.
3. **Sparse Solver Residual Baseline**:
   Reference solutions must satisfy $\|A \mathbf{T}_{\text{ref}} - \mathbf{b}\|_2 \le 10^{-10}\text{ W}$ to guarantee that ground truth data does not inject numerical noise into surrogate training.

---

## 6. Fair Latency and Throughput Benchmarking

Claims of surrogate speedup must be established with strict benchmarking methodology:

1. **Apples-to-Apples Hardware Alignment**:
   - Compare surrogate CPU inference against solver CPU execution on identical core counts.
   - If evaluating surrogate GPU acceleration, also compare against accelerated/parallel numerical solvers where available.
2. **Warmup Cycles**:
   - Discard the first 10 forward passes to avoid initialization, module loading, and JIT compilation overhead.
3. **CUDA Synchronization**:
   - All GPU latency measurements must use `torch.cuda.Event(enable_timing=True)` or wrap execution in `torch.cuda.synchronize()` before taking timestamps.
4. **Evaluation Modes**:
   - **Batch Size 1 (Real-time Twin Mode)**: Wall-clock latency for single-query forward prediction (crucial for interactive digital twin controllers).
   - **High-Throughput Batched Mode**: Maximum samples processed per second across varying batch sizes ($B \in \{16, 64, 256\}$).
5. **Reference Solver Factoring Consideration**:
   - If the numerical solver factors the stiffness matrix once for fixed geometry and solves for multiple right-hand sides, this distinction must be documented (full assemble+solve vs. back-substitution only).

---

## 7. Failure Case Preservation and Outlier Analysis

Cherry-picking successful predictions is strictly forbidden:

1. **Mandatory Failure Case Extraction**:
   - Every benchmark report must isolate and visualize the top 5% worst predictions ranked by $L_\infty$ error and PDE residual norm.
2. **Outlier Analysis**:
   - Inspect whether failure correlates with high conductivity contrast ratios, proximity to boundary condition discontinuities, or high-frequency source terms.
3. **Negative Temperature Detection**:
   - Check whether any unconstrained surrogate predictions violate physical bounds ($T_{\text{pred}} \le 0\text{ K}$).
4. **Archive Integrity**:
   - Failure cases must be saved alongside nominal results in `reports/` and `artifacts/` rather than discarded.

---

## 8. Physics-Aware Loss Ablation Protocol

### 8.1 Multi-Objective Loss Formulation
Surrogate models incorporating physics regularization optimize:
\[
\mathcal{L} = w_{\mathrm{data}} \mathcal{L}_{\mathrm{data}} + w_{\mathrm{pde}} \mathcal{L}_{\mathrm{pde}} + w_{\mathrm{bc}} \mathcal{L}_{\mathrm{bc}}
\]

### 8.2 Experimental Controls and Hygiene
1. **Disjoint Model Selection**:
   Hyperparameter weights ($w_{\mathrm{pde}}, w_{\mathrm{bc}}$) and early stopping checkpoints must be selected strictly on the `validation` partition. Test sets (`test_iid`, `test_ood`) must remain frozen and evaluated only once per final model checkpoint.
2. **Identical Seeds and Data Splits**:
   All ablation runs must share identical random seeds, data splits, batch sizes, optimizer schedules, and network widths to isolate the exact impact of physics losses.
3. **Controlled Ablation Matrix**:
   - `ablation_cnn_supervised`: $w_{\mathrm{data}}=1.0, w_{\mathrm{pde}}=0.0, w_{\mathrm{bc}}=0.0$
   - `ablation_cnn_physics`: $w_{\mathrm{data}}=1.0, w_{\mathrm{pde}}=0.01, w_{\mathrm{bc}}=0.0$ (`residual_type="full"`)
   - `ablation_fno_supervised`: $w_{\mathrm{data}}=1.0, w_{\mathrm{pde}}=0.0, w_{\mathrm{bc}}=0.0$
   - `ablation_fno_physics`: $w_{\mathrm{data}}=1.0, w_{\mathrm{pde}}=0.01, w_{\mathrm{bc}}=0.0$ (`residual_type="full"`)
   - `ablation_fno_physics_bc`: $w_{\mathrm{data}}=1.0, w_{\mathrm{pde}}=0.01, w_{\mathrm{bc}}=0.05$ (`residual_type="interior"`)
4. **Independent Residual Reporting**:
   Residual evaluations must use the conservative finite-volume stencil $A T - b$ and global flux balance $|Q_{\mathrm{boundary}} - Q_{\mathrm{gen}}|$ evaluated in physical Kelvin and Watts, entirely separate from the training loss calculation.

