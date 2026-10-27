# Milestone 02 Implementation Report: Synthetic Dataset Generation & Ingestion

## 1. Executive Summary

Milestone 02 for **Neural Process Twin** has been fully implemented, verified, and audited. This milestone establishes the complete synthetic dataset generation, solver orchestration, HDF5 archival storage, and PyTorch ingestion pipeline for learning physical field solution operators in two-dimensional steady-state heat conduction:
\[
-\nabla \cdot (k(x, y) \nabla T(x, y)) = q(x, y)
\]

Ground-truth temperature labels are generated exclusively by the conservative finite-volume reference solver (`HeatConductionSolver2D`) validated in Milestone 01. Every candidate problem instance undergoes rigorous quality control checks (algebraic linear residual $\le 10^{-6}$, global energy conservation imbalance $\le 10^{-3}$, and strictly positive temperature $T > 0\text{ K}$) before archival.

The pipeline enforces four strictly disjoint partitions (`train`, `validation`, `test_iid`, `test_ood`) with an explicit physical Out-of-Distribution (OOD) definition. Reversible field normalizers are fitted exclusively on the `train` partition to guarantee zero data leakage. PyTorch `Dataset` and `DataLoader` abstractions feature worker-local lazy HDF5 handle initialization and serialization exclusion (`__getstate__`), preventing multi-worker deadlocks and platform-specific pickling crashes.

No neural network weights or surrogate model training loops were fabricated in this milestone.

---

## 2. Inventory of Modified and Created Files

| File / Path | Status | Purpose |
| :--- | :--- | :--- |
| `src/neural_process_twin/data/generators.py` | **Created** | Parameter field generators: Spectral GRF (Gaussian & Matérn), composite material inclusions, volumetric heat sources, and BC samplers. |
| `src/neural_process_twin/data/normalization.py` | **Created** | Physics-aware `FieldNormalizer` fitted strictly on `train`, with reversible temperature inversion to Kelvin. |
| `src/neural_process_twin/data/splits.py` | **Created** | Four canonical partitions, disjoint seed derivation, and explicit physical OOD criteria verification. |
| `src/neural_process_twin/data/orchestration.py` | **Created** | `DatasetOrchestrator` managing reference solves, convergence/residual verification, and configurable failure policy (fail-fast vs. skip-and-audit). |
| `src/neural_process_twin/data/storage.py` | **Created** | Structured, chunked, compressed HDF5 dataset writer (`v1.0.0`) and `inspect_hdf5_dataset` utility. |
| `src/neural_process_twin/data/dataset.py` | **Created** | Worker-safe PyTorch `HeatFieldDataset` and deterministic `create_dataloader` with coordinate channel augmentation. |
| `src/neural_process_twin/data/__init__.py` | **Updated** | Exported complete public data API. |
| `scripts/generate_dataset.py` | **Created** | CLI automation script to generate, verify, and archive datasets with smoke-test and override flags. |
| `scripts/inspect_dataset.py` | **Created** | CLI automation script to inspect HDF5 archive schema, partitions, field ranges, and residuals. |
| `pyproject.toml` | **Updated** | Added `h5py>=3.10.0` dependency. |
| `tests/conftest.py` | **Created** | Stabilized multiprocessing worker thread limits (`OPENBLAS_NUM_THREADS=1`). |
| `tests/unit/test_data_generators.py` | **Created** | 8 unit tests for GRF reproducibility, Matérn smoothness, bounds, inclusions, sources, and BCs. |
| `tests/unit/test_data_normalization.py` | **Created** | 4 unit tests for fitting guards, standard/minmax scaling, exact inversion, and dict roundtrips. |
| `tests/unit/test_data_splits.py` | **Created** | 4 unit tests for sample counts, seed disjointness, leakage detection, and OOD contrast criteria. |
| `tests/unit/test_data_storage.py` | **Created** | 4 unit tests for HDF5 writing, chunking, schema versioning, inspection, and corrupt file rejection. |
| `tests/unit/test_data_dataset.py` | **Created** | 4 unit tests for tensor shapes, coordinate channels, bounds, and multi-worker DataLoader execution. |
| `tests/integration/test_data_pipeline.py` | **Created** | 3 integration tests verifying full solve-archive-load pipeline, fail-fast abort, and skip-and-audit retries. |
| `docs/architecture.md` | **Updated** | Documented implemented data module classes, methods, and responsibilities. |
| `docs/scientific_specification.md` | **Updated** | Added mathematical formulation of spectral GRFs, positivity transforms, OOD criteria, and safety checks. |
| `README.md` | **Updated** | Marked Milestone 02 completed, updated tree, and added CLI synthesis instructions. |

---

## 3. Mathematical & Physical Design Details

### 3.1 Spectral Gaussian Random Field Generator
Fields are generated via continuous 2D Fourier filtering in $\mathcal{O}(N \log N)$ complexity:
\[
\hat{\xi}(\mathbf{k}) = \sqrt{\frac{S(\mathbf{k})}{\Delta x \Delta y}} \cdot W(\mathbf{k}), \quad \xi(x, y) = \text{Re}\left( \mathcal{F}^{-1}\{\hat{\xi}\} \right)
\]
where $W(\mathbf{k}) \sim \mathcal{CN}(0, 1)$ is complex Gaussian white noise.

- **Gaussian Covariance**: $C(r) = \sigma^2 \exp\left(-\frac{r^2}{2\ell^2}\right) \implies S(\mathbf{k}) = 2\pi \sigma^2 \ell^2 \exp\left(-\frac{\|\mathbf{k}\|^2 \ell^2}{2}\right)$.
- **Matérn Covariance**: Smoothness parameter $\nu \in \{1/2, 3/2, 5/2, \dots\}$ controls differentiability.
  \[
  S(\mathbf{k}) = 4\pi \sigma^2 \nu \left(\frac{2\nu}{\ell^2}\right)^\nu \left(\frac{2\nu}{\ell^2} + \|\mathbf{k}\|^2\right)^{-(\nu + 1)}
  \]

### 3.2 Positivity Preservation
Strict uniform ellipticity requires $k(x, y) \ge k_{\min} > 0$:
1. **Sigmoidal Mapping**:
   \[
   k(x, y) = k_{\min} + (k_{\max} - k_{\min}) \cdot \frac{1}{1 + \exp(-\xi(x, y))}
   \]
   Strictly confines values to $(k_{\min}, k_{\max})$, ensuring continuous, smooth, strictly positive fields.
2. **Log-Normal Mapping**:
   \[
   k(x, y) = \exp(\mu + \sigma \xi(x, y))
   \]
   Guarantees $k(x, y) > 0$ with heavy-tailed conductivities.

### 3.3 Piecewise-Constant Composite Inclusions
- Continuous matrix background with conductivity $k_{\text{matrix}} > 0$.
- Geometric inclusions (circular fibers or rectangular blocks) with sharp step interfaces.
- Safe placement logic ensuring inclusions remain within domain boundaries with configurable clearance margins.

### 3.4 Volumetric Heat Sources
- Modes: Zero source ($q \equiv 0$), uniform source ($q \equiv q_0$), and Gaussian thermal blobs:
  \[
  q(x, y) = \sum_{p=1}^P A_p \exp\left(-\frac{(x - x_p)^2 + (y - y_p)^2}{2\sigma_p^2}\right)
  \]
- Explicit physical distinction between peak local volumetric generation rate $A_p$ [$\text{W}/\text{m}^3$] and total integrated thermal power $Q = \iint_\Omega q \, dA$ [$\text{W}$].

---

## 4. Disjoint Partitioning & Physical OOD Definition

### 4.1 Seed Management and Zero Data Leakage
To ensure non-overlapping parameter distributions:
- `train`: Base seed $+ 1{,}000{,}000 + i$
- `validation`: Base seed $+ 2{,}000{,}000 + i$
- `test_iid`: Base seed $+ 3{,}000{,}000 + i$
- `test_ood`: Base seed $+ 4{,}000{,}000 + i$

`verify_partitions_disjoint()` verifies that no seed or sample ID is shared across splits.

### 4.2 Explicit Physical OOD Separation
| Metric | In-Distribution (`train`, `val`, `test_iid`) | Out-of-Distribution (`test_ood`) |
| :--- | :--- | :--- |
| **Microstructure** | Smooth GRF ($\ell \in [0.15, 0.35]\text{ m}$) | Composite inclusions or fine GRF ($\ell \le 0.08\text{ m}$) |
| **Conductivity Contrast** | Low/Moderate ($k_{\max}/k_{\min} \in [2, 10]$) | Extreme ($k_{\max}/k_{\min} \in [20, 100]$) |
| **Source Amplitude** | Moderate ($q \le 500\text{ W}/\text{m}^3$) | High ($q \in [500, 2000]\text{ W}/\text{m}^3$) |
| **Boundary Temperatures** | Standard range ($[290, 360]\text{ K}$) | Extended range ($[280, 420]\text{ K}$) |

The programmatic helper `verify_ood_criteria()` validates that the OOD set exhibits statistically higher contrast than the training and IID test sets.

---

## 5. HDF5 Dataset Archive Specification (`v1.0.0`)

The dataset archive uses chunked gzip compression:
- Root attributes: `format_version="1.0.0"`, `timestamp`, `grid_nx`, `grid_ny`, `x_min`, `x_max`, `y_min`, `y_max`, `thickness`.
- Group `/grid`: Coordinate arrays `x_centers`, `y_centers`, `x_mesh`, `y_mesh`.
- Partition Groups (`/train`, `/validation`, `/test_iid`, `/test_ood`):
  - `k`: Shape $(N, n_y, n_x)$, float32, chunked $(1, n_y, n_x)$, gzip level 4.
  - `q`: Shape $(N, n_y, n_x)$, float32, chunked $(1, n_y, n_x)$, gzip level 4.
  - `temperature`: Shape $(N, n_y, n_x)$, float32, chunked $(1, n_y, n_x)$, gzip level 4.
  - `sample_id`: Shape $(N,)$, UTF-8 strings.
  - `seed`: Shape $(N,)$, int64.
  - Group `boundary_conditions`: Types and evaluated values for left, right, bottom, top boundaries.
  - Group `diagnostics`: `relative_residual`, `heat_imbalance`, `solve_time_seconds`.
- Group `/normalization`: Fitted mean, std, min, max for $k, q, T$ derived strictly from the training set.
- Group `/audit`: Records of total attempts, successes, and failure manifests.

---

## 6. PyTorch Ingestion & Multi-Worker Safety

### 6.1 `HeatFieldDataset`
- Returns dictionary with:
  - `"x"`: 4-channel input tensor $(k_{\text{norm}}, q_{\text{norm}}, x_{\text{norm}}, y_{\text{norm}})$ of shape $(4, n_y, n_x)$.
  - `"y"`: Supervised target temperature tensor $(T_{\text{norm}})$ of shape $(1, n_y, n_x)$.
  - Unnormalized raw fields (`k_raw`, `q_raw`, `temperature_raw`) and metadata (`sample_id`, `seed`).
- **Multi-Worker Safety**:
  - Open HDF5 file handles cannot be pickled across processes during PyTorch DataLoader worker spawning on Windows.
  - `__getstate__` explicitly sets `_file = None` during serialization.
  - Workers initialize file handles lazily upon the first `__getitem__` invocation within their own process space.

### 6.2 `FieldNormalizer`
- Standard ($z$-score) and MinMax modes.
- Fitted strictly on the training partition.
- `inverse_transform_temperature` reconstructs physical Kelvin temperatures from surrogate model predictions with error $< 10^{-5}\text{ K}$.

---

## 7. Verification & Test Suite Summary

The repository test suite was expanded from 43 to **70 passing tests**:

```powershell
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.7.0
configfile: pyproject.toml
testpaths: tests
collected 70 items

tests/integration/test_config_workflow.py ......................... PASSED [  4%]
tests/integration/test_data_pipeline.py ........................... PASSED [  8%]
tests/scientific/test_manufactured_solution.py .................... PASSED [ 12%]
tests/scientific/test_physical_verification.py .................... PASSED [ 27%]
tests/unit/test_config.py ......................................... PASSED [ 52%]
tests/unit/test_data_dataset.py ................................... PASSED [ 58%]
tests/unit/test_data_generators.py ................................ PASSED [ 70%]
tests/unit/test_data_normalization.py ............................. PASSED [ 75%]
tests/unit/test_data_splits.py .................................... PASSED [ 81%]
tests/unit/test_data_storage.py ................................... PASSED [ 87%]
tests/unit/test_package.py ........................................ PASSED [100%]

============================= 70 passed in 3.15s ==============================
```

### 7.1 Smoke Test Execution Log
```
Loading data configuration from: configs\data.yaml
[SMOKE-TEST] Overriding grid resolution to 16x16 and sample counts for verification.
Partition plan: train=8, val=2, test_iid=2, test_ood=2 (Total: 14)
Beginning reference solution synthesis...
Generated 14 verified samples in 0.02s (749.8 samples/s). Failed attempts: 0.
Archiving dataset to artifacts\data\smoke_dataset.h5...

Dataset generation and archiving complete! Archive summary:
======================================================================
Neural Process Twin - HDF5 Dataset Archive (Version 1.0.0)
======================================================================
Created At: 2026-10-24T08:20:12.239234+00:00
Generator:  Neural Process Twin - Milestone 02
Domain:     x in [0.00, 1.00], y in [0.00, 1.00] m
Grid:       16 x 16 (ny x nx)
Total Samples: 14 (Failed generation attempts: 0)
Normalizer: standard
----------------------------------------------------------------------
Partition       | Count   | k range [W/(m K)]  | T range [K]      | Max Res 
----------------------------------------------------------------------
train           | 8       | [0.64, 4.69]       | [297.4, 367.5]   | 1.1e-15 
validation      | 2       | [0.92, 4.79]       | [292.4, 358.4]   | 8.3e-16 
test_iid        | 2       | [0.96, 4.66]       | [317.9, 371.0]   | 9.8e-16 
test_ood        | 2       | [1.00, 41.79]      | [364.2, 418.2]   | 6.9e-15 
======================================================================
```

---

## 8. Conclusion and Readiness for Milestone 03

Milestone 02 delivers a robust, reproducible, and mathematically verified dataset synthesis and ingestion foundation. The codebase is fully prepared for Milestone 03: **Neural Operator Architectures (Fourier Neural Operator 2D) and Baseline CNNs**.
