# Artifacts Directory

This directory is reserved for exportable model artifacts, compiled computation graphs, calibration tables, and published baseline weights.

## Contents Guidelines

- `artifacts/models/`: Serialized model checkpoints intended for deployment or benchmarking against other PDE surrogates.
- `artifacts/calibration/`: Conformal prediction quantile tables and empirical calibration curves.
- `artifacts/benchmarks/`: Standardized reference solution datasets used across multiple experiment runs.

## Hygiene Rules

- Large binary artifacts (`*.pt`, `*.onnx`, `*.h5`) are ignored by Git in `.gitignore`.
- Provide sha256 checksums and provenance manifests (`manifest.json`) when sharing or archiving artifacts across teams or clusters.
