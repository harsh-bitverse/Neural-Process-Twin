# Experiments Directory

This directory stores experiment run manifests, logs, and checkpoints generated during model development.

## Directory Structure (Per Experiment)

Each experiment run must be placed in an isolated subdirectory named by timestamp or experiment tag:

```
experiments/
└── <timestamp>_<experiment_tag>/
    ├── config.resolved.yaml     # Frozen copy of fully merged configuration
    ├── run_metadata.json        # Git commit, Python/PyTorch versions, hardware specs
    ├── checkpoints/             # Model weights (best and final checkpoints)
    │   ├── best_model.pt
    │   └── final_checkpoint.pt
    ├── metrics/                 # Training logs and epoch histories
    │   └── train_history.json
    └── evaluation/              # Test evaluation outputs and failure cases
        ├── test_metrics.json
        └── worst_case_samples.png
```

## Hygiene and Data Management

- Model checkpoint binaries (`*.pt`, `*.ckpt`) and large evaluation outputs in this directory are excluded from Git tracking via `.gitignore`.
- Experiment metadata, resolved YAML configs, and summarized metrics JSON files should be tracked for permanent reproducibility.
