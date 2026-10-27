# Experiment Configurations

This directory contains specialized configuration override files for reproducible scientific experiments with Neural Process Twin.

## Configuration Hierarchy

1. **Base Defaults**: Default parameters are specified in:
   - `configs/data.yaml` (physical parameters, domain bounds, boundary conditions, dataset split sizes)
   - `configs/training.yaml` (model architecture, optimizer, loss definitions, hardware settings)

2. **Experiment Overrides**: Individual YAML files in `configs/experiments/` specify targeted deviations from the base defaults (e.g., higher grid resolution, out-of-distribution contrast tests, or alternative neural operator architectures).

3. **Merging Protocol**: The experiment runner (implemented in Milestone 04) merges the experiment file over the base configuration using recursive dictionary updates (`neural_process_twin.utils.deep_merge`).

## Directory Layout Guidelines

- `configs/experiments/baseline_heat2d.yaml`: Reference baseline experiment configuration using 2D Fourier Neural Operator.
- `configs/experiments/ablation_physics_loss.yaml`: (Planned) Physics-informed loss ablation.
- `configs/experiments/contrast_sweep_ood.yaml`: (Planned) Generalization across extreme conductivity contrast ratios.
