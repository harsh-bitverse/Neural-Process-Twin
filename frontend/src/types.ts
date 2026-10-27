/**
 * TypeScript type definitions for the Neural Process Twin dashboard.
 */

export type BoundarySide = 'left' | 'right' | 'bottom' | 'top';
export type BoundaryType = 'dirichlet' | 'neumann';
export type ConductivityPatternType = 'uniform' | 'inclusion' | 'gradient' | 'custom';
export type HeatSourcePatternType = 'uniform' | 'gaussian_spot' | 'custom';
export type SimulationMode = 'hybrid' | 'fvm' | 'comparison';

export interface GridConfig {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
  nx: number;
  ny: number;
  thickness: number;
}

export interface BoundaryConditionItem {
  side: BoundarySide;
  type: BoundaryType;
  value: number;
}

export interface BoundaryConditionsConfig {
  left: BoundaryConditionItem;
  right: BoundaryConditionItem;
  bottom: BoundaryConditionItem;
  top: BoundaryConditionItem;
}

export interface ConductivityConfig {
  pattern_type: ConductivityPatternType;
  uniform_value: number;
  inclusion_k_background: number;
  inclusion_k_feature: number;
  inclusion_center_x: number;
  inclusion_center_y: number;
  inclusion_radius: number;
  custom_values?: number[][];
}

export interface HeatSourceConfig {
  pattern_type: HeatSourcePatternType;
  uniform_value: number;
  spot_peak_value: number;
  spot_center_x: number;
  spot_center_y: number;
  spot_sigma: number;
  custom_values?: number[][];
}

export interface InferenceOptionsConfig {
  enable_fallback: boolean;
  fallback_isolation: 'thread' | 'process';
  provisional_mean_disagreement_threshold_k: number;
  provisional_max_disagreement_threshold_k: number;
  input_contrast_threshold: number;
  input_q_max_threshold: number;
  solver_timeout_seconds: number;
}

export interface InferenceRequestPayload {
  grid: GridConfig;
  conductivity: ConductivityConfig;
  source: HeatSourceConfig;
  boundary_conditions: BoundaryConditionsConfig;
  options: InferenceOptionsConfig;
}

export interface RejectionRecord {
  code: string;
  message: string;
  triggering_metric: string | null;
  observed_value: number | null;
  threshold_value: number | null;
}

export interface ConformalIntervals {
  nominal_confidence: number;
  pointwise_q_hat: number;
  simultaneous_q_hat: number;
  pointwise_mean_width_k: number;
  simultaneous_mean_width_k: number;
  calibration_dataset_id: string;
  disclaimer: string;
}

export interface PhysicalDiagnostics {
  pde_residual_l2_w: number;
  pde_residual_relative: number;
  pde_residual_linf_w: number;
  q_generated_total_w: number;
  q_boundary_out_total_w: number;
  signed_net_imbalance_w: number;
  absolute_flux_imbalance_w: number;
  relative_flux_imbalance: number;
  boundary_heat_flow_out_w: Record<string, number>;
}

export interface ExecutionLatency {
  input_validation_ms: number;
  surrogate_inference_ms: number;
  diagnostics_evaluation_ms: number;
  solver_fallback_ms: number;
  total_pipeline_ms: number;
}

export interface InferenceResult {
  status: 'accepted' | 'fallback_succeeded' | 'fallback_required' | 'invalid_input' | 'solver_failed';
  execution_path: 'surrogate_fast_path' | 'solver_fallback_path' | 'direct_fvm_solve' | 'aborted';
  temperature: number[][] | null;
  temperature_min_k: number | null;
  temperature_max_k: number | null;
  rejection: RejectionRecord;
  ensemble_disagreement_k: number[][] | null;
  mean_disagreement_k: number | null;
  max_disagreement_k: number | null;
  conformal_intervals: ConformalIntervals | null;
  diagnostics: PhysicalDiagnostics | null;
  latency: ExecutionLatency;
  provenance: Record<string, any>;
  conductivity_field: number[][];
  source_field: number[][];
  x_coords: number[];
  y_coords: number[];
}

export interface ComparisonResult {
  problem_summary: {
    grid: string;
    conductivity_contrast: number;
    source_q_max: number;
  };
  hybrid_inference: InferenceResult;
  fvm_reference: InferenceResult;
  temperature_difference_field: number[][] | null;
  rmse_k: number | null;
  relative_l2_error_pct: number | null;
  max_absolute_error_k: number | null;
  surrogate_latency_ms: number;
  fvm_latency_ms: number;
  fvm_speedup_factor: number;
  surrogate_acceptance_status: string;
  evaluation_notice: string;
}

export interface ValidationResult {
  is_valid: boolean;
  surrogate_compatible: boolean;
  k_min: number;
  k_max: number;
  k_contrast: number;
  q_max: number;
  within_contrast_envelope: boolean;
  within_source_envelope: boolean;
  errors: string[];
  warnings: string[];
}

export interface HealthInfo {
  status: string;
  version: string;
  device: string;
  torch_version: string;
  ensemble_loaded: boolean;
  ensemble_member_count: number;
}

export interface CapabilitiesInfo {
  surrogate_supported_resolution: [number, number];
  fvm_supported_resolution_range: string;
  ensemble_members: Array<{ member_name: string; sha256: string }>;
  training_domain_limits: {
    max_conductivity_contrast_ratio: number;
    max_volumetric_heat_source_w_m3: number;
    training_conductivity_range_w_m_k: [number, number];
  };
  disagreement_screening_thresholds: {
    spatial_mean_disagreement_tau_mean_k: number;
    peak_cell_disagreement_tau_max_k: number;
  };
  physical_units: Record<string, string>;
  performance_disclosure: string;
}
