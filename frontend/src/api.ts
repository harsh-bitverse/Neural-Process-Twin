/**
 * API client services for communicating with the Neural Process Twin backend.
 */

import type {
  CapabilitiesInfo,
  ComparisonResult,
  DualModeComparisonResponse,
  HealthInfo,
  InferenceRequestPayload,
  InferenceResult,
  SimulationMode,
  SimulationResponse,
  ValidationResult,
} from './types';

const API_BASE = '/api';

export async function fetchHealth(): Promise<HealthInfo> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed with status ${res.status}`);
  }
  return res.json();
}

export async function fetchCapabilities(): Promise<CapabilitiesInfo> {
  const res = await fetch(`${API_BASE}/capabilities`);
  if (!res.ok) {
    throw new Error(`Capabilities lookup failed with status ${res.status}`);
  }
  return res.json();
}

export async function validateProblem(payload: InferenceRequestPayload): Promise<ValidationResult> {
  const res = await fetch(`${API_BASE}/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Validation failed with status ${res.status}`);
  }
  return res.json();
}

export async function runHybridInference(
  payload: InferenceRequestPayload,
  signal?: AbortSignal
): Promise<InferenceResult> {
  const res = await fetch(`${API_BASE}/infer/hybrid`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Hybrid inference failed with status ${res.status}`);
  }
  return res.json();
}

export async function runFvmInference(
  payload: InferenceRequestPayload,
  signal?: AbortSignal
): Promise<InferenceResult> {
  const res = await fetch(`${API_BASE}/infer/fvm`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `FVM reference solve failed with status ${res.status}`);
  }
  return res.json();
}

export async function runComparison(
  payload: InferenceRequestPayload,
  signal?: AbortSignal
): Promise<ComparisonResult> {
  const res = await fetch(`${API_BASE}/infer/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Comparison solve failed with status ${res.status}`);
  }
  return res.json();
}

export async function runSimulation(
  payload: InferenceRequestPayload,
  mode: SimulationMode,
  signal?: AbortSignal
): Promise<SimulationResponse> {
  const reqBody = {
    ...payload,
    mode: mode === 'comparison' ? 'numerical_fvm' : mode,
  };
  const res = await fetch(`${API_BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reqBody),
    signal,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Simulation failed with status ${res.status}`);
  }
  return res.json();
}

export async function runDualModeComparison(
  payload: InferenceRequestPayload,
  mode: 'experimental_neural' | 'experimental_neural_corrected' = 'experimental_neural',
  signal?: AbortSignal
): Promise<DualModeComparisonResponse> {
  const reqBody = {
    ...payload,
    mode,
  };
  const res = await fetch(`${API_BASE}/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reqBody),
    signal,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Dual-mode comparison failed with status ${res.status}`);
  }
  return res.json();
}
