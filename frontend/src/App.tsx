/**
 * Neural Process Twin — Interactive Scientific Digital Twin Dashboard.
 * Main Application Component.
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  fetchCapabilities,
  fetchHealth,
  runDualModeComparison,
  runSimulation,
  validateProblem,
} from './api';
import './App.css';
import { ComparisonView } from './components/ComparisonView';
import { DiagnosticsPanel } from './components/DiagnosticsPanel';
import { FieldCanvas } from './components/FieldCanvas';
import { ProblemConfigEditor } from './components/ProblemConfigEditor';
import { ProvenanceModal } from './components/ProvenanceModal';
import { StatusBadge } from './components/StatusBadge';
import type {
  CapabilitiesInfo,
  DualModeComparisonResponse,
  HealthInfo,
  InferenceRequestPayload,
  InferenceResult,
  SimulationMode,
  SimulationResponse,
  ValidationResult,
} from './types';

export const DEFAULT_CONFIG: InferenceRequestPayload = {
  grid: {
    x_min: 0.0,
    x_max: 1.0,
    y_min: 0.0,
    y_max: 1.0,
    nx: 32,
    ny: 32,
    thickness: 1.0,
  },
  conductivity: {
    pattern_type: 'inclusion',
    uniform_value: 2.5,
    inclusion_k_background: 1.0,
    inclusion_k_feature: 5.0,
    inclusion_center_x: 0.5,
    inclusion_center_y: 0.5,
    inclusion_radius: 0.2,
  },
  source: {
    pattern_type: 'uniform',
    uniform_value: 50.0,
    spot_peak_value: 200.0,
    spot_center_x: 0.5,
    spot_center_y: 0.5,
    spot_sigma: 0.15,
  },
  boundary_conditions: {
    left: { side: 'left', type: 'dirichlet', value: 300.0 },
    right: { side: 'right', type: 'dirichlet', value: 350.0 },
    bottom: { side: 'bottom', type: 'neumann', value: 0.0 },
    top: { side: 'top', type: 'neumann', value: 0.0 },
  },
  options: {
    enable_fallback: true,
    fallback_isolation: 'thread',
    provisional_mean_disagreement_threshold_k: 2.5,
    provisional_max_disagreement_threshold_k: 5.0,
    input_contrast_threshold: 7.16,
    input_q_max_threshold: 912.8,
    solver_timeout_seconds: 15.0,
  },
};

type ActiveFieldTab = 'temperature' | 'disagreement' | 'conductivity' | 'source';

export const App: React.FC = () => {
  const [config, setConfig] = useState<InferenceRequestPayload>(DEFAULT_CONFIG);
  const [mode, setMode] = useState<SimulationMode>('numerical_fvm');
  const [activeTab, setActiveTab] = useState<ActiveFieldTab>('temperature');

  // Server state
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilitiesInfo | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);

  // Result state
  const [result, setResult] = useState<SimulationResponse | InferenceResult | null>(null);
  const [comparisonResult, setComparisonResult] = useState<DualModeComparisonResponse | null>(null);

  // UI state
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isProvenanceOpen, setIsProvenanceOpen] = useState<boolean>(false);

  const abortControllerRef = useRef<AbortController | null>(null);

  // Initial load: health & capabilities
  useEffect(() => {
    let mounted = true;
    const initBackend = async () => {
      try {
        const [h, c] = await Promise.all([fetchHealth(), fetchCapabilities()]);
        if (mounted) {
          setHealth(h);
          setCapabilities(c);
        }
      } catch (err: any) {
        if (mounted) {
          console.warn('Backend service offline or unreachable during initial load:', err.message);
        }
      }
    };
    initBackend();
    return () => {
      mounted = false;
    };
  }, []);

  // Validation action
  const handleValidate = async () => {
    setErrorMessage(null);
    try {
      const res = await validateProblem(config);
      setValidationResult(res);
    } catch (err: any) {
      setErrorMessage(`Validation error: ${err.message}`);
    }
  };

  // Run simulation action
  const handleRunSimulation = async () => {
    // Abort ongoing request if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setErrorMessage(null);

    try {
      if (mode === 'numerical_fvm') {
        setLoadingStep('Solving conservative discrete system with SuperLU reference FVM...');
        const res = await runSimulation(config, 'numerical_fvm', controller.signal);
        setResult(res);
        setComparisonResult(null);
      } else if (mode === 'experimental_neural') {
        setLoadingStep('Evaluating fast multi-resolution A3 RefinedFNO2D surrogate on CPU...');
        const res = await runSimulation(config, 'experimental_neural', controller.signal);
        setResult(res);
        setComparisonResult(null);
      } else if (mode === 'experimental_neural_corrected') {
        setLoadingStep('Evaluating A3 RefinedFNO2D + Selective Boundary Jacobi physics correction...');
        const res = await runSimulation(config, 'experimental_neural_corrected', controller.signal);
        setResult(res);
        setComparisonResult(null);
      } else if (mode === 'comparison') {
        setLoadingStep('Executing dual-mode evaluation: Reference FVM + Experimental Surrogate...');
        const comp = await runDualModeComparison(config, 'experimental_neural', controller.signal);
        setComparisonResult(comp);
        setResult(comp.experimental_surrogate);
      } else if (mode === 'legacy_hybrid') {
        setLoadingStep('Evaluating legacy R1 hybrid screener & ensemble...');
        const res = await runSimulation(config, 'legacy_hybrid', controller.signal);
        setResult(res);
        setComparisonResult(null);
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setErrorMessage(err.message || 'Execution failed.');
      }
    } finally {
      setIsLoading(false);
      setLoadingStep('');
      abortControllerRef.current = null;
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
      setLoadingStep('');
    }
  };

  const xBounds: [number, number] = [config.grid.x_min, config.grid.x_max];
  const yBounds: [number, number] = [config.grid.y_min, config.grid.y_max];

  return (
    <div className="dashboard-root">
      {/* Top Header */}
      <header className="dashboard-header">
        <div className="header-inner">
          <div className="brand-title-group">
            <span className="brand-icon">⚡</span>
            <div>
              <h1 className="app-heading">NEURAL PROCESS TWIN</h1>
              <p className="app-subheading">
                Scientific Digital Twin Dashboard • 2D Steady-State Heat Conduction
              </p>
            </div>
          </div>

          <div className="header-status-group">
            <div className="badge-online">
              <span className="pulse-dot"></span>
              {health?.ensemble_loaded
                ? `Ensemble Online (${health.ensemble_member_count}/5)`
                : 'Ensemble Initializing'}
            </div>
            <button
              type="button"
              className="btn-provenance"
              onClick={() => setIsProvenanceOpen(true)}
            >
              Audited Provenance
            </button>
          </div>
        </div>
      </header>

      {/* Mode Status & Disclosure Banner */}
      {(mode === 'experimental_neural' || mode === 'experimental_neural_corrected') && (
        <div
          className="performance-disclosure-banner"
          style={{
            borderColor: 'rgba(245, 158, 11, 0.4)',
            backgroundColor: 'rgba(245, 158, 11, 0.12)',
            color: '#fef3c7',
          }}
        >
          <div className="disclosure-inner">
            <span style={{ fontSize: '18px' }}>⚠️</span>
            <span>
              <strong>EXPERIMENTAL NEURAL SURROGATE MODE ACTIVE:</strong> Using fast multi-resolution neural
              operator ({mode === 'experimental_neural_corrected' ? 'A3 + A4 Jacobi' : 'A3 RefinedFNO2D'}).
              Predictions are <strong>not certified to satisfy MaxAE ≤ 10 K</strong> (historical audit failure rate ~36%–80% on fine meshes).
              For authoritative or safety-critical calculations, switch to <strong>Verified Numerical Mode</strong>.
            </span>
          </div>
        </div>
      )}

      {mode === 'numerical_fvm' && (
        <div
          className="performance-disclosure-banner"
          style={{
            borderColor: 'rgba(16, 185, 129, 0.35)',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            color: '#d1fae5',
          }}
        >
          <div className="disclosure-inner">
            <span style={{ fontSize: '18px' }}>✓</span>
            <span>
              <strong>VERIFIED NUMERICAL MODE (DEFAULT TRUSTED REFERENCE):</strong> Conservative Finite Volume Method (FVM)
              with harmonic interface conductivities and direct SuperLU factorization. Continuous discretization error is second-order O(h²);
              discrete algebraic linear system solved to residual norm &lt; 10⁻¹² W.
            </span>
          </div>
        </div>
      )}

      {mode === 'comparison' && (
        <div
          className="performance-disclosure-banner"
          style={{
            borderColor: 'rgba(99, 102, 241, 0.35)',
            backgroundColor: 'rgba(99, 102, 241, 0.1)',
            color: '#e0e7ff',
          }}
        >
          <div className="disclosure-inner">
            <span style={{ fontSize: '18px' }}>🔬</span>
            <span>
              <strong>DUAL-MODE SCIENTIFIC COMPARISON:</strong> Solves both Verified Numerical FVM and Experimental Neural Surrogate
              on the identical problem. Computes true error metrics (RMSE, Rel L2, MaxAE) against the actual numerical reference.
            </span>
          </div>
        </div>
      )}

      {mode === 'legacy_hybrid' && (
        <div className="performance-disclosure-banner">
          <div className="disclosure-inner">
            <span>🛡️</span>
            <span>
              <strong>Legacy Hybrid Engine (Milestone 06):</strong> 32×32 5-member ensemble screener with automatic fallback
              to direct FVM solve on out-of-distribution inputs. Direct FVM SuperLU solve is ~11–13× faster on single 32×32 CPU problems.
            </span>
          </div>
        </div>
      )}

      {/* Main Container */}
      <main className="dashboard-main">
        <div className="dashboard-grid">
          {/* Left Column: Problem Specification and Controls */}
          <div className="problem-editor-container">
            <ProblemConfigEditor
              config={config}
              onChange={(updated) => {
                setConfig(updated);
                // Clear validation if problem modified
                if (validationResult) setValidationResult(null);
              }}
              onRun={handleRunSimulation}
              onValidate={handleValidate}
              mode={mode}
              onModeChange={(newMode) => {
                setMode(newMode);
                setResult(null);
                setComparisonResult(null);
              }}
              isLoading={isLoading}
              validationResult={validationResult}
            />

            {isLoading && (
              <div
                style={{
                  padding: '12px',
                  backgroundColor: 'var(--bg-secondary)',
                  borderRadius: '6px',
                  border: '1px solid var(--border-color)',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                }}
              >
                <span style={{ fontSize: '13px', color: '#93c5fd' }}>
                  ⏳ {loadingStep || 'Processing...'}
                </span>
                <button
                  type="button"
                  onClick={handleCancel}
                  style={{
                    backgroundColor: 'rgba(239, 68, 68, 0.2)',
                    border: '1px solid rgba(239, 68, 68, 0.4)',
                    color: '#f87171',
                    borderRadius: '4px',
                    padding: '4px 8px',
                    fontSize: '12px',
                    cursor: 'pointer',
                  }}
                >
                  Cancel
                </button>
              </div>
            )}

            {errorMessage && (
              <div
                style={{
                  padding: '12px',
                  backgroundColor: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid rgba(239, 68, 68, 0.4)',
                  borderRadius: '6px',
                  color: '#fca5a5',
                  fontSize: '13px',
                }}
              >
                <strong>Execution Error:</strong> {errorMessage}
              </div>
            )}
          </div>

          {/* Right Column: Visualizations & Diagnostics */}
          <div className="visualization-column">
            {/* If in Comparison Mode and comparison result is available */}
            {mode === 'comparison' && comparisonResult && (
              <ComparisonView comparison={comparisonResult} />
            )}

            {/* Inference Status Badge */}
            {result && <StatusBadge result={result} />}

            {/* Field View Switcher Tabs */}
            {result && (
              <div className="field-view-tabs-bar">
                <button
                  type="button"
                  className={`field-tab-btn ${activeTab === 'temperature' ? 'active-tab' : ''}`}
                  onClick={() => setActiveTab('temperature')}
                >
                  🌡️ Temperature Field (T)
                </button>
                <button
                  type="button"
                  className={`field-tab-btn ${activeTab === 'disagreement' ? 'active-tab' : ''}`}
                  onClick={() => setActiveTab('disagreement')}
                >
                  🎯 Epistemic Uncertainty (σ_T)
                </button>
                <button
                  type="button"
                  className={`field-tab-btn ${activeTab === 'conductivity' ? 'active-tab' : ''}`}
                  onClick={() => setActiveTab('conductivity')}
                >
                  🧱 Thermal Conductivity (k)
                </button>
                <button
                  type="button"
                  className={`field-tab-btn ${activeTab === 'source' ? 'active-tab' : ''}`}
                  onClick={() => setActiveTab('source')}
                >
                  🔥 Volumetric Heat Source (q)
                </button>
              </div>
            )}

            {/* Active Field Canvas */}
            {result && (
              <>
                {activeTab === 'temperature' && (
                  <FieldCanvas
                    data={result.temperature}
                    title="Steady-State Temperature Field"
                    unit="K"
                    colormap="plasma"
                    xBounds={xBounds}
                    yBounds={yBounds}
                    emptyMessage="Temperature field unavailable. Surrogate was rejected and fallback was disabled by policy."
                  />
                )}

                {activeTab === 'disagreement' && (
                  <FieldCanvas
                    data={'ensemble_disagreement_k' in result ? (result as any).ensemble_disagreement_k : null}
                    title="Ensemble Pointwise Epistemic Uncertainty (Std Dev)"
                    unit="K"
                    colormap="viridis"
                    xBounds={xBounds}
                    yBounds={yBounds}
                    emptyMessage="Epistemic uncertainty unavailable. Direct FVM reference solve is strictly deterministic. Single-pass neural surrogate does not fabricate uncertainty without ensemble/dropout."
                  />
                )}

                {activeTab === 'conductivity' && (
                  <FieldCanvas
                    data={result.conductivity_field}
                    title="Spatial Thermal Conductivity Distribution"
                    unit="W/(m·K)"
                    colormap="viridis"
                    xBounds={xBounds}
                    yBounds={yBounds}
                  />
                )}

                {activeTab === 'source' && (
                  <FieldCanvas
                    data={result.source_field}
                    title="Volumetric Heat Generation Rate"
                    unit="W/m³"
                    colormap="inferno"
                    xBounds={xBounds}
                    yBounds={yBounds}
                  />
                )}
              </>
            )}

            {/* Diagnostics Panel */}
            {result && (
              <DiagnosticsPanel
                diagnostics={result.diagnostics}
                intervals={'conformal_intervals' in result ? (result as any).conformal_intervals : null}
                solverInfo={'numerical_solver_info' in result ? (result as any).numerical_solver_info : null}
                experimentalDiagnostics={'experimental_diagnostics' in result ? (result as any).experimental_diagnostics : null}
              />
            )}

            {/* Latency Breakdown Card */}
            {result && (
              <div className="diag-section">
                <h4 className="diag-section-heading">Execution Latency Breakdown</h4>
                <div className="diag-grid">
                  {'latency_ms' in result
                    ? Object.entries((result as SimulationResponse).latency_ms).map(([k, v]) => (
                        <div key={k} className="diag-item">
                          <span className="diag-label">{k.replace(/_/g, ' ')}</span>
                          <span className="diag-value">
                            {typeof v === 'number' ? `${v.toFixed(2)} ms` : String(v)}
                          </span>
                        </div>
                      ))
                    : [
                        { label: 'Input Validation', val: (result as InferenceResult).latency.input_validation_ms },
                        { label: 'Surrogate Inference', val: (result as InferenceResult).latency.surrogate_inference_ms },
                        { label: 'Diagnostics Evaluation', val: (result as InferenceResult).latency.diagnostics_evaluation_ms },
                        { label: 'Solver Fallback', val: (result as InferenceResult).latency.solver_fallback_ms },
                        { label: 'Total Pipeline', val: (result as InferenceResult).latency.total_pipeline_ms, highlight: true },
                      ].map((item, idx) => (
                        <div key={idx} className="diag-item">
                          <span className="diag-label">{item.label}</span>
                          <span className="diag-value" style={item.highlight ? { color: '#60a5fa' } : {}}>
                            {item.val.toFixed(2)} ms
                          </span>
                        </div>
                      ))}
                </div>
              </div>
            )}

            {/* Empty State / Initial Guide */}
            {!result && !isLoading && (
              <div className="scientific-field-card">
                <div className="field-empty-placeholder">
                  <span className="empty-icon">📊</span>
                  <h3 style={{ margin: '0 0 8px 0', color: '#fff', fontSize: '16px' }}>
                    Digital Twin Interactive Inspection
                  </h3>
                  <p style={{ margin: 0, maxWidth: '440px', fontSize: '13px', lineHeight: 1.5 }}>
                    Select a preset or configure physical parameters on the left. Click{' '}
                    <strong>Run Simulation</strong> to execute the hybrid inference pipeline,
                    inspect temperature distributions, monitor energy conservation, and evaluate
                    conformal uncertainty bounds.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="dashboard-footer">
        Neural Process Twin • Hybrid Physics-AI Architecture • Fail-Closed Safety Guarantees •
        Milestone 07
      </footer>

      {/* Provenance & Integrity Modal */}
      <ProvenanceModal
        isOpen={isProvenanceOpen}
        onClose={() => setIsProvenanceOpen(false)}
        capabilities={capabilities}
      />
    </div>
  );
};

export default App;
