/**
 * Neural Process Twin — Interactive Scientific Digital Twin Dashboard.
 * Main Application Component.
 */

import React, { useEffect, useRef, useState } from 'react';
import {
  fetchCapabilities,
  fetchHealth,
  runComparison,
  runFvmInference,
  runHybridInference,
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
  ComparisonResult,
  HealthInfo,
  InferenceRequestPayload,
  InferenceResult,
  SimulationMode,
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
  const [mode, setMode] = useState<SimulationMode>('hybrid');
  const [activeTab, setActiveTab] = useState<ActiveFieldTab>('temperature');

  // Server state
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilitiesInfo | null>(null);
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);

  // Result state
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [comparisonResult, setComparisonResult] = useState<ComparisonResult | null>(null);

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
      if (mode === 'hybrid') {
        setLoadingStep('Evaluating physical gates & neural surrogate ensemble...');
        const res = await runHybridInference(config, controller.signal);
        setResult(res);
        setComparisonResult(null);
      } else if (mode === 'fvm') {
        setLoadingStep('Solving conservative discrete system with SuperLU reference FVM...');
        const res = await runFvmInference(config, controller.signal);
        setResult(res);
        setComparisonResult(null);
      } else if (mode === 'comparison') {
        setLoadingStep('Running dual evaluation: Neural Ensemble + Reference FVM...');
        const comp = await runComparison(config, controller.signal);
        setComparisonResult(comp);
        setResult(comp.hybrid_inference);
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

      {/* Performance Disclosure Banner */}
      <div className="performance-disclosure-banner">
        <div className="disclosure-inner">
          <span>ℹ️</span>
          <span>
            <strong>Scientific Performance Note:</strong> Direct FVM SuperLU solve is ~11–13× faster
            on single 32×32 CPU problems than evaluating the 5-member neural surrogate ensemble. The
            neural operator is designed for batched GPU workloads and rapid parameter sweeps.
          </span>
        </div>
      </div>

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
                    data={result.ensemble_disagreement_k}
                    title="Ensemble Pointwise Epistemic Uncertainty (Std Dev)"
                    unit="K"
                    colormap="viridis"
                    xBounds={xBounds}
                    yBounds={yBounds}
                    emptyMessage="Uncertainty unavailable. Direct FVM reference solve is strictly deterministic and does not fabricate uncertainty."
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
                intervals={result.conformal_intervals}
              />
            )}

            {/* Latency Breakdown Card */}
            {result && (
              <div className="diag-section">
                <h4 className="diag-section-heading">Execution Latency Breakdown</h4>
                <div className="diag-grid">
                  <div className="diag-item">
                    <span className="diag-label">Input Validation</span>
                    <span className="diag-value">
                      {result.latency.input_validation_ms.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">Surrogate Inference</span>
                    <span className="diag-value">
                      {result.latency.surrogate_inference_ms.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">Diagnostics Evaluation</span>
                    <span className="diag-value">
                      {result.latency.diagnostics_evaluation_ms.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">Solver Fallback</span>
                    <span className="diag-value">
                      {result.latency.solver_fallback_ms.toFixed(2)} ms
                    </span>
                  </div>
                  <div className="diag-item">
                    <span className="diag-label">Total Pipeline</span>
                    <span className="diag-value" style={{ color: '#60a5fa' }}>
                      {result.latency.total_pipeline_ms.toFixed(2)} ms
                    </span>
                  </div>
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
