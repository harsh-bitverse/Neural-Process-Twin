/**
 * Interactive physical problem configuration editor for 2D heat conduction.
 */

import React from 'react';
import type {
  ConductivityConfig,
  HeatSourceConfig,
  InferenceRequestPayload,
  SimulationMode,
  ValidationResult,
} from '../types';

interface ProblemConfigEditorProps {
  config: InferenceRequestPayload;
  onChange: (updated: InferenceRequestPayload) => void;
  onRun: () => void;
  onValidate: () => void;
  mode: SimulationMode;
  onModeChange: (mode: SimulationMode) => void;
  isLoading: boolean;
  validationResult: ValidationResult | null;
}

export const ProblemConfigEditor: React.FC<ProblemConfigEditorProps> = ({
  config,
  onChange,
  onRun,
  onValidate,
  mode,
  onModeChange,
  isLoading,
  validationResult,
}) => {
  // Preset handler
  const applyPreset = (presetName: string) => {
    const updated = JSON.parse(JSON.stringify(config)) as InferenceRequestPayload;

    if (presetName === 'nominal_composite') {
      updated.grid.nx = 32;
      updated.grid.ny = 32;
      updated.conductivity = {
        pattern_type: 'inclusion',
        uniform_value: 2.5,
        inclusion_k_background: 1.0,
        inclusion_k_feature: 5.0,
        inclusion_center_x: 0.5,
        inclusion_center_y: 0.5,
        inclusion_radius: 0.2,
      };
      updated.source = {
        pattern_type: 'uniform',
        uniform_value: 50.0,
        spot_peak_value: 200.0,
        spot_center_x: 0.5,
        spot_center_y: 0.5,
        spot_sigma: 0.15,
      };
      updated.boundary_conditions.left = { side: 'left', type: 'dirichlet', value: 300.0 };
      updated.boundary_conditions.right = { side: 'right', type: 'dirichlet', value: 350.0 };
      updated.boundary_conditions.bottom = { side: 'bottom', type: 'neumann', value: 0.0 };
      updated.boundary_conditions.top = { side: 'top', type: 'neumann', value: 0.0 };
    } else if (presetName === 'high_contrast') {
      updated.grid.nx = 32;
      updated.grid.ny = 32;
      updated.conductivity = {
        pattern_type: 'inclusion',
        uniform_value: 1.0,
        inclusion_k_background: 1.0,
        inclusion_k_feature: 30.0, // 30:1 > 7.16 triggers fallback
        inclusion_center_x: 0.5,
        inclusion_center_y: 0.5,
        inclusion_radius: 0.25,
      };
      updated.source = {
        pattern_type: 'uniform',
        uniform_value: 50.0,
        spot_peak_value: 200.0,
        spot_center_x: 0.5,
        spot_center_y: 0.5,
        spot_sigma: 0.15,
      };
      updated.boundary_conditions.left = { side: 'left', type: 'dirichlet', value: 300.0 };
      updated.boundary_conditions.right = { side: 'right', type: 'dirichlet', value: 350.0 };
      updated.boundary_conditions.bottom = { side: 'bottom', type: 'neumann', value: 0.0 };
      updated.boundary_conditions.top = { side: 'top', type: 'neumann', value: 0.0 };
    } else if (presetName === 'uniform_conduction') {
      updated.grid.nx = 32;
      updated.grid.ny = 32;
      updated.conductivity = {
        pattern_type: 'uniform',
        uniform_value: 2.5,
        inclusion_k_background: 1.0,
        inclusion_k_feature: 5.0,
        inclusion_center_x: 0.5,
        inclusion_center_y: 0.5,
        inclusion_radius: 0.2,
      };
      updated.source = {
        pattern_type: 'uniform',
        uniform_value: 50.0,
        spot_peak_value: 200.0,
        spot_center_x: 0.5,
        spot_center_y: 0.5,
        spot_sigma: 0.15,
      };
      updated.boundary_conditions.left = { side: 'left', type: 'dirichlet', value: 300.0 };
      updated.boundary_conditions.right = { side: 'right', type: 'dirichlet', value: 350.0 };
      updated.boundary_conditions.bottom = { side: 'bottom', type: 'neumann', value: 0.0 };
      updated.boundary_conditions.top = { side: 'top', type: 'neumann', value: 0.0 };
    } else if (presetName === 'extreme_source') {
      updated.grid.nx = 32;
      updated.grid.ny = 32;
      updated.conductivity = {
        pattern_type: 'uniform',
        uniform_value: 2.0,
        inclusion_k_background: 1.0,
        inclusion_k_feature: 5.0,
        inclusion_center_x: 0.5,
        inclusion_center_y: 0.5,
        inclusion_radius: 0.2,
      };
      updated.source = {
        pattern_type: 'gaussian_spot',
        uniform_value: 0.0,
        spot_peak_value: 1200.0, // 1200 > 912.8 triggers fallback
        spot_center_x: 0.5,
        spot_center_y: 0.5,
        spot_sigma: 0.12,
      };
      updated.boundary_conditions.left = { side: 'left', type: 'dirichlet', value: 300.0 };
      updated.boundary_conditions.right = { side: 'right', type: 'dirichlet', value: 300.0 };
      updated.boundary_conditions.bottom = { side: 'bottom', type: 'dirichlet', value: 300.0 };
      updated.boundary_conditions.top = { side: 'top', type: 'dirichlet', value: 300.0 };
    } else if (presetName === 'grid_32') {
      updated.grid.nx = 32;
      updated.grid.ny = 32;
    } else if (presetName === 'grid_64') {
      updated.grid.nx = 64;
      updated.grid.ny = 64;
    } else if (presetName === 'grid_128') {
      updated.grid.nx = 128;
      updated.grid.ny = 128;
    } else if (presetName === 'grid_256') {
      updated.grid.nx = 256;
      updated.grid.ny = 256;
    }

    onChange(updated);
  };

  const isGrid32 = config.grid.nx === 32 && config.grid.ny === 32;
  const isMultiResSupported =
    config.grid.nx === config.grid.ny && [32, 64, 128, 256].includes(config.grid.nx);

  let gridBadgeText = '✓ Verified Numerical Mesh';
  let gridBadgeClass = 'badge-surrogate-supported';

  if (mode === 'experimental_neural' || mode === 'experimental_neural_corrected') {
    if (isMultiResSupported) {
      gridBadgeText = `✓ ${config.grid.nx}x${config.grid.ny} Multi-Res Surrogate Validated`;
      gridBadgeClass = 'badge-surrogate-supported';
    } else {
      gridBadgeText = `⚠️ Unvalidated for Neural Surrogate (Requires 32, 64, 128, 256)`;
      gridBadgeClass = 'badge-fvm-routed';
    }
  } else if (mode === 'legacy_hybrid') {
    gridBadgeText = isGrid32 ? '✓ 32x32 Surrogate Optimized' : 'ℹ️ Direct FVM Solver Route';
    gridBadgeClass = isGrid32 ? 'badge-surrogate-supported' : 'badge-fvm-routed';
  } else {
    gridBadgeText = `✓ ${config.grid.nx}x${config.grid.ny} Conservative Numerical Reference`;
    gridBadgeClass = 'badge-surrogate-supported';
  }

  return (
    <div className="problem-editor-container">
      {/* Simulation Mode Selector */}
      <div className="mode-selector-card">
        <label className="section-title">Inference Execution Mode</label>
        <div className="mode-button-group">
          <button
            type="button"
            className={`mode-btn ${mode === 'numerical_fvm' ? 'active-mode' : ''}`}
            onClick={() => onModeChange('numerical_fvm')}
          >
            📐 Verified Numerical FVM (Trusted Reference)
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === 'experimental_neural' ? 'active-mode' : ''}`}
            onClick={() => onModeChange('experimental_neural')}
          >
            ⚡ Experimental Neural (A3 Refined FNO)
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === 'experimental_neural_corrected' ? 'active-mode' : ''}`}
            onClick={() => onModeChange('experimental_neural_corrected')}
          >
            🧪 Experimental Neural + Jacobi (A4 Corrected)
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === 'comparison' ? 'active-mode' : ''}`}
            onClick={() => onModeChange('comparison')}
          >
            🔬 Dual-Mode Comparison (Reference vs Neural)
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === 'legacy_hybrid' ? 'active-mode' : ''}`}
            onClick={() => onModeChange('legacy_hybrid')}
          >
            🛡️ Legacy Hybrid Twin (32x32 R1)
          </button>
        </div>
      </div>

      {/* Preset Buttons */}
      <div className="editor-card">
        <label className="section-title">Research Benchmark Presets</label>
        <div className="preset-buttons-row">
          <button type="button" className="preset-pill" onClick={() => applyPreset('nominal_composite')}>
            Nominal Composite
          </button>
          <button type="button" className="preset-pill" onClick={() => applyPreset('high_contrast')}>
            High Contrast (OOD)
          </button>
          <button type="button" className="preset-pill" onClick={() => applyPreset('uniform_conduction')}>
            Uniform Baseline
          </button>
          <button type="button" className="preset-pill" onClick={() => applyPreset('extreme_source')}>
            Extreme Source (OOD)
          </button>
        </div>
        <div className="preset-buttons-row" style={{ marginTop: '8px' }}>
          <span style={{ fontSize: '12px', color: '#94a3b8', alignSelf: 'center', marginRight: '6px' }}>Mesh:</span>
          <button type="button" className="preset-pill" onClick={() => applyPreset('grid_32')}>
            32x32
          </button>
          <button type="button" className="preset-pill" onClick={() => applyPreset('grid_64')}>
            64x64
          </button>
          <button type="button" className="preset-pill" onClick={() => applyPreset('grid_128')}>
            128x128
          </button>
          <button type="button" className="preset-pill" onClick={() => applyPreset('grid_256')}>
            256x256
          </button>
        </div>
      </div>

      {/* Domain & Grid Resolution */}
      <div className="editor-card">
        <div className="card-header-with-badge">
          <label className="section-title">Domain Geometry & Resolution</label>
          <span className={`grid-badge ${gridBadgeClass}`}>
            {gridBadgeText}
          </span>
        </div>

        <div className="editor-form-grid-3">
          <div className="form-group">
            <label>Length x [m]</label>
            <input
              type="number"
              step="0.1"
              value={config.grid.x_max}
              onChange={(e) =>
                onChange({ ...config, grid: { ...config.grid, x_max: parseFloat(e.target.value) || 1.0 } })
              }
            />
          </div>
          <div className="form-group">
            <label>Length y [m]</label>
            <input
              type="number"
              step="0.1"
              value={config.grid.y_max}
              onChange={(e) =>
                onChange({ ...config, grid: { ...config.grid, y_max: parseFloat(e.target.value) || 1.0 } })
              }
            />
          </div>
          <div className="form-group">
            <label>Thickness t [m]</label>
            <input
              type="number"
              step="0.1"
              value={config.grid.thickness}
              onChange={(e) =>
                onChange({ ...config, grid: { ...config.grid, thickness: parseFloat(e.target.value) || 1.0 } })
              }
            />
          </div>
        </div>

        <div className="editor-form-grid-2">
          <div className="form-group">
            <label>Grid Cells nx (max 256)</label>
            <input
              type="number"
              step="1"
              min="4"
              max="256"
              value={config.grid.nx}
              onChange={(e) =>
                onChange({ ...config, grid: { ...config.grid, nx: parseInt(e.target.value, 10) || 32 } })
              }
            />
          </div>
          <div className="form-group">
            <label>Grid Cells ny (max 256)</label>
            <input
              type="number"
              step="1"
              min="4"
              max="256"
              value={config.grid.ny}
              onChange={(e) =>
                onChange({ ...config, grid: { ...config.grid, ny: parseInt(e.target.value, 10) || 32 } })
              }
            />
          </div>
        </div>
      </div>

      {/* Thermal Conductivity */}
      <div className="editor-card">
        <label className="section-title">Thermal Conductivity k(x, y) [W/(m K)]</label>
        <div className="form-group">
          <label>Material Pattern</label>
          <select
            value={config.conductivity.pattern_type}
            onChange={(e) =>
              onChange({
                ...config,
                conductivity: {
                  ...config.conductivity,
                  pattern_type: e.target.value as ConductivityConfig['pattern_type'],
                },
              })
            }
          >
            <option value="uniform">Uniform Material</option>
            <option value="inclusion">Circular Inclusion Composite</option>
            <option value="gradient">Linear Gradient Interface</option>
          </select>
        </div>

        {config.conductivity.pattern_type === 'uniform' ? (
          <div className="form-group">
            <label>Uniform Conductivity [W/(m K)]</label>
            <input
              type="number"
              step="0.1"
              value={config.conductivity.uniform_value}
              onChange={(e) =>
                onChange({
                  ...config,
                  conductivity: { ...config.conductivity, uniform_value: parseFloat(e.target.value) || 1.0 },
                })
              }
            />
          </div>
        ) : (
          <div className="editor-form-grid-2">
            <div className="form-group">
              <label>Matrix k_bg [W/(m K)]</label>
              <input
                type="number"
                step="0.1"
                value={config.conductivity.inclusion_k_background}
                onChange={(e) =>
                  onChange({
                    ...config,
                    conductivity: { ...config.conductivity, inclusion_k_background: parseFloat(e.target.value) || 1.0 },
                  })
                }
              />
            </div>
            <div className="form-group">
              <label>Feature k_feat [W/(m K)]</label>
              <input
                type="number"
                step="0.5"
                value={config.conductivity.inclusion_k_feature}
                onChange={(e) =>
                  onChange({
                    ...config,
                    conductivity: { ...config.conductivity, inclusion_k_feature: parseFloat(e.target.value) || 5.0 },
                  })
                }
              />
            </div>
          </div>
        )}
      </div>

      {/* Volumetric Heat Source */}
      <div className="editor-card">
        <label className="section-title">Volumetric Heat Generation q(x, y) [W/m^3]</label>
        <div className="form-group">
          <label>Source Distribution</label>
          <select
            value={config.source.pattern_type}
            onChange={(e) =>
              onChange({
                ...config,
                source: { ...config.source, pattern_type: e.target.value as HeatSourceConfig['pattern_type'] },
              })
            }
          >
            <option value="uniform">Uniform Heat Source</option>
            <option value="gaussian_spot">Local Gaussian Hotspot</option>
          </select>
        </div>

        {config.source.pattern_type === 'uniform' ? (
          <div className="form-group">
            <label>Uniform Heat Generation [W/m^3]</label>
            <input
              type="number"
              step="5"
              value={config.source.uniform_value}
              onChange={(e) =>
                onChange({
                  ...config,
                  source: { ...config.source, uniform_value: parseFloat(e.target.value) || 0.0 },
                })
              }
            />
          </div>
        ) : (
          <div className="editor-form-grid-2">
            <div className="form-group">
              <label>Peak Heat Generation [W/m^3]</label>
              <input
                type="number"
                step="25"
                value={config.source.spot_peak_value}
                onChange={(e) =>
                  onChange({
                    ...config,
                    source: { ...config.source, spot_peak_value: parseFloat(e.target.value) || 100.0 },
                  })
                }
              />
            </div>
            <div className="form-group">
              <label>Spot Sigma Radius [m]</label>
              <input
                type="number"
                step="0.02"
                value={config.source.spot_sigma}
                onChange={(e) =>
                  onChange({
                    ...config,
                    source: { ...config.source, spot_sigma: parseFloat(e.target.value) || 0.15 },
                  })
                }
              />
            </div>
          </div>
        )}
      </div>

      {/* Boundary Conditions on 4 Edges */}
      <div className="editor-card">
        <label className="section-title">Edge Boundary Conditions</label>
        {(['left', 'right', 'bottom', 'top'] as const).map((side) => {
          const bc = config.boundary_conditions[side];
          return (
            <div key={side} className="bc-row">
              <span className="bc-side-label">{side.toUpperCase()}</span>
              <select
                value={bc.type}
                onChange={(e) => {
                  const updated = { ...config };
                  updated.boundary_conditions[side].type = e.target.value as 'dirichlet' | 'neumann';
                  onChange(updated);
                }}
              >
                <option value="dirichlet">Dirichlet (Temp [K])</option>
                <option value="neumann">Neumann (Flux [W/m^2])</option>
              </select>
              <input
                type="number"
                step={bc.type === 'dirichlet' ? '5' : '10'}
                value={bc.value}
                onChange={(e) => {
                  const updated = { ...config };
                  updated.boundary_conditions[side].value = parseFloat(e.target.value) || 0.0;
                  onChange(updated);
                }}
              />
            </div>
          );
        })}
      </div>

      {/* Validation Feedback */}
      {validationResult && (
        <div className={`validation-box ${validationResult.is_valid ? 'val-box-valid' : 'val-box-error'}`}>
          <div className="val-box-header">
            <strong>{validationResult.is_valid ? '✓ Validation Passed' : '⚠️ Validation Failed'}</strong>
            <span>Contrast: {validationResult.k_contrast.toFixed(2)}:1</span>
          </div>
          {validationResult.errors.map((err, idx) => (
            <p key={idx} className="val-error-text">• {err}</p>
          ))}
          {validationResult.warnings.map((warn, idx) => (
            <p key={idx} className="val-warn-text">• {warn}</p>
          ))}
        </div>
      )}

      {/* Action Buttons */}
      <div className="editor-actions-row">
        <button
          type="button"
          className="btn-validate"
          onClick={onValidate}
          disabled={isLoading}
        >
          Check Validation
        </button>
        <button
          type="button"
          className="btn-run-simulation"
          onClick={onRun}
          disabled={isLoading}
        >
          {isLoading ? 'Computing Solution...' : 'Run Simulation'}
        </button>
      </div>
    </div>
  );
};
