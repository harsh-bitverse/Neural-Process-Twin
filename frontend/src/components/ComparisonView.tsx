/**
 * Research comparison view comparing the neural surrogate directly against reference FVM.
 */

import React from 'react';
import type { ComparisonResult } from '../types';
import { FieldCanvas } from './FieldCanvas';

interface ComparisonViewProps {
  comparison: ComparisonResult;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({ comparison }) => {
  const {
    rmse_k,
    relative_l2_error_pct,
    max_absolute_error_k,
    surrogate_latency_ms,
    fvm_latency_ms,
    fvm_speedup_factor,
    surrogate_acceptance_status,
    evaluation_notice,
    temperature_difference_field,
  } = comparison;

  return (
    <div className="comparison-view-card">
      <div className="comparison-header">
        <div className="comparison-title-group">
          <h3 className="comparison-heading">Surrogate vs. Reference FVM Comparison</h3>
          <span className={`surrogate-eval-tag ${surrogate_acceptance_status === 'accepted' ? 'tag-accepted' : 'tag-rejected'}`}>
            Surrogate Status: {surrogate_acceptance_status.toUpperCase()}
          </span>
        </div>
      </div>

      <p className="comparison-notice">{evaluation_notice}</p>

      {/* Quantitative Error Metrics */}
      <div className="metrics-summary-grid">
        <div className="metric-box">
          <span className="metric-label">Root Mean Square Error (RMSE)</span>
          <span className="metric-number">
            {rmse_k !== null ? `${rmse_k.toFixed(3)} K` : 'N/A'}
          </span>
        </div>

        <div className="metric-box">
          <span className="metric-label">Relative L2 Error</span>
          <span className="metric-number">
            {relative_l2_error_pct !== null ? `${relative_l2_error_pct.toFixed(2)}%` : 'N/A'}
          </span>
        </div>

        <div className="metric-box">
          <span className="metric-label">Maximum Absolute Error (MaxAE)</span>
          <span className="metric-number">
            {max_absolute_error_k !== null ? `${max_absolute_error_k.toFixed(2)} K` : 'N/A'}
          </span>
        </div>

        <div className="metric-box metric-box-speedup">
          <span className="metric-label">Direct FVM vs. Surrogate Speedup</span>
          <span className="metric-number">
            {fvm_speedup_factor.toFixed(1)}x faster
          </span>
          <span className="metric-sublabel">
            FVM: {fvm_latency_ms.toFixed(1)} ms | Surrogate: {surrogate_latency_ms.toFixed(1)} ms
          </span>
        </div>
      </div>

      {/* Difference Heatmap Canvas */}
      {temperature_difference_field && (
        <div className="diff-canvas-section">
          <FieldCanvas
            data={temperature_difference_field}
            title="Temperature Difference Field ΔT = T_surr - T_fvm"
            unit="[K]"
            colormap="coolwarm"
            isDiverging={true}
          />
        </div>
      )}

      {/* Performance disclosure note */}
      <div className="disclosure-box">
        <strong>Scientific Reality Disclosure:</strong> Direct finite-volume solution is substantially faster 
        ({fvm_speedup_factor.toFixed(1)}x) than the 5-member neural ensemble for single 32x32 CPU queries. 
        The neural surrogate is designed for batch GPU evaluation and large-scale 3D domains where sparse 
        solvers scale superlinearly.
      </div>
    </div>
  );
};
