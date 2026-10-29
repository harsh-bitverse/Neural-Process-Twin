/**
 * Research comparison view comparing the neural surrogate directly against reference FVM.
 */

import React from 'react';
import type { ComparisonResult, DualModeComparisonResponse } from '../types';
import { FieldCanvas } from './FieldCanvas';

interface ComparisonViewProps {
  comparison: DualModeComparisonResponse | ComparisonResult;
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({ comparison }) => {
  const isDualMode = 'numerical_reference' in comparison;
  const {
    rmse_k,
    relative_l2_error_pct,
    max_absolute_error_k,
    surrogate_latency_ms,
    fvm_latency_ms,
    evaluation_notice,
    temperature_difference_field,
  } = comparison;

  const speedup = isDualMode
    ? (comparison as DualModeComparisonResponse).speedup_factor
    : (comparison as ComparisonResult).fvm_speedup_factor;

  const statusTag = isDualMode
    ? ((max_absolute_error_k ?? 999) <= 10.0 ? 'ACCEPTED (MaxAE ≤ 10 K)' : 'REJECTED (MaxAE > 10 K)')
    : (comparison as ComparisonResult).surrogate_acceptance_status.toUpperCase();

  const isAccepted = statusTag.includes('ACCEPTED');

  return (
    <div className="comparison-view-card">
      <div className="comparison-header">
        <div className="comparison-title-group">
          <h3 className="comparison-heading">Surrogate vs. Reference FVM Comparison</h3>
          <span className={`surrogate-eval-tag ${isAccepted ? 'tag-accepted' : 'tag-rejected'}`}>
            Surrogate Status: {statusTag}
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
          <span className={`metric-number ${max_absolute_error_k !== null && max_absolute_error_k > 10.0 ? 'value-warn' : ''}`}>
            {max_absolute_error_k !== null ? `${max_absolute_error_k.toFixed(2)} K` : 'N/A'}
          </span>
          <span className="metric-sublabel">Tolerance: ≤ 10.0 K</span>
        </div>

        <div className="metric-box metric-box-speedup">
          <span className="metric-label">Reference FVM vs Surrogate Speedup</span>
          <span className="metric-number">
            {speedup >= 1.0 ? `${speedup.toFixed(2)}x faster` : `${(1.0 / Math.max(speedup, 1e-6)).toFixed(2)}x slower`}
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
        <strong>Dual-Mode Scientific Platform Reality:</strong> Ground-truth error metrics are computed strictly
        from the actual Verified Numerical FVM execution. At 32×32 CPU execution, direct FVM is faster than neural ensembles.
        At 128×128 and 256×256, the single-pass neural surrogate achieves ~1.1x to 7.0x speedup over direct SuperLU solves,
        but remains uncertified for safety-critical deployment due to localized boundary errors exceeding 10 K.
      </div>
    </div>
  );
};
