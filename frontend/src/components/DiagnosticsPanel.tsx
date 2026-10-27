/**
 * Detailed scientific physical diagnostics panel for energy conservation,
 * algebraic residuals, and calibrated conformal intervals.
 */

import React from 'react';
import type { ConformalIntervals, PhysicalDiagnostics } from '../types';

interface DiagnosticsPanelProps {
  diagnostics: PhysicalDiagnostics | null;
  intervals: ConformalIntervals | null;
}

export const DiagnosticsPanel: React.FC<DiagnosticsPanelProps> = ({ diagnostics, intervals }) => {
  return (
    <div className="scientific-diagnostics-panel">
      {/* 1. Global Heat Balance */}
      <div className="diag-section">
        <h4 className="diag-section-heading">Global Heat Conservation</h4>
        {diagnostics ? (
          <div className="diag-grid">
            <div className="diag-item">
              <span className="diag-label">Volumetric Generation (Q_gen)</span>
              <span className="diag-value">{diagnostics.q_generated_total_w.toFixed(2)} W</span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Boundary Outflow (Q_out)</span>
              <span className="diag-value">{diagnostics.q_boundary_out_total_w.toFixed(2)} W</span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Signed Net Imbalance</span>
              <span className={`diag-value ${Math.abs(diagnostics.signed_net_imbalance_w) > 5.0 ? 'value-warn' : 'value-ok'}`}>
                {diagnostics.signed_net_imbalance_w > 0 ? '+' : ''}{diagnostics.signed_net_imbalance_w.toFixed(2)} W
              </span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Absolute Flux Imbalance</span>
              <span className="diag-value">{diagnostics.absolute_flux_imbalance_w.toFixed(2)} W</span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Relative Flux Error</span>
              <span className="diag-value">{(diagnostics.relative_flux_imbalance * 100).toFixed(3)}%</span>
            </div>
          </div>
        ) : (
          <p className="diag-unavailable-text">Conservation diagnostics unavailable.</p>
        )}

        {diagnostics?.boundary_heat_flow_out_w && (
          <div className="boundary-flow-breakdown">
            <span className="sub-heading">Boundary Face Outflows [W]:</span>
            <div className="boundary-pills">
              <span>Left: <strong>{diagnostics.boundary_heat_flow_out_w.left?.toFixed(1) ?? '0.0'} W</strong></span>
              <span>Right: <strong>{diagnostics.boundary_heat_flow_out_w.right?.toFixed(1) ?? '0.0'} W</strong></span>
              <span>Bottom: <strong>{diagnostics.boundary_heat_flow_out_w.bottom?.toFixed(1) ?? '0.0'} W</strong></span>
              <span>Top: <strong>{diagnostics.boundary_heat_flow_out_w.top?.toFixed(1) ?? '0.0'} W</strong></span>
            </div>
          </div>
        )}
      </div>

      {/* 2. Discrete PDE Residuals */}
      <div className="diag-section">
        <h4 className="diag-section-heading">Discrete PDE Residuals</h4>
        {diagnostics ? (
          <div className="diag-grid">
            <div className="diag-item">
              <span className="diag-label">Relative Residual ||r||_2 / ||b||_2</span>
              <span className={`diag-value ${diagnostics.pde_residual_relative > 0.05 ? 'value-warn' : 'value-ok'}`}>
                {diagnostics.pde_residual_relative.toExponential(3)}
              </span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Euclidean Norm ||r||_2</span>
              <span className="diag-value">{diagnostics.pde_residual_l2_w.toFixed(3)} W</span>
            </div>
            <div className="diag-item">
              <span className="diag-label">Max-Norm ||r||_inf</span>
              <span className="diag-value">{diagnostics.pde_residual_linf_w.toFixed(3)} W</span>
            </div>
          </div>
        ) : (
          <p className="diag-unavailable-text">PDE residuals unavailable.</p>
        )}
      </div>

      {/* 3. Conformal Prediction Intervals */}
      <div className="diag-section">
        <h4 className="diag-section-heading">Conformal Prediction Intervals</h4>
        {intervals ? (
          <div>
            <div className="diag-grid">
              <div className="diag-item">
                <span className="diag-label">Pointwise 95% Multiplier (q̂_pt)</span>
                <span className="diag-value">{intervals.pointwise_q_hat.toFixed(3)}x</span>
              </div>
              <div className="diag-item">
                <span className="diag-label">Simultaneous 95% Multiplier (q̂_sim)</span>
                <span className="diag-value">{intervals.simultaneous_q_hat.toFixed(3)}x</span>
              </div>
              <div className="diag-item">
                <span className="diag-label">Pointwise Mean Width</span>
                <span className="diag-value">±{intervals.pointwise_mean_width_k.toFixed(2)} K</span>
              </div>
              <div className="diag-item">
                <span className="diag-label">Simultaneous Mean Width</span>
                <span className="diag-value">±{intervals.simultaneous_mean_width_k.toFixed(2)} K</span>
              </div>
            </div>
            <p className="conformal-disclaimer-note">
              ℹ️ <strong>Coverage Note:</strong> {intervals.disclaimer}
            </p>
          </div>
        ) : (
          <p className="diag-unavailable-text">
            Conformal intervals not applicable to deterministic reference solve.
          </p>
        )}
      </div>
    </div>
  );
};
