/**
 * Status and decision badge rendering acceptance or conservative fallback details.
 */

import React from 'react';
import type { InferenceResult, SimulationResponse } from '../types';

interface StatusBadgeProps {
  result: SimulationResponse | InferenceResult;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ result }) => {
  const isSimulationResponse = 'is_verified_numerical' in result;

  let badgeClass = 'status-pill-accepted';
  let statusLabel = 'ACCEPTED';
  let summaryText = 'Surrogate prediction passed all physical safety gates.';
  let executionPath = isSimulationResponse
    ? (result as SimulationResponse).mode
    : (result as InferenceResult).execution_path;

  if (isSimulationResponse) {
    const simRes = result as SimulationResponse;
    if (simRes.is_verified_numerical) {
      badgeClass = 'status-pill-accepted';
      statusLabel = '✓ VERIFIED NUMERICAL REFERENCE';
      summaryText = 'Conservative Finite-Volume solution with second-order spatial discretization, solved via direct SuperLU factorization.';
      executionPath = 'direct_fvm_solve';
    } else {
      badgeClass = 'status-pill-fallback';
      statusLabel = '⚠️ EXPERIMENTAL NEURAL SURROGATE';
      summaryText = 'Fast multi-resolution neural surrogate approximation. UNCERTIFIED: Not guaranteed to satisfy MaxAE <= 10 K.';
      executionPath = simRes.mode;
    }
  } else {
    const infRes = result as InferenceResult;
    const { status } = infRes;
    if (status === 'fallback_succeeded') {
      badgeClass = 'status-pill-fallback';
      statusLabel = 'FALLBACK SUCCEEDED';
      summaryText = 'Surrogate triggered conservative fallback. Verified solution computed via reference FVM.';
    } else if (status === 'fallback_required') {
      badgeClass = 'status-pill-required';
      statusLabel = 'FALLBACK REQUIRED (FAIL-CLOSED)';
      summaryText = 'Surrogate was rejected and solver fallback was disabled by policy. Prediction is null.';
    } else if (status === 'invalid_input') {
      badgeClass = 'status-pill-error';
      statusLabel = 'INVALID INPUT';
      summaryText = 'Physical problem configuration aborted input domain checks.';
    } else if (status === 'solver_failed') {
      badgeClass = 'status-pill-error';
      statusLabel = 'SOLVER FAILED (FAIL-CLOSED)';
      summaryText = 'Numerical reference solver aborted. No unverified surrogate prediction returned.';
    }
  }

  const rejection = !isSimulationResponse ? (result as InferenceResult).rejection : null;
  const warnings = isSimulationResponse ? (result as SimulationResponse).warnings : [];
  const disclaimers = isSimulationResponse ? (result as SimulationResponse).disclaimers : [];

  return (
    <div className="status-decision-card">
      <div className="status-card-top">
        <div className="status-header-group">
          <span className={`status-pill ${badgeClass}`}>{statusLabel}</span>
          <span className="execution-path-tag">
            Mode / Path: <code>{executionPath}</code>
          </span>
        </div>
      </div>

      <p className="status-summary-text">{summaryText}</p>

      {warnings && warnings.length > 0 && (
        <div className="rejection-audit-box" style={{ borderColor: 'rgba(245, 158, 11, 0.4)', backgroundColor: 'rgba(245, 158, 11, 0.1)' }}>
          <div className="rejection-title" style={{ color: '#fbbf24' }}>
            <span className="warning-icon">⚠️</span>
            <strong>Scientific Platform Warning:</strong>
          </div>
          {warnings.map((w, idx) => (
            <p key={idx} className="rejection-message" style={{ color: '#fef3c7', marginBottom: '4px' }}>
              {w}
            </p>
          ))}
        </div>
      )}

      {disclaimers && disclaimers.length > 0 && (
        <div style={{ marginTop: '8px', padding: '8px 12px', background: 'rgba(255, 255, 255, 0.03)', borderRadius: '4px', fontSize: '11px', color: '#94a3b8' }}>
          {disclaimers.map((d, idx) => (
            <div key={idx}>ℹ️ {d}</div>
          ))}
        </div>
      )}

      {rejection && rejection.code !== 'none' && (
        <div className="rejection-audit-box">
          <div className="rejection-title">
            <span className="warning-icon">⚠️</span>
            <strong>Rejection Reason: {rejection.code}</strong>
          </div>
          <p className="rejection-message">{rejection.message}</p>
          {rejection.triggering_metric && (
            <div className="rejection-metrics-grid">
              <span>Metric: <strong>{rejection.triggering_metric}</strong></span>
              <span>Observed: <strong>{rejection.observed_value?.toFixed(4)}</strong></span>
              <span>Threshold: <strong>{rejection.threshold_value?.toFixed(4)}</strong></span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
