/**
 * Status and decision badge rendering acceptance or conservative fallback details.
 */

import React from 'react';
import type { InferenceResult } from '../types';

interface StatusBadgeProps {
  result: InferenceResult;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ result }) => {
  const { status, execution_path, rejection } = result;

  let badgeClass = 'status-pill-accepted';
  let statusLabel = 'ACCEPTED';
  let summaryText = 'Surrogate prediction passed all physical safety gates.';

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

  return (
    <div className="status-decision-card">
      <div className="status-card-top">
        <div className="status-header-group">
          <span className={`status-pill ${badgeClass}`}>{statusLabel}</span>
          <span className="execution-path-tag">
            Path: <code>{execution_path}</code>
          </span>
        </div>
      </div>

      <p className="status-summary-text">{summaryText}</p>

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
