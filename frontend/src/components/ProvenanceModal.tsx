/**
 * Modal displaying model provenance, checkpoint SHA-256 digests, and calibration metadata.
 */

import React from 'react';
import type { CapabilitiesInfo } from '../types';

interface ProvenanceModalProps {
  isOpen: boolean;
  onClose: () => void;
  capabilities: CapabilitiesInfo | null;
}

export const ProvenanceModal: React.FC<ProvenanceModalProps> = ({ isOpen, onClose, capabilities }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="modal-title">Model Provenance & Integrity Audit</h3>
          <button type="button" className="modal-close-btn" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="modal-body">
          <div className="modal-section">
            <h4>Frozen FNO-2D Checkpoints (Verified SHA-256)</h4>
            <div className="checkpoint-list">
              {capabilities?.ensemble_members.map((m) => (
                <div key={m.member_name} className="checkpoint-item">
                  <span className="checkpoint-name">📦 {m.member_name}</span>
                  <code className="checkpoint-sha">{m.sha256}</code>
                </div>
              ))}
            </div>
          </div>

          <div className="modal-section">
            <h4>Training Envelopes & Physics Bounds</h4>
            <ul className="provenance-list">
              <li>
                <strong>Max Thermal Conductivity Contrast:</strong> {capabilities?.training_domain_limits.max_conductivity_contrast_ratio}:1
              </li>
              <li>
                <strong>Max Volumetric Heat Generation:</strong> {capabilities?.training_domain_limits.max_volumetric_heat_source_w_m3} W/m³
              </li>
              <li>
                <strong>Trained Grid Discretization:</strong> 32x32 cell-centered 2D slab
              </li>
              <li>
                <strong>Provisional Spatial Mean Disagreement (τ_mean):</strong> {capabilities?.disagreement_screening_thresholds.spatial_mean_disagreement_tau_mean_k} K
              </li>
              <li>
                <strong>Provisional Peak Disagreement (τ_max):</strong> {capabilities?.disagreement_screening_thresholds.peak_cell_disagreement_tau_max_k} K
              </li>
            </ul>
          </div>

          <div className="modal-section">
            <h4>Independent Calibration & Verification Provenance</h4>
            <p className="provenance-text">
              Calibrated strictly on <code>artifacts/data/calibration_dataset.h5</code> (200 independent samples, seed offset &gt; 999,800).
              All 5 model checkpoints are immutable and verified on initialization against authoritative golden digests.
            </p>
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="btn-modal-close" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
