/**
 * Interactive 2D scientific field canvas with colormap rendering and hover inspection.
 */

import React, { useEffect, useRef, useState } from 'react';
import type { ColormapName } from '../colormaps';
import { getColor, getColorCss } from '../colormaps';

interface FieldCanvasProps {
  data: number[][] | null;
  title: string;
  unit: string;
  colormap: ColormapName;
  isDiverging?: boolean;
  xBounds?: [number, number];
  yBounds?: [number, number];
  emptyMessage?: string;
}

export const FieldCanvas: React.FC<FieldCanvasProps> = ({
  data,
  title,
  unit,
  colormap,
  isDiverging = false,
  xBounds = [0, 1],
  yBounds = [0, 1],
  emptyMessage = 'Field data unavailable (enforcing fail-closed contract).',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoverInfo, setHoverInfo] = useState<{
    i: number;
    j: number;
    x: number;
    y: number;
    val: number;
  } | null>(null);

  // Compute min, max, and span
  let minVal = 0;
  let maxVal = 1;
  let hasValidData = false;

  if (data && data.length > 0 && data[0].length > 0) {
    let min = Infinity;
    let max = -Infinity;
    for (let j = 0; j < data.length; j++) {
      for (let i = 0; i < data[j].length; i++) {
        const v = data[j][i];
        if (Number.isFinite(v)) {
          if (v < min) min = v;
          if (v > max) max = v;
        }
      }
    }
    if (min !== Infinity && max !== -Infinity) {
      hasValidData = true;
      if (isDiverging) {
        const absMax = Math.max(Math.abs(min), Math.abs(max), 1e-6);
        minVal = -absMax;
        maxVal = absMax;
      } else {
        minVal = min;
        maxVal = max === min ? min + 1e-6 : max;
      }
    }
  }

  // Draw 2D heatmap on HTML5 Canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || !hasValidData) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const ny = data.length;
    const nx = data[0].length;
    const width = canvas.width;
    const height = canvas.height;

    const cellW = width / nx;
    const cellH = height / ny;

    ctx.clearRect(0, 0, width, height);

    // Render cells (row j=0 is at bottom y=y_min, row j=ny-1 is at top y=y_max)
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const val = data[j][i];
        let t = (val - minVal) / Math.max(maxVal - minVal, 1e-12);
        t = Math.max(0, Math.min(1, t));

        const { r, g, b } = getColor(colormap, t);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;

        const xPx = i * cellW;
        // Invert y so j=0 is bottom
        const yPx = height - (j + 1) * cellH;

        ctx.fillRect(Math.floor(xPx), Math.floor(yPx), Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    // Draw grid mesh border
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, width, height);
  }, [data, colormap, minVal, maxVal, hasValidData]);

  // Handle cursor hover for coordinate and scalar inspection
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !data || !hasValidData) return;

    const rect = canvas.getBoundingClientRect();
    const xRel = e.clientX - rect.left;
    const yRel = e.clientY - rect.top;

    const ny = data.length;
    const nx = data[0].length;

    const fracX = Math.max(0, Math.min(1, xRel / rect.width));
    const fracY = Math.max(0, Math.min(1, 1 - yRel / rect.height)); // j=0 at bottom

    const i = Math.min(nx - 1, Math.floor(fracX * nx));
    const j = Math.min(ny - 1, Math.floor(fracY * ny));

    const physX = xBounds[0] + (i + 0.5) * ((xBounds[1] - xBounds[0]) / nx);
    const physY = yBounds[0] + (j + 0.5) * ((yBounds[1] - yBounds[0]) / ny);
    const val = data[j][i];

    setHoverInfo({ i, j, x: physX, y: physY, val });
  };

  const handleMouseLeave = () => {
    setHoverInfo(null);
  };

  return (
    <div className="scientific-field-card">
      <div className="field-card-header">
        <div className="field-card-title-group">
          <span className="field-card-title">{title}</span>
          <span className="field-card-unit">{unit}</span>
        </div>
        {hasValidData && (
          <div className="field-card-stats">
            <span>Min: <strong>{minVal.toFixed(2)}</strong></span>
            <span>Max: <strong>{maxVal.toFixed(2)}</strong></span>
          </div>
        )}
      </div>

      <div className="field-canvas-container">
        {!hasValidData ? (
          <div className="field-empty-placeholder">
            <span className="empty-icon">⚠️</span>
            <p className="empty-message">{emptyMessage}</p>
          </div>
        ) : (
          <div className="canvas-wrapper">
            <div className="y-axis-label">y: {yBounds[1]} m</div>
            <canvas
              ref={canvasRef}
              width={384}
              height={384}
              className="field-canvas"
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
            />
            <div className="x-axis-footer">
              <span>x: {xBounds[0]} m</span>
              <span className="coord-bottom-indicator">y: {yBounds[0]} m</span>
              <span>x: {xBounds[1]} m</span>
            </div>
          </div>
        )}

        {/* Colorbar */}
        {hasValidData && (
          <div className="colorbar-container">
            <div
              className="colorbar-gradient"
              style={{
                background: `linear-gradient(to top, 
                  ${getColorCss(colormap, 0)}, 
                  ${getColorCss(colormap, 0.25)}, 
                  ${getColorCss(colormap, 0.5)}, 
                  ${getColorCss(colormap, 0.75)}, 
                  ${getColorCss(colormap, 1.0)})`,
              }}
            />
            <div className="colorbar-ticks">
              <span>{maxVal >= 1000 ? maxVal.toExponential(1) : maxVal.toFixed(2)}</span>
              <span>{((maxVal + minVal) / 2).toFixed(2)}</span>
              <span>{minVal <= -1000 ? minVal.toExponential(1) : minVal.toFixed(2)}</span>
            </div>
          </div>
        )}
      </div>

      {/* Hover Tooltip Readout */}
      {hoverInfo && (
        <div className="field-hover-readout">
          <span>Coords: <strong>({hoverInfo.x.toFixed(3)} m, {hoverInfo.y.toFixed(3)} m)</strong></span>
          <span>Cell: <strong>[j={hoverInfo.j}, i={hoverInfo.i}]</strong></span>
          <span>Value: <strong>{hoverInfo.val.toFixed(3)} {unit}</strong></span>
        </div>
      )}
    </div>
  );
};
