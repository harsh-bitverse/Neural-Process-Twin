/**
 * Perceptually uniform scientific colormaps for physical field rendering.
 * Provides Viridis, Plasma, Inferno, Coolwarm (diverging), and Cividis.
 */

export type ColormapName = 'viridis' | 'plasma' | 'inferno' | 'coolwarm' | 'cividis';

export interface RGB {
  r: number;
  g: number;
  b: number;
}

// Sample points for Viridis (normalized [0, 1])
const VIRIDIS_STOPS: [number, RGB][] = [
  [0.0, { r: 68, g: 1, b: 84 }],
  [0.2, { r: 59, g: 82, b: 139 }],
  [0.4, { r: 33, g: 145, b: 140 }],
  [0.6, { r: 94, g: 201, b: 98 }],
  [0.8, { r: 178, g: 226, b: 54 }],
  [1.0, { r: 253, g: 231, b: 37 }],
];

// Sample points for Plasma
const PLASMA_STOPS: [number, RGB][] = [
  [0.0, { r: 13, g: 8, b: 135 }],
  [0.2, { r: 84, g: 2, b: 163 }],
  [0.4, { r: 156, g: 23, b: 158 }],
  [0.6, { r: 216, g: 87, b: 107 }],
  [0.8, { r: 248, g: 149, b: 64 }],
  [1.0, { r: 240, g: 249, b: 33 }],
];

// Sample points for Inferno
const INFERNO_STOPS: [number, RGB][] = [
  [0.0, { r: 0, g: 0, b: 4 }],
  [0.2, { r: 66, g: 10, b: 104 }],
  [0.4, { r: 147, g: 38, b: 103 }],
  [0.6, { r: 221, g: 81, b: 58 }],
  [0.8, { r: 252, g: 165, b: 10 }],
  [1.0, { r: 252, g: 255, b: 164 }],
];

// Sample points for Coolwarm (Diverging: Blue -> Gray -> Red)
const COOLWARM_STOPS: [number, RGB][] = [
  [0.0, { r: 59, g: 76, b: 192 }],
  [0.25, { r: 122, g: 157, b: 248 }],
  [0.5, { r: 221, g: 221, b: 221 }],
  [0.75, { r: 244, g: 154, b: 123 }],
  [1.0, { r: 180, g: 4, b: 38 }],
];

// Sample points for Cividis
const CIVIDIS_STOPS: [number, RGB][] = [
  [0.0, { r: 0, g: 32, b: 76 }],
  [0.25, { r: 42, g: 65, b: 106 }],
  [0.5, { r: 124, g: 123, b: 120 }],
  [0.75, { r: 194, g: 182, b: 103 }],
  [1.0, { r: 255, g: 234, b: 70 }],
];

function interpolateColor(stops: [number, RGB][], t: number): RGB {
  const clampedT = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (clampedT >= t0 && clampedT <= t1) {
      const alpha = (clampedT - t0) / Math.max(t1 - t0, 1e-12);
      return {
        r: Math.round(c0.r + alpha * (c1.r - c0.r)),
        g: Math.round(c0.g + alpha * (c1.g - c0.g)),
        b: Math.round(c0.b + alpha * (c1.b - c0.b)),
      };
    }
  }
  return stops[stops.length - 1][1];
}

export function getColor(colormap: ColormapName, t: number): RGB {
  switch (colormap) {
    case 'viridis':
      return interpolateColor(VIRIDIS_STOPS, t);
    case 'plasma':
      return interpolateColor(PLASMA_STOPS, t);
    case 'inferno':
      return interpolateColor(INFERNO_STOPS, t);
    case 'coolwarm':
      return interpolateColor(COOLWARM_STOPS, t);
    case 'cividis':
      return interpolateColor(CIVIDIS_STOPS, t);
  }
}

export function getColorCss(colormap: ColormapName, t: number): string {
  const { r, g, b } = getColor(colormap, t);
  return `rgb(${r}, ${g}, ${b})`;
}
