import type { LossHistoryItem, OutlierMode, SmoothedLossItem } from "./types";

export const CHART_SYNC_ID = "train-metrics-sync";
export const MAX_RENDER_POINTS = 800;
export const DEFAULT_VISIBLE_POINTS = 160;

export const placeholderEvalData = [
  { step: 0, loss: 2.8 },
  { step: 50, loss: 2.4 },
  { step: 100, loss: 2.0 },
  { step: 150, loss: 1.7 },
  { step: 200, loss: 1.5 },
];

export function toLog1p(value: number): number {
  const safe = Number.isFinite(value) ? Math.max(value, 0) : 0;
  return Math.log10(safe + 1);
}

export function fromLog1p(value: number): number {
  return Math.max(0, 10 ** value - 1);
}

export function formatMetric(value: number): string {
  if (!Number.isFinite(value)) return "0";
  if (value === 0) return "0";
  if (value >= 1000) return value.toFixed(0);
  if (value >= 1) return value.toFixed(2);
  return value.toExponential(2);
}

export function formatStepTick(value: number): string {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}k`;
  }
  return String(Math.round(value));
}

export function compressSeries<T>(data: T[], maxPoints: number): T[] {
  if (data.length <= maxPoints) {
    return data;
  }

  const stride = Math.ceil(data.length / maxPoints);
  return data.filter(
    (_item, index) => index % stride === 0 || index === data.length - 1,
  );
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function getDefaultWindowSize(totalSteps: number): number {
  if (totalSteps <= 1) {
    return Math.max(totalSteps, 1);
  }
  if (totalSteps <= DEFAULT_VISIBLE_POINTS) {
    return clamp(Math.floor(totalSteps * 0.6), 1, totalSteps);
  }
  return DEFAULT_VISIBLE_POINTS;
}

export function buildStepTicks(min: number, max: number, targetCount = 6): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }
  if (max <= min) {
    return [min, max];
  }

  const stepSize = Math.max(1, Math.ceil((max - min) / (targetCount - 1)));
  const ticks: number[] = [];
  let current = min;

  while (current < max) {
    ticks.push(current);
    current += stepSize;
  }

  ticks.push(max);
  return Array.from(new Set(ticks));
}

export function buildYDomain(values: number[]): [number, number] {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (finiteValues.length === 0) {
    return [0, 1];
  }

  const min = Math.min(...finiteValues);
  const max = Math.max(...finiteValues);

  if (min === max) {
    const base = Math.abs(min);
    const pad = base > 0 ? base * 0.08 : 0.1;
    return [min - pad, max + pad];
  }

  const pad = (max - min) * 0.12;
  return [min - pad, max + pad];
}

function getUpperPercentile(values: number[], mode: OutlierMode): number | null {
  if (mode === "none") return null;
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (finiteValues.length < 3) return null;

  const sorted = [...finiteValues].sort((a, b) => a - b);
  const q = mode === "p99" ? 0.99 : 0.95;
  const index = Math.max(
    0,
    Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)),
  );
  return sorted[index] ?? null;
}

export function applyOutlierCap(values: number[], mode: OutlierMode): number[] {
  const cap = getUpperPercentile(values, mode);
  if (cap == null) return values;
  return values.map((value) => Math.min(value, cap));
}

export function ema(data: LossHistoryItem[], alpha: number): SmoothedLossItem[] {
  if (data.length === 0) {
    return [];
  }

  let s = data[0].loss;
  return data.map((d) => {
    s = alpha * d.loss + (1 - alpha) * s;
    return { ...d, smoothed: +s.toFixed(4) };
  });
}
