// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LossHistoryItem, OutlierMode, SmoothedLossItem } from "./types";

export const CHART_SYNC_ID = "train-metrics-sync";
export const MAX_RENDER_POINTS = 800;
export const DEFAULT_VISIBLE_POINTS = 160;
export const CHART_CONTAINER_CLASS = "h-[220px] w-full";
export const DEFAULT_CHART_MARGIN = { top: 4, right: 8, bottom: 0, left: 4 };
export const DEFAULT_Y_AXIS_WIDTH = 45;
const TRAILING_ZEROES_RE = /\.?0+$/;
const NEGATIVE_ZERO_RE = /^-0$/;

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
  if (!Number.isFinite(value)) {
    return "0";
  }
  const abs = Math.abs(value);
  let decimals = 6;

  if (abs >= 1000) {
    decimals = 0;
  } else if (abs >= 100) {
    decimals = 2;
  } else if (abs >= 1) {
    decimals = 4;
  } else if (abs >= 0.01) {
    decimals = 5;
  } else if (abs >= 0.0001) {
    decimals = 6;
  } else {
    decimals = 8;
  }

  return value
    .toFixed(decimals)
    .replace(TRAILING_ZEROES_RE, "")
    .replace(NEGATIVE_ZERO_RE, "0");
}

export function formatAxisMetric(value: number): string {
  if (!Number.isFinite(value)) {
    return "0";
  }

  const abs = Math.abs(value);
  let decimals = 4;

  if (abs >= 1000) {
    decimals = 0;
  } else if (abs >= 100) {
    decimals = 1;
  } else if (abs >= 1) {
    decimals = 3;
  } else if (abs >= 0.01) {
    decimals = 4;
  } else {
    decimals = 5;
  }

  return value
    .toFixed(decimals)
    .replace(TRAILING_ZEROES_RE, "")
    .replace(NEGATIVE_ZERO_RE, "0");
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

export function buildStepTicks(
  min: number,
  max: number,
  targetCount = 6,
): number[] {
  if (!(Number.isFinite(min) && Number.isFinite(max))) {
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

function getUpperPercentile(
  values: number[],
  mode: OutlierMode,
): number | null {
  if (mode === "none") {
    return null;
  }
  const finiteValues = values.filter((value) => Number.isFinite(value));
  if (finiteValues.length < 3) {
    return null;
  }

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
  if (cap == null) {
    return values;
  }
  return values.map((value) => Math.min(value, cap));
}

export function ema(
  data: LossHistoryItem[],
  alpha: number,
): SmoothedLossItem[] {
  if (data.length === 0) {
    return [];
  }

  const values = data.map((point) => point.loss);
  const isConstant = values.every((value) => value === values[0]);

  let last = 0;
  let count = 0;

  return data.map((point) => {
    const next = point.loss;
    if (!Number.isFinite(next) || isConstant) {
      return { ...point, smoothed: next };
    }

    last = last * alpha + (1 - alpha) * next;
    count += 1;

    const debias = alpha === 1 ? 1 : 1 - alpha ** count;
    return { ...point, smoothed: last / debias };
  });
}
