// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ActivationRecord } from "@/features/training";

// ── Thresholds ────────────────────────────────────────────────────────────────

export const DEAD_THRESHOLD = 0.01;
export const CONSTANT_CV_THRESHOLD = 0.05;
// Require at least this many steps before labelling a neuron "constant".
// With only 2–3 samples the CV is trivially low and produces false positives.
export const MIN_STEPS_FOR_CONSTANT = 5;

// A sample the backend stored as null (it was NaN or inf) is missing, not a measurement.
function isSample(v: number | null | undefined): v is number {
  return typeof v === "number" && Number.isFinite(v);
}

// ── Types ─────────────────────────────────────────────────────────────────────

export interface LayerStats {
  layer: string;
  deadPct: number;
  constantPct: number;
  totalChannels: number;
}

export interface OverlaySets {
  dead: Set<string>;      // "layerKey:channelIdx"
  constant: Set<string>;
  onsetDead: Set<string>;
}

export interface TrendPoint {
  stepIndex: number;
  step: number;
  deadPct: number;
  constantPct: number;
}

// ── computeLayerStats ─────────────────────────────────────────────────────────
// Returns per-layer dead/constant percentages across the given record slice.

export function computeLayerStats(records: ActivationRecord[]): LayerStats[] {
  if (records.length === 0) return [];

  const layerKeys = new Set<string>();
  for (const rec of records) {
    for (const k of Object.keys(rec.layers)) layerKeys.add(k);
  }

  const sorted = [...layerKeys].sort((a, b) => Number(a) - Number(b));

  return sorted.map((layerKey) => {
    const channelSeries = new Map<number, number[]>();
    for (const rec of records) {
      const layer = rec.layers[layerKey];
      if (!layer) continue;
      layer.mean_abs.forEach((val, idx) => {
        if (!channelSeries.has(idx)) channelSeries.set(idx, []);
        if (isSample(val)) channelSeries.get(idx)!.push(val);
      });
    }

    // A channel with no valid sample has nothing to judge, so it is not counted at all.
    const measured = [...channelSeries.values()].filter((vals) => vals.length > 0);
    const totalChannels = measured.length;
    let deadCount = 0;
    let constantCount = 0;

    for (const vals of measured) {
      const maxVal = Math.max(...vals);
      if (maxVal < DEAD_THRESHOLD) {
        deadCount++;
        continue;
      }
      if (vals.length >= MIN_STEPS_FOR_CONSTANT) {
        const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
        const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
        const cv = mean > 0 ? std / mean : 0;
        if (cv < CONSTANT_CV_THRESHOLD) constantCount++;
      }
    }

    return {
      layer: `L${layerKey}`,
      deadPct: totalChannels > 0 ? (deadCount / totalChannels) * 100 : 0,
      constantPct: totalChannels > 0 ? (constantCount / totalChannels) * 100 : 0,
      totalChannels,
    };
  });
}

// ── computeOverlaySets ────────────────────────────────────────────────────────
// Returns sets of "layerKey:channelIdx" strings for each overlay type,
// computed over records[0..stepIndex].

export function computeOverlaySets(
  records: ActivationRecord[],
  stepIndex: number,
): OverlaySets {
  const dead = new Set<string>();
  const constant = new Set<string>();
  const onsetDead = new Set<string>();

  if (records.length === 0) return { dead, constant, onsetDead };

  const slice = records.slice(0, stepIndex + 1);
  const firstRecord = records[0];
  const layerKeys = Object.keys(firstRecord.layers);

  for (const key of layerKeys) {
    const numCh = firstRecord.layers[key]?.mean_abs.length ?? 0;

    for (let ci = 0; ci < numCh; ci++) {
      const vals = slice
        .map((r) => r.layers[key]?.mean_abs[ci])
        .filter(isSample);
      if (vals.length === 0) continue;
      const firstVal = firstRecord.layers[key]?.mean_abs[ci];
      const maxVal = Math.max(...vals);
      const id = `${key}:${ci}`;

      if (maxVal < DEAD_THRESHOLD) {
        dead.add(id);
        if (isSample(firstVal) && firstVal >= DEAD_THRESHOLD) onsetDead.add(id);
        continue;
      }

      if (vals.length >= MIN_STEPS_FOR_CONSTANT) {
        const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
        const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length);
        const cv = mean > 0 ? std / mean : 0;
        if (cv < CONSTANT_CV_THRESHOLD) constant.add(id);
      }
    }
  }

  return { dead, constant, onsetDead };
}

// ── computeTrendData ──────────────────────────────────────────────────────────
// For each step s, compute cumulative avg dead% and constant% across layers.

export function computeTrendData(records: ActivationRecord[]): TrendPoint[] {
  return records.map((rec, si) => {
    const slice = records.slice(0, si + 1);
    const stats = computeLayerStats(slice);
    if (stats.length === 0) {
      return { stepIndex: si, step: rec.step, deadPct: 0, constantPct: 0 };
    }
    const avgDead = stats.reduce((s, l) => s + l.deadPct, 0) / stats.length;
    const avgConst = stats.reduce((s, l) => s + l.constantPct, 0) / stats.length;
    return { stepIndex: si, step: rec.step, deadPct: avgDead, constantPct: avgConst };
  });
}

// ── computeViewValues ─────────────────────────────────────────────────────────
// Returns per-channel float arrays for each layer, ready for the heatmap canvas.

export type ViewMode = "activations" | "gradients" | "lora_norms" | "delta" | "trend";

export function computeViewValues(
  records: ActivationRecord[],
  stepIndex: number,
  viewMode: ViewMode,
): Record<string, number[]> | null {
  const record = records[stepIndex];
  if (!record) return null;

  const layerKeys = Object.keys(record.layers).sort((a, b) => Number(a) - Number(b));

  switch (viewMode) {
    case "activations": {
      const result: Record<string, number[]> = {};
      for (const key of layerKeys) result[key] = record.layers[key].mean_abs;
      return result;
    }

    case "gradients": {
      if (!record.grad_norms) return null;
      const result: Record<string, number[]> = {};
      for (const key of layerKeys) {
        const val = record.grad_norms[key] ?? 0;
        const numCh = record.layers[key]?.mean_abs.length ?? 1;
        result[key] = Array<number>(numCh).fill(val);
      }
      return result;
    }

    case "lora_norms": {
      if (!record.lora_norms) return null;
      const result: Record<string, number[]> = {};
      for (const key of layerKeys) {
        const val = record.lora_norms[key] ?? 0;
        const numCh = record.layers[key]?.mean_abs.length ?? 1;
        result[key] = Array<number>(numCh).fill(val);
      }
      return result;
    }

    case "delta": {
      const result: Record<string, number[]> = {};
      if (stepIndex === 0) {
        for (const key of layerKeys)
          result[key] = Array<number>(record.layers[key].mean_abs.length).fill(0);
        return result;
      }
      const prev = records[stepIndex - 1];
      for (const key of layerKeys) {
        const curr = record.layers[key]?.mean_abs ?? [];
        const p = prev?.layers[key]?.mean_abs ?? [];
        // Missing on either side stays missing (NaN), which the heatmap draws as such.
        result[key] = curr.map((v, i) => {
          const before = p[i] === undefined ? 0 : p[i];
          return isSample(v) && isSample(before) ? Math.abs(v - before) : Number.NaN;
        });
      }
      return result;
    }

    case "trend": {
      const result: Record<string, number[]> = {};
      const slice = records.slice(0, stepIndex + 1);
      const N = slice.length;

      if (N < 2) {
        for (const key of layerKeys)
          result[key] = Array<number>(record.layers[key].mean_abs.length).fill(0);
        return result;
      }

      for (const key of layerKeys) {
        const numCh = record.layers[key]?.mean_abs.length ?? 0;
        const slopes: number[] = [];
        for (let ci = 0; ci < numCh; ci++) {
          // Fit over the valid samples only, at their own step positions.
          const pts = slice
            .map((r, i) => [i, r.layers[key]?.mean_abs[ci]] as const)
            .filter(([, v]) => isSample(v));
          if (pts.length < 2) {
            slopes.push(Number.NaN);
            continue;
          }
          const xMean = pts.reduce((s, [i]) => s + i, 0) / pts.length;
          const yMean = pts.reduce((s, [, v]) => s + v, 0) / pts.length;
          const xVar = pts.reduce((s, [i]) => s + (i - xMean) ** 2, 0);
          const cov = pts.reduce((s, [i, v]) => s + (i - xMean) * (v - yMean), 0);
          slopes.push(Math.abs(xVar > 0 ? cov / xVar : 0));
        }
        result[key] = slopes;
      }
      return result;
    }
  }
}
