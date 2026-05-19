// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ActivationRecord } from "@/features/training/api/train-api";

// ── Thresholds ────────────────────────────────────────────────────────────────

export const DEAD_THRESHOLD = 0.01;
export const CONSTANT_CV_THRESHOLD = 0.05;

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
        channelSeries.get(idx)!.push(val);
      });
    }

    const totalChannels = channelSeries.size;
    let deadCount = 0;
    let constantCount = 0;

    for (const vals of channelSeries.values()) {
      const maxVal = Math.max(...vals);
      if (maxVal < DEAD_THRESHOLD) {
        deadCount++;
        continue;
      }
      if (vals.length >= 2) {
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
      const vals = slice.map((r) => r.layers[key]?.mean_abs[ci] ?? 0);
      const firstVal = firstRecord.layers[key]?.mean_abs[ci] ?? 0;
      const maxVal = Math.max(...vals);
      const id = `${key}:${ci}`;

      if (maxVal < DEAD_THRESHOLD) {
        dead.add(id);
        if (firstVal >= DEAD_THRESHOLD) onsetDead.add(id);
        continue;
      }

      if (vals.length >= 2) {
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
        result[key] = curr.map((v, i) => Math.abs(v - (p[i] ?? 0)));
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

      const tMean = (N - 1) / 2;
      const tVar = slice.reduce((s, _, i) => s + (i - tMean) ** 2, 0);

      for (const key of layerKeys) {
        const numCh = record.layers[key]?.mean_abs.length ?? 0;
        const slopes: number[] = [];
        for (let ci = 0; ci < numCh; ci++) {
          const ys = slice.map((r) => r.layers[key]?.mean_abs[ci] ?? 0);
          const yMean = ys.reduce((s, v) => s + v, 0) / N;
          const cov = ys.reduce((s, v, i) => s + (i - tMean) * (v - yMean), 0);
          slopes.push(Math.abs(tVar > 0 ? cov / tVar : 0));
        }
        result[key] = slopes;
      }
      return result;
    }
  }
}
