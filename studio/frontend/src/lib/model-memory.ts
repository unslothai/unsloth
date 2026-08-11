// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Splits a downloaded model's VRAM footprint into weights and what the current
 * settings add on top: KV cache at the context it will load with, plus an MTP
 * draft reserve when speculative decoding is on.
 *
 * They're kept apart because the fixes differ. Oversized weights need a smaller
 * quant; a total that only overflows once context is added needs a shorter
 * context or a quantized KV cache.
 *
 * Budget matches `gguf-fit.ts` and the backend's `_select_gpus`, so the bar and
 * the fit badge on a row can't contradict each other.
 */

import { VRAM_HEADROOM_RATIO } from "@/lib/gguf-fit";

const BYTES_PER_GB = 1024 ** 3;

export interface ModelMemoryInput {
  /** GGUF weights on disk. Null when the quant could not be resolved. */
  weightsBytes?: number | null;
  /** KV cache at the effective context length. */
  kvBytes?: number | null;
  /** MTP draft reserve; null for ngram (which is free) or a model without one. */
  specBytes?: number | null;
  /** Total VRAM of the GPU the model would load onto. */
  gpuGb?: number | null;
  /** Context the KV figure was measured at, for the per-token rate. */
  nCtx?: number | null;
}

/**
 * How close the total sits to the budget.
 *
 * Studio's live meters step at 70/90 (`resources-tab.tsx`), but they show usage
 * as it happens, where a sustained 70% is worth flagging. This is a reservation
 * you can see coming, so it holds the accent until 80.
 */
export type ModelMemoryPressure = "normal" | "high" | "critical";

/** Fill fraction (%) at which the bar leaves the accent colour. */
export const PRESSURE_HIGH_PCT = 80;
/** Fill fraction (%) at which the bar turns destructive. */
export const PRESSURE_CRITICAL_PCT = 90;

export type ModelMemoryStatus =
  /** Not enough information to draw anything. */
  | "unknown"
  /** Weights + settings both inside the budget. */
  | "fits"
  /** Weights fit alone, but the configured context pushes the total over. */
  | "context-exceeds"
  /** The weights alone are over budget; context settings cannot save it. */
  | "model-exceeds";

export interface ModelMemorySegments {
  status: ModelMemoryStatus;
  /** Usable VRAM the bar's full width represents. */
  budgetGb: number;
  modelGb: number;
  /** KV cache alone, at the effective context length. */
  kvGb: number;
  /** MTP draft reserve. Zero for ngram or a model with no drafter. */
  specGb: number;
  totalGb: number;
  /** Widths as percentages of the track, already clamped to sum <= 100. */
  modelPct: number;
  kvPct: number;
  specPct: number;
  /** KV cost per token, for judging what a shorter context would save. */
  kvBytesPerToken: number;
  /** Total as a percentage of budget; reads above 100 when over. */
  fillPct: number;
  pressure: ModelMemoryPressure;
}

const EMPTY: ModelMemorySegments = {
  status: "unknown",
  budgetGb: 0,
  modelGb: 0,
  kvGb: 0,
  specGb: 0,
  totalGb: 0,
  modelPct: 0,
  kvPct: 0,
  specPct: 0,
  kvBytesPerToken: 0,
  fillPct: 0,
  pressure: "normal",
};

function toGb(bytes?: number | null): number {
  return bytes && bytes > 0 ? bytes / BYTES_PER_GB : 0;
}

/**
 * Bar geometry and fit status for one row.
 *
 * Weights are required, KV is not: a row whose estimate hasn't arrived draws
 * its weights now and fills in the rest, instead of appearing late.
 */
export function computeModelMemory(
  input: ModelMemoryInput,
): ModelMemorySegments {
  const gpuGb = input.gpuGb ?? 0;
  const modelGb = toGb(input.weightsBytes);
  if (gpuGb <= 0 || modelGb <= 0) return EMPTY;

  const budgetGb = gpuGb * VRAM_HEADROOM_RATIO;
  const kvGb = toGb(input.kvBytes);
  const specGb = toGb(input.specBytes);
  const totalGb = modelGb + kvGb + specGb;

  const status: ModelMemoryStatus =
    modelGb > budgetGb
      ? "model-exceeds"
      : totalGb > budgetGb
        ? "context-exceeds"
        : "fits";

  // Clamp to the track, so an oversized model can't push the later segments
  // out of the row.
  const modelPct = Math.min(100, (modelGb / budgetGb) * 100);
  const kvPct = Math.min(100 - modelPct, (kvGb / budgetGb) * 100);
  const specPct = Math.min(100 - modelPct - kvPct, (specGb / budgetGb) * 100);

  // KV is linear in context above the SWA floor, so the marginal rate is what
  // says whether a shorter context would actually help.
  const nCtx = input.nCtx && input.nCtx > 0 ? input.nCtx : 0;
  const kvBytesPerToken =
    nCtx > 0 && input.kvBytes && input.kvBytes > 0 ? input.kvBytes / nCtx : 0;

  // Uncapped: widths clamp to the track, but pressure still needs to tell
  // "just full" from "twice over".
  const fillPct = (totalGb / budgetGb) * 100;
  const pressure: ModelMemoryPressure =
    fillPct >= PRESSURE_CRITICAL_PCT
      ? "critical"
      : fillPct >= PRESSURE_HIGH_PCT
        ? "high"
        : "normal";

  return {
    status,
    budgetGb,
    modelGb,
    kvGb,
    specGb,
    kvPct,
    specPct,
    kvBytesPerToken,
    totalGb,
    modelPct,
    fillPct,
    pressure,
  };
}

/** Compact label for a per-token KV rate, which lands in KB or MB. */
export function formatKvRate(bytes: number): string {
  if (bytes <= 0) return "0 KB";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb < 10 ? kb.toFixed(1) : Math.round(kb)} KB`;
  const mb = kb / 1024;
  return `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`;
}

/** Compact GB label for the bar's readout ("7.2 GB"). */
export function formatMemoryGb(gb: number): string {
  if (gb <= 0) return "0 GB";
  return `${gb < 10 ? gb.toFixed(1) : Math.round(gb)} GB`;
}
