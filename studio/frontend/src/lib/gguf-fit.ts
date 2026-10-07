// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Mirrors llama_cpp.py `_select_gpus`, so the Hub card and chat picker cannot disagree. */

export type GgufFitClass = "fits" | "marginal" | "partial" | "ram" | "oom";

export interface GgufFitInput {
  gpuGb?: number;
  systemRamGb?: number;
  /** Absent falls back to the shared default, NOT the whole card. Keeps the badge in step with the
   * memory bar on the same row. */
  budgetFraction?: number;
  /** Only the reserve floor uses it; absent means one card. */
  gpuCount?: number;
}

/** Shared with the memory bar; the loader's `_CTX_FIT_VRAM_FRACTION`. 0.90 was tried and
 * reverted (#5106). Remaining badge/bar gaps come from the estimator, not this constant. */
export { DEFAULT_VRAM_BUDGET_FRACTION as VRAM_HEADROOM_RATIO } from "./memory/thresholds.ts";
import { DEFAULT_VRAM_BUDGET_FRACTION } from "./memory/thresholds.ts";
/** `_VRAM_FLOOR_RESERVE_MIB` (512 MiB) in llama_cpp.py, capped at 3% of the card. */
const VRAM_RESERVE_FLOOR_GB = 512 / 1024;
/** GGUF weights are file size; runtime activations add roughly this fraction. */
const ACTIVATIONS_RATIO = 0.15;
/** Flat KV/context allowance at a typical 4K window. */
const CONTEXT_OVERHEAD_GB = 1.0;
const RAM_OFFLOAD_USABLE_RATIO = 0.5;

export function requiredGgufMemoryGb(
  sizeBytes: number,
  contextOverheadGb = CONTEXT_OVERHEAD_GB,
): number {
  const sizeGb = sizeBytes / 1024 ** 3;
  return sizeGb * (1 + ACTIVATIONS_RATIO) + contextOverheadGb;
}

export function classifyGgufFit(
  sizeBytes: number,
  { gpuGb, systemRamGb, budgetFraction, gpuCount }: GgufFitInput,
): GgufFitClass {
  const required = requiredGgufMemoryGb(sizeBytes);
  if (!gpuGb || gpuGb <= 0) {
    const ramBudget = (systemRamGb ?? 0) * RAM_OFFLOAD_USABLE_RATIO;
    return required <= ramBudget ? "ram" : "oom";
  }
  // A 0 or non-finite fraction would score every quant as an overage.
  const fraction =
    typeof budgetFraction === "number" &&
    Number.isFinite(budgetFraction) &&
    budgetFraction > 0 &&
    budgetFraction <= 1
      ? budgetFraction
      : DEFAULT_VRAM_BUDGET_FRACTION;
  // Mirrors `_vram_usable_mib`: reserve is max((1 - frac) * total, min(512 MiB, 3%)), charged per
  // card since `_select_gpus` sums per device.
  const cards =
    typeof gpuCount === "number" && Number.isFinite(gpuCount) && gpuCount >= 1
      ? Math.floor(gpuCount)
      : 1;
  const reserveFloorGb =
    cards *
    Math.min(
      VRAM_RESERVE_FLOOR_GB,
      (1 - DEFAULT_VRAM_BUDGET_FRACTION) * (gpuGb / cards),
    );
  const budget = gpuGb - Math.max((1 - fraction) * gpuGb, reserveFloorGb);
  if (required <= budget) return "fits";
  // Raw card on purpose: "over budget but card-sized" is a warning tier, unreachable against `budget`.
  if (required <= gpuGb) return "marginal";
  // Once layers spill, the GPU contributes only its budget; the reserve is off-limits
  // (vram_budget_settings.py).
  const combined = budget + (systemRamGb ?? 0) * RAM_OFFLOAD_USABLE_RATIO;
  if (required <= combined) return "partial";
  return "oom";
}

/** Structural so this module stays below the features that own it. */
export interface GgufVariantSizes {
  size_bytes: number;
  download_size_bytes?: number;
}

/** The larger of `size_bytes` and `download_size_bytes` (companions like mmproj or a drafter).
 * Leans high; the offline path reports the total as weights and older backends as zero. */
export function ggufVariantFitSizeBytes(variant: GgufVariantSizes): number {
  return Math.max(variant.size_bytes, variant.download_size_bytes ?? 0);
}

export function classifyGgufVariantFit(
  variant: GgufVariantSizes,
  input: GgufFitInput,
): GgufFitClass {
  return classifyGgufFit(ggufVariantFitSizeBytes(variant), input);
}
