// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Single source of truth for "will this GGUF fit on the user's GPU?". Mirrors the
 * backend's GPU selection (llama_cpp.py `_select_gpus`): 90% of GPU memory for
 * weights + KV cache, else `--fit` CPU offload. One formula so the Hub card and
 * chat picker can't disagree.
 */

export type GgufFitClass = "fits" | "marginal" | "partial" | "ram" | "oom";

export interface GgufFitInput {
  gpuGb?: number;
  systemRamGb?: number;
}

/** Fraction of GPU VRAM treated as usable, matching the backend's 0.90 budget. */
const VRAM_HEADROOM_RATIO = 0.9;
/** GGUF weights are file size; runtime activations add roughly this fraction. */
const ACTIVATIONS_RATIO = 0.15;
/** Flat KV/context allowance at a typical 4K window. */
const CONTEXT_OVERHEAD_GB = 1.0;
/** Conservative share of system RAM usable for CPU offload. */
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
  { gpuGb, systemRamGb }: GgufFitInput,
): GgufFitClass {
  const required = requiredGgufMemoryGb(sizeBytes);
  if (!gpuGb || gpuGb <= 0) {
    const ramBudget = (systemRamGb ?? 0) * RAM_OFFLOAD_USABLE_RATIO;
    return required <= ramBudget ? "ram" : "oom";
  }
  const budget = gpuGb * VRAM_HEADROOM_RATIO;
  if (required <= budget) return "fits";
  if (required <= gpuGb) return "marginal";
  const combined = gpuGb + (systemRamGb ?? 0) * RAM_OFFLOAD_USABLE_RATIO;
  if (required <= combined) return "partial";
  return "oom";
}
