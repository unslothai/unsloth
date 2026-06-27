// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// GPU-aware model-fit filtering: classifies whether a model fits the device
// and provides a filter predicate for the Hub page and model selector.

import type { GpuInfo } from "@/hooks/use-gpu-info";
import {
  activeOrEffectiveParamsFromId,
  estimateQuantBytes,
  paramsFromId,
} from "@/components/assistant-ui/model-selector/recommended-fit";

/** The three filter states exposed in the toolbar dropdown. */
export type GpuFitFilter = "all" | "fits" | "comfortable";

/** Per-model fit classification. */
export type GpuFitLevel = "comfortable" | "fits" | "oom";

/**
 * Classify whether a model fits the device.
 *
 * - "comfortable": estimated size ≤ 70% of GPU VRAM (runs fully in VRAM)
 * - "fits": estimated size ≤ 70% GPU + 70% system RAM (runs with CPU offload)
 * - "oom": exceeds both budgets
 *
 * Returns null when we can't determine the size (unknown → no badge).
 */
export function classifyGpuFit(opts: {
  totalParams?: number;
  estimatedSizeBytes?: number;
  repoId: string;
  gpu: GpuInfo;
}): GpuFitLevel | null {
  const { totalParams, estimatedSizeBytes, repoId, gpu } = opts;
  const gpuGb = gpu.memoryTotalGb;
  const ramGb = gpu.systemRamAvailableGb;
  if (gpuGb <= 0 && ramGb <= 0) return null; // no budget info

  // Active/effective model tokens (for example MoE A3B) describe runnable size
  // better than HF total-parameter metadata; otherwise prefer exact metadata.
  const activeOrEffectiveParams = activeOrEffectiveParamsFromId(repoId);
  const params = activeOrEffectiveParams ?? totalParams ?? paramsFromId(repoId);
  const sizeBytes =
    activeOrEffectiveParams
      ? estimateQuantBytes(activeOrEffectiveParams)
      : estimatedSizeBytes ?? (params ? estimateQuantBytes(params) : undefined);

  if (!sizeBytes || sizeBytes <= 0) return null; // can't determine

  const sizeGb = sizeBytes / 1024 ** 3;
  let comfortBudget: number;
  let fitBudget: number;

  if (!gpu.available || gpuGb <= 0) {
    // Unified memory system (no discrete GPU)
    comfortBudget = ramGb * 0.7;
    fitBudget = ramGb * 0.7;
  } else {
    // Discrete GPU
    comfortBudget = gpuGb * 0.7;
    fitBudget = gpuGb * 0.7 + ramGb * 0.7;
  }

  if (sizeGb <= comfortBudget) return "comfortable";
  if (sizeGb <= fitBudget) return "fits";
  return "oom";
}

/** Whether a row passes the given GPU fit filter. */
export function matchesGpuFitFilter(
  level: GpuFitLevel | null,
  filter: GpuFitFilter,
): boolean {
  if (filter === "all") return true;
  if (level === null) return false;
  if (filter === "comfortable") return level === "comfortable";
  // "fits" shows both comfortable and fits
  return level === "comfortable" || level === "fits";
}
