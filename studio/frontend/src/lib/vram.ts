// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingMethod } from "@/types/training";

/**
 * 4-bit bnb load VRAM: totalParams * 0.90 + 1.4 GB, calibrated on RTX 5070 Ti (2026.2) within
 * -3% to +6% for 0.5B-8B models.
 */


/** Above raw 0.5 because embeddings/lm_head stay fp16 and bnb adds block metadata. */
export const BNB_4BIT_LOADING_BYTES = 0.9;

/** CUDA context and PyTorch runtime overhead, measured at 1.34-1.46 GB. */
export const LOADING_OVERHEAD_GB = 1.4;

export type VramFitStatus = "fits" | "tight" | "exceeds";

/** Theoretical, not calibrated; real usage may run slightly higher. */
export const FP16_LOADING_BYTES = 2.0;

function usesQuantizedLoading(
  method: TrainingMethod,
  modelId?: string,
): boolean {
  if (method === "qlora") return true;
  return method === "cpt" && (modelId ?? "").toLowerCase().includes("4bit");
}

export function estimateLoadingVram(
  totalParams: number,
  method: TrainingMethod = "qlora",
  modelId?: string,
): number {
  const bytesPerParam = usesQuantizedLoading(method, modelId)
    ? BNB_4BIT_LOADING_BYTES
    : FP16_LOADING_BYTES;
  const gb = (totalParams / 1e9) * bytesPerParam + LOADING_OVERHEAD_GB;
  return Math.round(gb * 10) / 10;
}

/** fits <= 75%; tight 75-100%; exceeds > 100%. */
export function checkVramFit(
  requiredGb: number,
  availableGb: number,
): VramFitStatus {
  if (availableGb <= 0) return requiredGb <= 0 ? "fits" : "exceeds";
  const ratio = requiredGb / availableGb;
  if (ratio <= 0.75) return "fits";
  if (ratio <= 1.0) return "tight";
  return "exceeds";
}

export interface ModelVramMapInput {
  id: string;
  totalParams?: number;
}

export interface ModelVramMapEntry {
  est: number;
  status: VramFitStatus | null;
}

export function buildModelVramMap(
  models: ModelVramMapInput[],
  method: TrainingMethod,
  gpu: { available: boolean; memoryTotalGb: number },
): Map<string, ModelVramMapEntry> {
  const map = new Map<string, ModelVramMapEntry>();
  for (const model of models) {
    if (!model.totalParams) {
      map.set(model.id, { est: 0, status: null });
      continue;
    }

    const est = estimateLoadingVram(model.totalParams, method, model.id);
    const status = gpu.available ? checkVramFit(est, gpu.memoryTotalGb) : null;
    map.set(model.id, { est, status });
  }
  return map;
}
