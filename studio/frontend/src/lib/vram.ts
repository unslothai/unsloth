// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * VRAM estimation for model loading (4-bit quantization via bitsandbytes).
 *
 * Estimates the total driver-level VRAM (what nvidia-smi reports) to load a
 * model in 4-bit with Unsloth / bitsandbytes, to check it fits the GPU before
 * training.
 *
 * Formula: totalParams * 0.90 + 1.4 GB
 *
 * Calibrated against isolated Unsloth loads on RTX 5070 Ti (2026.2):
 *   Qwen2.5-0.5B  (0.49B) : est 1.8 vs actual 1.86 GB  (-3%)
 *   Llama-3.2-1B  (1.24B) : est 2.5 vs actual 2.54 GB   (-1%)
 *   Llama-3.2-3B  (3.21B) : est 4.3 vs actual 4.40 GB   (-2%)
 *   Llama-3.1-8B  (8.03B) : est 8.6 vs actual 8.14 GB   (+6%)
 *
 * Accuracy: within 3% for 0.5B-3B, within 6% for 8B.
 */

// Constants (exported for testing)

/**
 * Effective bytes per parameter for 4-bit weights at driver level. Raw bnb
 * 4-bit is ~0.5, but embedding/lm_head stay fp16 and bnb adds per-block
 * metadata, giving ~0.84-0.93 across architectures; 0.9 is the calibrated mid.
 */
export const BNB_4BIT_LOADING_BYTES = 0.9;

/**
 * Fixed overhead (GB) for the CUDA driver context and PyTorch runtime,
 * independent of model size. Measured at 1.34-1.46 GB; we use 1.4.
 */
export const LOADING_OVERHEAD_GB = 1.4;

export type VramFitStatus = "fits" | "tight" | "exceeds";

/**
 * Bytes per parameter at fp16/bf16 (LoRA, full FT). Theoretical (2 bytes = 16
 * bits); not yet calibrated, so real usage may run slightly higher (as 4-bit
 * is 0.9 vs 0.5).
 */
export const FP16_LOADING_BYTES = 2.0;

export type TrainingMethod = "qlora" | "lora" | "full" | "cpt";

function usesQuantizedLoading(method: TrainingMethod, modelId?: string): boolean {
  if (method === "qlora") return true;
  return method === "cpt" && (modelId ?? "").toLowerCase().includes("4bit");
}

/**
 * Estimate VRAM (GB) to load a model with Unsloth. Bytes/param by method:
 *   QLoRA: 4-bit bnb -> 0.90 (calibrated); LoRA/Full/CPT: fp16 -> 2.0
 *   (theoretical). Formula: totalParams * bytesPerParam + 1.4 GB overhead.
 */
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

/**
 * Check whether a model fits in available GPU VRAM.
 *   fits: <= 75%; tight: 75-100%; exceeds: > 100%.
 */
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
