// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Training VRAM estimation for model fitness badges.
 *
 * Estimates total GPU VRAM needed to *train* a model with Unsloth — not just
 * load it.  This accounts for model weights, LoRA adapters, optimizer states,
 * gradients, activations, and CUDA overhead.
 *
 * The formulas mirror the backend fallback path in hardware.py and are
 * calibrated for default training params:
 *   batch=4, seq=2048, rank=16, adamw_8bit, unsloth gradient checkpointing
 *
 * For models in the search dropdown, `totalParams` (from HF safetensors
 * metadata) is used as input.  For MoE models this includes all expert
 * parameters, which can overestimate; once a model is selected the
 * authoritative architecture-based backend estimate is used instead (stored
 * in the training config store as `vramEstimate*Gb`).
 *
 * Constants:
 *   QUANT_4BIT_FACTOR  - fp16 → bnb 4-bit compression ratio (16/5 = 3.2×)
 *   LOADING_OVERHEAD_GB - CUDA driver + PyTorch runtime baseline (~1.4 GB)
 */

// ---------------------------------------------------------------------------
// Constants (exported for testing)
// ---------------------------------------------------------------------------

export const QUANT_4BIT_FACTOR = 16 / 5; // 3.2×
export const LOADING_OVERHEAD_GB = 1.4;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type VramFitStatus = "fits" | "tight" | "exceeds";

export type TrainingMethod = "qlora" | "lora" | "full";

// ---------------------------------------------------------------------------
// Estimation
// ---------------------------------------------------------------------------

/**
 * Estimate total training VRAM (GB) needed for a model given its parameter
 * count, using the backend's fallback formula.
 *
 * Where fp16Gb = totalParams × 2 bytes / 1e9:
 *   QLoRA  : fp16Gb / 3.2  +  fp16Gb × 0.04  +  fp16Gb × 0.15  +  1.4
 *   LoRA   : fp16Gb        +  fp16Gb × 0.04  +  fp16Gb × 0.15  +  1.4
 *   Full   : fp16Gb × 3.5  +  1.4
 *
 * The 0.04 term is LoRA adapter + optimizer overhead.
 * The 0.15 term is activations (with unsloth gradient checkpointing, batch=4).
 * The 3.5× factor for full FT covers weights + optimizer (Adam: ×3) + gradients.
 */
export function estimateTrainingVram(
  totalParams: number,
  method: TrainingMethod = "qlora",
): number {
  const fp16Gb = (totalParams * 2) / 1e9;

  let gb: number;
  if (method === "qlora") {
    gb = fp16Gb / QUANT_4BIT_FACTOR + fp16Gb * 0.04 + fp16Gb * 0.15 + LOADING_OVERHEAD_GB;
  } else if (method === "lora") {
    gb = fp16Gb + fp16Gb * 0.04 + fp16Gb * 0.15 + LOADING_OVERHEAD_GB;
  } else {
    // full fine-tuning
    gb = fp16Gb * 3.5 + LOADING_OVERHEAD_GB;
  }

  return Math.round(gb * 10) / 10;
}

/**
 * Check whether a model fits in the available GPU VRAM.
 *
 *   fits     -  uses <= 75% of available
 *   tight    -  uses 75-100% of available
 *   exceeds  -  uses > 100% of available
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

    const est = estimateTrainingVram(model.totalParams, method);
    const status = gpu.available ? checkVramFit(est, gpu.memoryTotalGb) : null;
    map.set(model.id, { est, status });
  }
  return map;
}
