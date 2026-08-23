// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const GGUF_SUFFIX_RE = /-GGUF(?:$|-)/i;
// First "<n>B" token in a repo id; digits must be separator-bounded so we never
// read "16" from "bf16" or the "2" in "Kimi-K2".
const PARAM_RE = /(?:^|[-_/. ])[eE]?(\d+(?:\.\d+)?)\s*[bB](?=$|[-_./ ])/;
// Smallest practical GGUF/MLX quant (~Q2_K). The fit check asks whether a model
// can run at all, so it uses this rather than a default 4-bit size.
const MIN_QUANT_BYTES_PER_PARAM = 0.4;

export function isGgufId(id: string, hintedIsGguf?: boolean): boolean {
  return Boolean(hintedIsGguf) || GGUF_SUFFIX_RE.test(id);
}

export function paramsFromId(id: string): number | undefined {
  const match = PARAM_RE.exec(id);
  if (!match) return undefined;
  const billions = Number.parseFloat(match[1]);
  return Number.isFinite(billions) && billions > 0 ? billions * 1e9 : undefined;
}

export function estimateQuantBytes(params: number): number {
  return params * MIN_QUANT_BYTES_PER_PARAM;
}

export interface ModelFitCandidate {
  id: string;
  totalParams?: number;
  estimatedSizeBytes?: number;
  curatedSizeBytes?: number;
  isGguf?: boolean;
}

export interface ModelFitGpu {
  memoryTotalGb: number;
  systemRamAvailableGb: number;
  budgetKnown?: boolean;
}

/** A model fits when its on-disk size (or a precomputed VRAM estimate) is within
 * the device budget (0.7*GPU + 0.7*RAM; unified-memory hosts report RAM with no
 * GPU, so the budget must include both). An entirely unknown budget fits freely.
 * An unknown size fits too unless `requireKnown` hides what cannot be sized. */
export function fitsDevice(options: {
  sizeBytes?: number;
  estimatedVramGb?: number;
  gpuGb?: number;
  systemRamGb?: number;
  budgetKnown?: boolean;
  requireKnown?: boolean;
}): boolean {
  const {
    sizeBytes,
    estimatedVramGb,
    gpuGb,
    systemRamGb,
    budgetKnown,
    requireKnown,
  } = options;
  const budgetGb =
    Math.max(0, gpuGb ?? 0) * 0.7 + Math.max(0, systemRamGb ?? 0) * 0.7;
  if (budgetGb <= 0) return !budgetKnown;
  if (sizeBytes && sizeBytes > 0) {
    return sizeBytes / 1024 ** 3 <= budgetGb;
  }
  if (estimatedVramGb && estimatedVramGb > 0) {
    return estimatedVramGb <= budgetGb;
  }
  return !requireKnown;
}

/** Fit predicate for one Hub listing row, shared by the chat model selector and
 * the Hub page "Fits on device" filter. GGUF repos: metadata size (actual
 * weights) or the smallest-quant estimate from the param count. Safetensors /
 * MLX repos: always the params-based smallest-quant estimate; their
 * estimatedSizeBytes is the full-precision checkpoint and would wrongly hide
 * models the quantized load path can run. `curatedSizeBytes` outranks both.
 * Anything still unsizable is hidden (requireKnown) so over-budget models with
 * no metadata don't slip through. */
export function hfModelFitsDevice(
  model: ModelFitCandidate,
  gpu: ModelFitGpu,
): boolean {
  if (
    gpu.memoryTotalGb <= 0 &&
    gpu.systemRamAvailableGb <= 0 &&
    !gpu.budgetKnown
  ) {
    return true;
  }
  const params = model.totalParams ?? paramsFromId(model.id);
  const quantBytes = params ? estimateQuantBytes(params) : undefined;
  const sizeBytes =
    model.curatedSizeBytes ??
    (isGgufId(model.id, model.isGguf)
      ? (model.estimatedSizeBytes ?? quantBytes)
      : (quantBytes ?? model.estimatedSizeBytes));
  return fitsDevice({
    sizeBytes,
    gpuGb: gpu.memoryTotalGb,
    systemRamGb: gpu.systemRamAvailableGb,
    budgetKnown: gpu.budgetKnown,
    requireKnown: true,
  });
}

export function hubModelFitsDevice(
  model: ModelFitCandidate,
  gpu: ModelFitGpu,
  inferenceGpu: ModelFitGpu,
): boolean {
  return hfModelFitsDevice(
    model,
    isGgufId(model.id, model.isGguf) ? inferenceGpu : gpu,
  );
}
