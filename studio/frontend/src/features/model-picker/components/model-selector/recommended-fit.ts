// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure helpers for the Recommended list: which formats to surface and whether a
// model fits the device. No React/DOM deps so they are easy to test.

const GGUF_SUFFIX_RE = /-GGUF(?:$|-)/i;
const MLX_RE = /-MLX(?:$|-)/i;

export function isGgufId(id: string, hintedIsGguf?: boolean): boolean {
  return Boolean(hintedIsGguf) || GGUF_SUFFIX_RE.test(id);
}

export function isMlxId(id: string): boolean {
  return MLX_RE.test(id);
}

// "mobile" build token (e.g. "gemma-4-E4B-it-qat-mobile-GGUF"); bounded so it
// never matches inside a longer word.
const MOBILE_RE = /(?:^|[-_/. ])mobile(?:$|[-_/. ])/i;

/** A mobile-targeted build, which we keep out of the Recommended list. */
export function isMobileVariant(id: string): boolean {
  return MOBILE_RE.test(id);
}

/** Recommended only surfaces ready-to-run local formats (GGUF / MLX). */
export function isRunnableRecommendedFormat(
  id: string,
  hintedIsGguf?: boolean,
): boolean {
  return isGgufId(id, hintedIsGguf) || isMlxId(id);
}

/** What Recommended is allowed to suggest: GGUF anywhere; on Mac also MLX and
 * safetensors (both now run locally there). GPU keeps GGUF-only recommendations. */
export function isRecommendableFormat(
  id: string,
  hintedIsGguf: boolean | undefined,
  isMac: boolean,
): boolean {
  if (isGgufId(id, hintedIsGguf)) return true;
  return isMac;
}

/** Format filter for the listing toggle. "safetensors" means anything that is
 * neither GGUF nor MLX. */
export type FormatFilter = "all" | "gguf" | "mlx" | "safetensors";

export function matchesFormatFilter(
  id: string,
  hintedIsGguf: boolean | undefined,
  filter: FormatFilter,
): boolean {
  switch (filter) {
    case "gguf":
      return isGgufId(id, hintedIsGguf);
    case "mlx":
      return isMlxId(id);
    case "safetensors":
      return !isGgufId(id, hintedIsGguf) && !isMlxId(id);
    default:
      return true;
  }
}

// First "<n>B" token in a repo id, e.g. "Qwen3-30B-A3B" -> 30 (MoE total),
// "gemma-4-E4B" -> 4. Digits must be separator-bounded so we never read "16"
// from "bf16" or the "2" in "Kimi-K2".
const PARAM_RE = /(?:^|[-_/. ])[eE]?(\d+(?:\.\d+)?)\s*[bB](?=$|[-_./ ])/;

/** Parameter count (absolute, e.g. 4e9) parsed from a repo id, or undefined
 * when the id has no size token (so callers can treat the size as unknown). */
export function paramsFromId(id: string): number | undefined {
  const match = PARAM_RE.exec(id);
  if (!match) return undefined;
  const billions = parseFloat(match[1]);
  return Number.isFinite(billions) && billions > 0 ? billions * 1e9 : undefined;
}

// Smallest practical GGUF/MLX quant (~Q2_K). The fit check asks whether a model
// can run at all, so it uses this rather than a default 4-bit size.
const MIN_QUANT_BYTES_PER_PARAM = 0.4;

/** Rough on-disk bytes for the smallest practical quant of `params` weights. */
export function estimateQuantBytes(params: number): number {
  return params * MIN_QUANT_BYTES_PER_PARAM;
}

/** A model fits when its on-disk size (or a precomputed VRAM estimate) is within
 * the device budget (0.7*GPU + 0.7*RAM). Unknown device means we cannot tell, so
 * treat it as fitting. Unknown size normally fits too, but Recommended passes
 * `requireKnown` so a model we cannot size (e.g. a huge GGUF with no metadata or
 * size token) is hidden rather than wrongly shown. */
export function fitsDevice(opts: {
  sizeBytes?: number;
  estimatedVramGb?: number;
  gpuGb?: number;
  systemRamGb?: number;
  requireKnown?: boolean;
}): boolean {
  const { sizeBytes, estimatedVramGb, gpuGb, systemRamGb, requireKnown } = opts;
  // Unified-memory hosts (Mac / no discrete GPU) report system RAM but no GPU,
  // so the budget must include RAM. Only an entirely unknown budget fits freely.
  const budgetGb = Math.max(0, gpuGb ?? 0) * 0.7 + Math.max(0, systemRamGb ?? 0) * 0.7;
  if (budgetGb <= 0) return true;
  if (sizeBytes && sizeBytes > 0) {
    return sizeBytes / 1024 ** 3 <= budgetGb;
  }
  if (estimatedVramGb && estimatedVramGb > 0) {
    return estimatedVramGb <= budgetGb;
  }
  return requireKnown ? false : true;
}

/** Fit predicate for one Hub listing row, shared by the chat model selector
 * and the Hub page "Fits on device" filter. GGUF repos: metadata size (actual
 * weights) or the smallest-quant estimate from the param count. Safetensors /
 * MLX repos: always the params-based smallest-quant estimate, matching the
 * VRAM badge's quantized-load assumption; their estimatedSizeBytes is the
 * full-precision checkpoint and would wrongly hide models the quantized load
 * path can run. Anything unsizable is hidden (requireKnown) so over-budget
 * models with no metadata don't slip through. An unknown device budget keeps
 * everything. */
export function hfModelFitsDevice(
  model: {
    id: string;
    totalParams?: number;
    estimatedSizeBytes?: number;
    isGguf?: boolean;
  },
  gpu: {
    memoryTotalGb: number;
    ggufMemoryTotalGb?: number;
    systemRamAvailableGb: number;
  },
): boolean {
  // "Can this run at all" assumes the quantized llama.cpp path, so budget
  // against the devices llama-server can use (can exceed the torch view on a
  // Vulkan build, e.g. a pre-ROCm card next to a ROCm one).
  const gpuGb = gpu.ggufMemoryTotalGb ?? gpu.memoryTotalGb;
  if (gpuGb <= 0 && gpu.systemRamAvailableGb <= 0) return true;
  const params = model.totalParams ?? paramsFromId(model.id);
  const quantBytes = params ? estimateQuantBytes(params) : undefined;
  const sizeBytes = isGgufId(model.id, model.isGguf)
    ? (model.estimatedSizeBytes ?? quantBytes)
    : (quantBytes ?? model.estimatedSizeBytes);
  return fitsDevice({
    sizeBytes,
    gpuGb,
    systemRamGb: gpu.systemRamAvailableGb,
    requireKnown: true,
  });
}
