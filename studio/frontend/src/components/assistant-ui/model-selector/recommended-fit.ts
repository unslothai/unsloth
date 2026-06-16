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

/** Recommended only surfaces ready-to-run local formats (GGUF / MLX). */
export function isRunnableRecommendedFormat(
  id: string,
  hintedIsGguf?: boolean,
): boolean {
  return isGgufId(id, hintedIsGguf) || isMlxId(id);
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

// First "<n>B" token in a repo id, e.g. "Qwen3-4B-GGUF" -> 4, "gpt-oss-20b" ->
// 20, "Qwen3-30B-A3B" -> 30 (MoE total), "gemma-4-E4B" -> 4 (effective-param
// "E" series). The digits must be bounded by a separator so we never read "16"
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

// Representative 4-bit weight size (~Q4_K_M / MLX 4bit). GGUF/MLX repos rarely
// expose safetensors metadata, so this is our fallback on-disk estimate.
const QUANT_BYTES_PER_PARAM = 0.6;

/** Rough on-disk bytes for a 4-bit quant of `params` weights. */
export function estimateQuantBytes(params: number): number {
  return params * QUANT_BYTES_PER_PARAM;
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
  if (!gpuGb || gpuGb <= 0) return true;
  const budgetGb = gpuGb * 0.7 + (systemRamGb ?? 0) * 0.7;
  if (sizeBytes && sizeBytes > 0) {
    return sizeBytes / 1024 ** 3 <= budgetGb;
  }
  if (estimatedVramGb && estimatedVramGb > 0) {
    return estimatedVramGb <= budgetGb;
  }
  return requireKnown ? false : true;
}
