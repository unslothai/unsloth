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

// Model-size extraction from repo id, matching the backend's 3-regex priority:
//   active params (MoE "A3B") > effective params (Gemma "E4B") > total ("8B").
// Bounded by separators so we never read "16" from "bf16" or "2" from "Kimi-K2".
// Examples: "Qwen3.5-35B-A3B" -> 3, "gemma-4-E4B" -> 4, "Llama-3-8B" -> 8.
const ACTIVE_PARAM_RE = /(?:^|[-_/. ])a(\d+(?:\.\d+)?)\s*[bB](?=$|[-_/. ])/i;
const EFFECTIVE_PARAM_RE = /(?:^|[-_/. ])e(\d+(?:\.\d+)?)\s*[bB](?=$|[-_/. ])/i;
const TOTAL_PARAM_RE = /(?:^|[-_/. ])(\d+(?:\.\d+)?)\s*[bB](?=$|[-_/. ])/;

function paramsFromMatch(match: RegExpExecArray | null): number | undefined {
  if (!match) return undefined;
  const billions = parseFloat(match[1]);
  return Number.isFinite(billions) && billions > 0
    ? billions * 1e9
    : undefined;
}

/** Active/effective parameter count parsed from a repo id, if it uses explicit
 * MoE/Gemma-style notation such as A3B or E4B. */
export function activeOrEffectiveParamsFromId(id: string): number | undefined {
  return (
    paramsFromMatch(ACTIVE_PARAM_RE.exec(id)) ??
    paramsFromMatch(EFFECTIVE_PARAM_RE.exec(id))
  );
}

/** Parameter count (absolute, e.g. 4e9) parsed from a repo id, or undefined
 * when the id has no size token (so callers can treat the size as unknown).
 * Prefers MoE active-param notation (A3B) over effective (E4B) over total. */
export function paramsFromId(id: string): number | undefined {
  return (
    activeOrEffectiveParamsFromId(id) ?? paramsFromMatch(TOTAL_PARAM_RE.exec(id))
  );
}

// Smallest practical GGUF/MLX quant (~Q2_K, low-bit). The fit check asks whether
// a model can run at all, so it uses this rather than a default 4-bit size; a
// user with a smaller device can still pick a low-bit variant.
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
