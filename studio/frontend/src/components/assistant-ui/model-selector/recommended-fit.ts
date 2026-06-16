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

/** A model fits when its on-disk size (or a precomputed VRAM estimate) is within
 * the device budget (0.7*GPU + 0.7*RAM). Unknown device or size means we cannot
 * tell, so treat it as fitting (don't hide). */
export function fitsDevice(opts: {
  sizeBytes?: number;
  estimatedVramGb?: number;
  gpuGb?: number;
  systemRamGb?: number;
}): boolean {
  const { sizeBytes, estimatedVramGb, gpuGb, systemRamGb } = opts;
  if (!gpuGb || gpuGb <= 0) return true;
  const budgetGb = gpuGb * 0.7 + (systemRamGb ?? 0) * 0.7;
  if (sizeBytes && sizeBytes > 0) {
    return sizeBytes / 1024 ** 3 <= budgetGb;
  }
  if (estimatedVramGb && estimatedVramGb > 0) {
    return estimatedVramGb <= budgetGb;
  }
  return true;
}
