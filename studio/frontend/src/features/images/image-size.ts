// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionConditioning } from "./api";

export const MIN_DIM = 256;
export const MAX_DIM = 2048;

export interface SizeLimits {
  multiple: number;
  maxSide: number;
  maxPixels: number;
}

export const DEFAULT_SIZE_LIMITS: SizeLimits = {
  multiple: 16,
  maxSide: MAX_DIM,
  maxPixels: MAX_DIM * MAX_DIM,
};

export function sizeLimitsFrom(
  conditioning: DiffusionConditioning | null | undefined,
): SizeLimits {
  if (!conditioning) return DEFAULT_SIZE_LIMITS;
  return {
    multiple: conditioning.dimension_multiple || DEFAULT_SIZE_LIMITS.multiple,
    maxSide: conditioning.max_output_side || DEFAULT_SIZE_LIMITS.maxSide,
    maxPixels: conditioning.max_output_pixels || DEFAULT_SIZE_LIMITS.maxPixels,
  };
}

export function snapDim(
  value: number,
  limits: SizeLimits = DEFAULT_SIZE_LIMITS,
): number {
  if (!Number.isFinite(value)) return 1024;
  const m = limits.multiple;
  const top = Math.floor(limits.maxSide / m) * m;
  const bottom = Math.ceil(MIN_DIM / m) * m;
  return Math.min(top, Math.max(bottom, Math.round(value / m) * m));
}

export function fitSize(
  width: number,
  height: number,
  limits: SizeLimits = DEFAULT_SIZE_LIMITS,
): { width: number; height: number } {
  let w = snapDim(width, limits);
  let h = snapDim(height, limits);
  if (w * h <= limits.maxPixels) return { width: w, height: h };
  const scale = Math.sqrt(limits.maxPixels / (w * h));
  const m = limits.multiple;
  w = Math.max(snapDim(MIN_DIM, limits), Math.floor((w * scale) / m) * m);
  h = Math.max(snapDim(MIN_DIM, limits), Math.floor((h * scale) / m) * m);
  return { width: w, height: h };
}

/** Mirrors the backend's match_source_size. */
export function matchSourceSize(
  sourceWidth: number,
  sourceHeight: number,
  resolution: number,
  limits: SizeLimits = DEFAULT_SIZE_LIMITS,
): { width: number; height: number } {
  const m = limits.multiple;
  const ratio = Math.max(1e-6, sourceWidth / Math.max(1, sourceHeight));
  let area = resolution * resolution;
  let w = m;
  let h = m;
  for (let i = 0; i < 64; i++) {
    w = Math.max(m, Math.round(Math.sqrt(area * ratio) / m) * m);
    h = Math.max(m, Math.round(Math.sqrt(area / ratio) / m) * m);
    if (Math.max(w, h) <= limits.maxSide && w * h <= limits.maxPixels) break;
    area *= 0.9;
  }
  const shortMin = Math.ceil(MIN_DIM / m) * m;
  const longMax = Math.floor(limits.maxSide / m) * m;
  if (Math.min(w, h) < shortMin) {
    let longSide = Math.round((shortMin * Math.max(ratio, 1 / ratio)) / m) * m;
    longSide = Math.min(Math.max(longSide, shortMin), longMax);
    while (longSide > shortMin && longSide * shortMin > limits.maxPixels)
      longSide -= m;
    return ratio >= 1
      ? { width: longSide, height: shortMin }
      : { width: shortMin, height: longSide };
  }
  return { width: w, height: h };
}

/**
 * Scaled as a pair to keep the aspect ratio. Transform clamps only the offending side, since
 * img2img treats the size as a fit-within box and growing it changes the output.
 */
export function restorableSize(
  width: number,
  height: number,
  workflow?: string | null,
  limits: SizeLimits = DEFAULT_SIZE_LIMITS,
): { width: number; height: number } {
  if (workflow === "img2img") {
    return { width: snapDim(width, limits), height: snapDim(height, limits) };
  }
  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return { width: snapDim(width, limits), height: snapDim(height, limits) };
  }
  const upTo = Math.max(MIN_DIM / width, MIN_DIM / height);
  const downTo = Math.min(
    limits.maxSide / width,
    limits.maxSide / height,
    Math.sqrt(limits.maxPixels / (width * height)),
  );
  const scale = upTo > downTo ? 1 : Math.min(Math.max(1, upTo), downTo);
  return fitSize(width * scale, height * scale, limits);
}
