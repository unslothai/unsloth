// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionConditioning } from "./api";

// Z-Image's range, which every family but Qwen-Image-2.1 keeps. ImageGenerationPresetParams
// enforces MIN_DIM and the transport ceiling on the persisted recipe.
export const MIN_DIM = 256;
export const MAX_DIM = 2048;

/** The output grid and bounds of the loaded model. */
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

/** The loaded model's limits, or the historical ones when the backend reports none. */
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

/** A (width, height) pair on the grid and inside both the side and the area bound. The pair is
 *  shrunk together, so its aspect ratio survives; each side alone would not. */
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

/** Output size with the source's aspect ratio at a resolution x resolution area. Mirrors the
 *  backend's match_source_size, so the size shown is the size generated: a source too elongated
 *  to keep its ratio inside [MIN_DIM, maxSide] keeps the short side at MIN_DIM and caps the long
 *  side, so the result is always a size the request accepts. */
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

/** A gallery record's size as the Create form can hold it. Scaled as a pair, so the recipe's
 *  aspect ratio survives; clamping each side alone would not.
 *
 *  Transform is the exception. img2img treats the requested size as a BOX it fits the upload
 *  inside (_fit_within, which never enlarges) rather than as the output size, so a side that was
 *  already in range has to be left alone: growing it moves the box and the re-run produces a
 *  different image than the one being restored. Clamping only the offending side reproduces the
 *  record exactly there, and the shape is the upload's anyway, not the form's. */
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
  // A ratio too extreme to fit both bounds at any scale falls back to per-side clamping.
  const scale = upTo > downTo ? 1 : Math.min(Math.max(1, upTo), downTo);
  return fitSize(width * scale, height * scale, limits);
}
