// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Z-Image's range, which ImageGenerationPresetParams also enforces on the persisted recipe.
export const MIN_DIM = 256;
export const MAX_DIM = 2048;

export function snapDim(value: number): number {
  if (!Number.isFinite(value)) return 1024;
  return Math.min(MAX_DIM, Math.max(MIN_DIM, Math.round(value / 16) * 16));
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
): { width: number; height: number } {
  if (workflow === "img2img") {
    return { width: snapDim(width), height: snapDim(height) };
  }
  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return { width: snapDim(width), height: snapDim(height) };
  }
  const upTo = Math.max(MIN_DIM / width, MIN_DIM / height);
  const downTo = Math.min(MAX_DIM / width, MAX_DIM / height);
  // A ratio too extreme to fit both bounds at any scale falls back to per-side clamping.
  const scale = upTo > downTo ? 1 : Math.min(Math.max(1, upTo), downTo);
  return { width: snapDim(width * scale), height: snapDim(height * scale) };
}
