// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { closestResolutionIndex } from "../generation-presets/preset-policy.ts";
import type { VideoGenerationDefaults } from "./api";

const MAX_VIDEO_SIDE = 2048;
const MAX_ASPECT = 4;

/** What validate_video_request_shape accepts for the loaded family. */
export interface VideoSizeLimits {
  multiple: number;
  minPixels: number;
  maxPixels: number;
  maxAspect: number;
  presets: Array<[number, number]>;
  // An older backend refuses anything off the presets.
  presetsOnly: boolean;
}

/** Mirrors the backend's video_custom_size_bounds for the pre-load fallback presets. */
function boundsFromPresets(presets: Array<[number, number]>) {
  const areas = presets.map(([w, h]) => w * h);
  const smallest = Math.min(...areas);
  const largest = Math.max(...areas);
  return {
    minPixels: Math.min(smallest, Math.floor(largest / 2)),
    maxPixels: largest,
    maxAspect: Math.max(
      MAX_ASPECT,
      ...presets.map(([w, h]) => Math.max(w, h) / Math.min(w, h)),
    ),
  };
}

export function videoSizeLimits(
  defaults: VideoGenerationDefaults | null | undefined,
  presets: Array<[number, number]>,
): VideoSizeLimits {
  const multiple = defaults?.resolution_multiple || 32;
  if (defaults?.custom_size_max_pixels && defaults.custom_size_min_pixels) {
    return {
      multiple,
      minPixels: defaults.custom_size_min_pixels,
      maxPixels: defaults.custom_size_max_pixels,
      maxAspect: defaults.custom_size_max_aspect || MAX_ASPECT,
      presets,
      presetsOnly: false,
    };
  }
  return {
    multiple,
    ...boundsFromPresets(presets),
    presets,
    presetsOnly: defaults != null,
  };
}

export function isPresetSize(
  width: number,
  height: number,
  limits: VideoSizeLimits,
): boolean {
  return limits.presets.some(([w, h]) => w === width && h === height);
}

export function isValidVideoSize(
  width: number,
  height: number,
  limits: VideoSizeLimits,
): boolean {
  if (isPresetSize(width, height, limits)) return true;
  if (limits.presetsOnly) return false;
  const m = limits.multiple;
  if (width <= 0 || height <= 0 || width % m || height % m) return false;
  if (width > MAX_VIDEO_SIDE || height > MAX_VIDEO_SIDE) return false;
  const area = width * height;
  if (area < limits.minPixels || area > limits.maxPixels) return false;
  return (
    Math.max(width, height) / Math.min(width, height) <= limits.maxAspect + 1e-9
  );
}

function closestPreset(
  width: number,
  height: number,
  limits: VideoSizeLimits,
): [number, number] {
  return (
    limits.presets[closestResolutionIndex(limits.presets, width, height)] ??
    limits.presets[0] ?? [768, 512]
  );
}

/** One side on the family grid and inside the request bounds. */
export function snapVideoDim(value: number, limits: VideoSizeLimits): number {
  const m = limits.multiple;
  if (!Number.isFinite(value) || value <= 0) return m;
  return Math.min(
    Math.floor(MAX_VIDEO_SIDE / m) * m,
    Math.max(m, Math.round(value / m) * m),
  );
}

/** The nearest size the backend accepts: kept when valid, else rescaled into the area band at the
 *  nearest ratio (clamped to the aspect cap), else the closest trained preset. */
export function fitVideoSize(
  width: number,
  height: number,
  limits: VideoSizeLimits,
): [number, number] {
  if (isValidVideoSize(width, height, limits)) return [width, height];
  if (limits.presetsOnly || width <= 0 || height <= 0)
    return closestPreset(width, height, limits);
  const m = limits.multiple;
  const ratio = Math.min(
    limits.maxAspect,
    Math.max(1 / limits.maxAspect, width / height),
  );
  const area = Math.min(
    limits.maxPixels,
    Math.max(limits.minPixels, width * height),
  );
  // Grid search, ratio weighted over area.
  let best: [number, number] | null = null;
  let bestScore = Number.POSITIVE_INFINITY;
  for (let h = m; h <= MAX_VIDEO_SIDE; h += m) {
    for (const w of [
      Math.floor((h * ratio) / m) * m,
      Math.ceil((h * ratio) / m) * m,
    ]) {
      if (!isValidVideoSize(w, h, limits)) continue;
      const score =
        Math.abs(Math.log(w / h / ratio)) * 4 +
        Math.abs(Math.log((w * h) / area));
      if (score < bestScore) {
        best = [w, h];
        bestScore = score;
      }
    }
  }
  if (best) return best;
  return closestPreset(width, height, limits);
}

const DIM_CANDIDATES = [
  384, 448, 480, 512, 544, 576, 640, 704, 720, 768, 832, 864, 896, 960, 1024,
  1088, 1152, 1216, 1280, 1344, 1408, 1536, 1664, 1792, 1920, 2048,
];

/** Sides offered in the width/height menus: common sizes on the grid that some valid frame can use. */
export function videoDimOptions(limits: VideoSizeLimits): number[] {
  const sides = new Set(limits.presets.flat());
  if (!limits.presetsOnly) {
    const longest = Math.sqrt(limits.maxPixels * limits.maxAspect);
    const shortest = Math.sqrt(limits.minPixels / limits.maxAspect);
    for (const n of DIM_CANDIDATES) {
      if (n % limits.multiple === 0 && n <= longest && n >= shortest)
        sides.add(n);
    }
  }
  return [...sides].sort((a, b) => a - b);
}
