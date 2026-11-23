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

// A trained size cut to the family's grid sits up to about 2.5% off the ratio it was made for (832x480 against 16:9).
const PRESET_ASPECT_TOLERANCE = 0.03;

/** The Aspect ratio a size shows under. Like matchAspect (an exact ratio within 0.01), except a trained size
 *  takes the nearest named ratio within 3%, so 832x480 reads as 16:9 instead of Custom. */
export function videoSizeAspect(
  width: number,
  height: number,
  limits: VideoSizeLimits,
  ratios: Readonly<Record<string, readonly [number, number]>>,
): { key: string; portrait: boolean } {
  const portrait = height > width;
  const target = Math.max(width, height) / Math.min(width, height);
  const preset = isPresetSize(width, height, limits);
  let best: string | null = null;
  let bestOff = Number.POSITIVE_INFINITY;
  for (const [key, [a, b]] of Object.entries(ratios)) {
    const ratio = a / b;
    const off = preset ? Math.abs(target / ratio - 1) : Math.abs(target - ratio);
    if (off < (preset ? PRESET_ASPECT_TOLERANCE : 0.01) && off < bestOff) {
      best = key;
      bestOff = off;
    }
  }
  return { key: best ?? "custom", portrait };
}

/** The Trained sizes select's value: the shown size when it is a trained one, else "" (its placeholder). */
export function trainedSizeValue(
  size: readonly [number, number],
  limits: VideoSizeLimits,
  sourceSized = false,
): string {
  return !sourceSized && isPresetSize(size[0], size[1], limits) ? `${size[0]}x${size[1]}` : "";
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

/** The size a width (or height) edit gives under a locked named ratio. A trained size of that ratio with the edited
 *  side wins (832 under 16:9 gives 832x480, not 832x464), since presets are cut to the grid a few percent off the
 *  exact ratio; otherwise the other side follows the exact ratio, as before. */
export function lockedVideoSize(
  side: "width" | "height",
  value: number,
  ratio: readonly [number, number],
  portrait: boolean,
  limits: VideoSizeLimits,
): [number, number] {
  const wh = portrait ? ratio[1] / ratio[0] : ratio[0] / ratio[1];
  const preset = limits.presets.find(
    ([w, h]) =>
      (side === "width" ? w === value : h === value) &&
      h > w === portrait &&
      Math.abs(w / h / wh - 1) < PRESET_ASPECT_TOLERANCE,
  );
  if (preset) return [preset[0], preset[1]];
  return side === "width"
    ? [value, snapVideoDim(value / wh, limits)]
    : [snapVideoDim(value * wh, limits), value];
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
