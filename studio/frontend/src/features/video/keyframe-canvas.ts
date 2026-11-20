// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { VideoGenerationDefaults, VideoStatus } from "./api";

/** The preset whose aspect ratio is nearest the picture's, compared in log space so 2:1 and 1:2 are
 *  equally far from square; ties keep the list order, so the family default wins. Mirrors the
 *  backend's first_frame_canvas. */
export function nearestPresetCanvas(
  aspectWidth: number,
  aspectHeight: number,
  presets: ReadonlyArray<readonly [number, number]>,
): [number, number] | null {
  if (aspectWidth <= 0 || aspectHeight <= 0 || presets.length === 0) return null;
  const target = Math.log(aspectWidth / aspectHeight);
  let best: readonly [number, number] | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const preset of presets) {
    if (preset[0] <= 0 || preset[1] <= 0) continue;
    const distance = Math.abs(Math.log(preset[0] / preset[1]) - target);
    if (distance < bestDistance) {
      best = preset;
      bestDistance = distance;
    }
  }
  return best ? [best[0], best[1]] : null;
}

/** Preview the backend-owned canvas rule from status defaults. Returns null when the family lacks a
 *  rule or the aspect ratio is unsupported. */
export function matchedCanvas(
  aspectWidth: number,
  aspectHeight: number,
  defaults: VideoGenerationDefaults | null | undefined,
): [number, number] | null {
  const shortEdge = defaults?.canvas_short_edge;
  const maxPixels = defaults?.canvas_max_pixels;
  const multiple = defaults?.resolution_multiple;
  if ((!shortEdge || !maxPixels) && defaults?.canvas_from_presets) {
    return nearestPresetCanvas(aspectWidth, aspectHeight, defaults.resolution_presets ?? []);
  }
  if (!shortEdge || !maxPixels || !multiple) return null;
  if (aspectWidth <= 0 || aspectHeight <= 0) return null;
  const ratio = aspectWidth / aspectHeight;
  if (ratio < 1 / 4 || ratio > 4) return null;
  let width = ratio >= 1 ? shortEdge * ratio : shortEdge;
  let height = ratio >= 1 ? shortEdge : shortEdge / ratio;
  const area = width * height;
  if (area > maxPixels) {
    const scale = Math.sqrt(maxPixels / area);
    width *= scale;
    height *= scale;
  }
  return snapCanvasWithinCap(width, height, multiple, maxPixels);
}

/** Each side to the nearest multiple; when that pushes the area past the cap, a side rounds down
 *  instead (largest area under the cap, then the aspect nearest the unsnapped one). Mirrors the
 *  backend's snap_canvas_within_cap. */
export function snapCanvasWithinCap(
  width: number,
  height: number,
  multiple: number,
  maxPixels: number,
): [number, number] {
  const down = (value: number) => Math.max(multiple, Math.floor(value / multiple) * multiple);
  const near = (value: number) => Math.max(multiple, Math.floor(value / multiple + 0.5) * multiple);
  const w = near(width);
  const h = near(height);
  if (w * h <= maxPixels) return [w, h];
  const target = Math.log(width / height);
  const fits = (
    [
      [down(width), h],
      [w, down(height)],
      [down(width), down(height)],
    ] as [number, number][]
  ).filter(([cw, ch]) => cw * ch <= maxPixels);
  if (fits.length === 0) return [down(width), down(height)];
  let best = fits[0];
  for (const candidate of fits.slice(1)) {
    const area = candidate[0] * candidate[1];
    const bestArea = best[0] * best[1];
    const distance = Math.abs(Math.log(candidate[0] / candidate[1]) - target);
    const bestDistance = Math.abs(Math.log(best[0] / best[1]) - target);
    if (area > bestArea || (area === bestArea && distance < bestDistance)) best = candidate;
  }
  return best;
}

/** The Default preset's size: the family's default_resolution when it is one of the presets, else the first. */
export function defaultVideoResolution(
  defaults: Pick<VideoGenerationDefaults, "default_resolution"> | null | undefined,
  presets: ReadonlyArray<readonly [number, number]>,
): [number, number] | null {
  const wanted = defaults?.default_resolution;
  if (wanted && presets.some(([w, h]) => w === wanted[0] && h === wanted[1])) {
    return [wanted[0], wanted[1]];
  }
  return presets[0] ? [presets[0][0], presets[0][1]] : null;
}

/** The size a family declares for its first render (MiniMax-H3: the reference templates' 864x480), when it is
 *  one of its presets. A family that declares none keeps the size the page already holds (`held`) when that is one
 *  of its trained sizes (null), else starts at its first preset: the page's 768x512 fallback or the previous
 *  family's size would otherwise be fitted to an untrained size. */
export function familyFirstRenderResolution(
  defaults: Pick<VideoGenerationDefaults, "default_resolution"> | null | undefined,
  presets: ReadonlyArray<readonly [number, number]>,
  held?: readonly [number, number] | null,
): [number, number] | null {
  const isPreset = (size: readonly [number, number]) =>
    presets.some(([w, h]) => w === size[0] && h === size[1]);
  const wanted = defaults?.default_resolution;
  if (wanted && isPreset(wanted)) return [wanted[0], wanted[1]];
  const first = presets[0];
  if (held && first && !isPreset(held)) return [first[0], first[1]];
  return null;
}

/** Whether staging a keyframe switches the size to "Match source". A family whose keyframes keep the chosen
 *  canvas (MiniMax-H3, like the reference workflow) leaves the size alone. */
export function keyframeSelectsMatchSource(
  defaults: Pick<VideoGenerationDefaults, "keyframe_keeps_canvas"> | null | undefined,
): boolean {
  return defaults?.keyframe_keeps_canvas !== true;
}

/** The width and height a request sends: the chosen size, or for "Match source" nothing (the backend derives it),
 *  except where an omitted size means the family default canvas, which then needs the matched size itself. */
export function requestCanvas(
  sourceSized: boolean,
  matched: readonly [number, number] | null,
  chosen: readonly [number, number],
  keepsCanvas: boolean,
): { width: number | undefined; height: number | undefined } {
  if (!sourceSized) return { width: chosen[0], height: chosen[1] };
  if (keepsCanvas && matched) return { width: matched[0], height: matched[1] };
  return { width: undefined, height: undefined };
}

/** Which keyframes the loaded checkpoint takes. Newer backends report each one; an older backend
 *  only has supports_keyframes, which meant MiniMax-H3's first-and/or-last pair. */
export function keyframeSupport(
  status: Pick<VideoStatus, "supports_keyframes" | "supports_first_frame" | "supports_last_frame">
    | null
    | undefined,
): { first: boolean; last: boolean } {
  const legacy = status?.supports_keyframes === true;
  return {
    first: typeof status?.supports_first_frame === "boolean" ? status.supports_first_frame : legacy,
    last: typeof status?.supports_last_frame === "boolean" ? status.supports_last_frame : legacy,
  };
}
