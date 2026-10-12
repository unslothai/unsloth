// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { VideoGenerationDefaults, VideoStatus } from "./api";

export const MATCH_SOURCE_RESOLUTION = -1;

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
  const snap = (value: number) => Math.max(multiple, Math.round(value / multiple) * multiple);
  return [snap(width), snap(height)];
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
