// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { VideoGenerationDefaults } from "./api";

/** Sentinel for deriving resolution from the keyframe aspect ratio. */
export const MATCH_SOURCE_RESOLUTION = -1;

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
