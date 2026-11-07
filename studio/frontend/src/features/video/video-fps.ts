// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { VideoGenerationDefaults } from "./api";

const FALLBACK_FPS = 24;

/** Frame rates the loaded family lets a request pick (validate_video_fps), or null when the rate is fixed. */
export function videoFpsOptions(
  defaults: VideoGenerationDefaults | null | undefined,
): number[] | null {
  const choices = defaults?.fps_choices ?? [];
  if (choices.length > 0) return [...new Set(choices)].sort((a, b) => a - b);
  const range = defaults?.fps_range;
  if (range && range.length === 2 && range[0] <= range[1]) {
    const out: number[] = [];
    for (let f = range[0]; f <= range[1]; f++) out.push(f);
    return out;
  }
  return null;
}

/** The rate a run uses: the picked one when the family offers it, else the family default. */
export function resolveVideoFps(
  defaults: VideoGenerationDefaults | null | undefined,
  picked: number | null,
): number {
  const fallback = defaults?.fps ?? FALLBACK_FPS;
  if (picked == null) return fallback;
  return videoFpsOptions(defaults)?.includes(picked) ? picked : fallback;
}

/** Frame counts nearest each target length at ``fps``, on the k * step + offset lattice and inside the
 *  family's frame window when the backend sends one, deduped. */
export function videoDurationOptions(
  targets: number[],
  fps: number,
  frameStep: number,
  frameOffset: number,
  minFrames?: number | null,
  maxFrames?: number | null,
): Array<{ frames: number; seconds: number }> {
  const seen = new Set<number>();
  const out: Array<{ frames: number; seconds: number }> = [];
  for (const t of targets) {
    const desired = t * fps;
    const k = Math.max(1, Math.round((desired - frameOffset) / frameStep));
    let frames = k * frameStep + frameOffset;
    if (minFrames != null && frames < minFrames) {
      frames = Math.ceil((minFrames - frameOffset) / frameStep) * frameStep + frameOffset;
    }
    if (maxFrames != null && frames > maxFrames) {
      frames = Math.floor((maxFrames - frameOffset) / frameStep) * frameStep + frameOffset;
    }
    if (seen.has(frames)) continue;
    seen.add(frames);
    out.push({ frames, seconds: frames / fps });
  }
  return out;
}

/** The duration option a restored clip maps to at its own rate: its frame count when the loaded family offers it,
 *  else the option nearest its length, so a clip from another family never sets a count this one refuses. */
export function restoredClipDuration(
  options: Array<{ frames: number; seconds: number }>,
  frames: number,
  fps: number,
): { frames: number; seconds: number } | null {
  return options.find((o) => o.frames === frames) ?? nearestDuration(options, frames / fps);
}

function nearestDuration(
  options: Array<{ frames: number; seconds: number }>,
  seconds: number,
): { frames: number; seconds: number } | null {
  let best: { frames: number; seconds: number } | null = null;
  for (const o of options) {
    if (best == null || Math.abs(o.seconds - seconds) < Math.abs(best.seconds - seconds)) best = o;
  }
  return best;
}

/** The frame count to keep when the duration options change. After a new family or frame rate it is the option
 *  nearest the length the user picked: the same count at another rate is another length. */
export function keptNumFrames(
  options: Array<{ frames: number; seconds: number }>,
  current: number,
  intentSeconds: number,
  reseat: boolean,
): number {
  if (!reseat && options.some((o) => o.frames === current)) return current;
  return nearestDuration(options, intentSeconds)?.frames ?? current;
}
