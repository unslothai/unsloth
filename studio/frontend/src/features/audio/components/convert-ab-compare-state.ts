// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ABPlayback {
  time: number;
  playing: boolean;
  duration: number;
}

/** Same moment on the other side, parked just before its end so it does not finish at once. */
export function nextPlayback(
  prev: ABPlayback,
  nextDuration: number,
): { seek: number; resume: boolean } {
  const time = Number.isFinite(prev.time) && prev.time > 0 ? prev.time : 0;
  if (!Number.isFinite(nextDuration) || nextDuration <= 0) {
    return { seek: time, resume: prev.playing };
  }
  const lastMoment = Math.max(0, nextDuration - 0.05);
  const seek = Math.min(time, lastMoment);
  const resume = prev.playing && time < nextDuration;
  return { seek, resume };
}
