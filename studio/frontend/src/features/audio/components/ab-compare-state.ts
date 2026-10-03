// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The A/B player's switch rule, free of React and the DOM so the node test runner can load it.

/** Where the player was when the listener switched sides. */
export interface ABPlayback {
  /** Seconds into the side that was playing. */
  time: number;
  playing: boolean;
  /** That side's length in seconds; NaN or 0 while unknown. */
  duration: number;
}

/** Where the other side starts once its metadata loads: the same moment, clamped inside its
 *  length (a little before the end, so it does not finish at once), and playing again only if
 *  the first side was. */
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
  // A side shorter than where the other was has nothing left to play.
  const resume = prev.playing && time < nextDuration;
  return { seek, resume };
}
