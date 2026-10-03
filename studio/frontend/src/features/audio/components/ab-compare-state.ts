// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The A/B player's position rules: switching between the original and the edited clip keeps
// the place and whether it was playing, so the same moment can be heard both ways. Free of app
// imports so the node test runner can load it directly.

export type ABSide = "original" | "edited";

export interface ABState {
  side: ABSide;
  /** Seconds into the current side. */
  position: number;
  playing: boolean;
}

/** A new result opens on the edit, stopped at the start. */
export const INITIAL_AB_STATE: ABState = {
  side: "edited",
  position: 0,
  playing: false,
};

const knownDuration = (duration: number | null | undefined): number | null =>
  typeof duration === "number" && Number.isFinite(duration) && duration > 0
    ? duration
    : null;

/** A position kept inside 0..duration (only the lower bound when the duration is unknown). */
export function clampPosition(
  seconds: number,
  duration: number | null | undefined,
): number {
  const value = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const limit = knownDuration(duration);
  return limit === null ? value : Math.min(value, limit);
}

/** Switch sides at the same moment, clamped to the other clip's length, still playing if it was. */
export function switchSide(
  state: ABState,
  side: ABSide,
  newDuration: number | null | undefined,
): ABState {
  if (side === state.side) return state;
  return {
    side,
    position: clampPosition(state.position, newDuration),
    playing: state.playing,
  };
}

/** Move by `delta` seconds within the current clip. */
export function seekBy(
  state: ABState,
  delta: number,
  duration: number | null | undefined,
): ABState {
  return {
    ...state,
    position: clampPosition(state.position + delta, duration),
  };
}

/** Where to put the new clip once its length is known after a switch, and whether to resume.
 *  A position at or past the end restarts from the top rather than ending at once. */
export function resumeAfterLoad(
  state: ABState,
  mediaDuration: number | null | undefined,
): { time: number; play: boolean } {
  const limit = knownDuration(mediaDuration);
  const time = clampPosition(state.position, limit);
  return {
    time: limit !== null && time >= limit - 0.05 ? 0 : time,
    play: state.playing,
  };
}
