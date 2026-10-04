// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ABSide = "original" | "edited";

export interface ABState {
  side: ABSide;
  position: number;
  playing: boolean;
}

export const INITIAL_AB_STATE: ABState = {
  side: "edited",
  position: 0,
  playing: false,
};

const knownDuration = (duration: number | null | undefined): number | null =>
  typeof duration === "number" && Number.isFinite(duration) && duration > 0
    ? duration
    : null;

export function clampPosition(
  seconds: number,
  duration: number | null | undefined,
): number {
  const value = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const limit = knownDuration(duration);
  return limit === null ? value : Math.min(value, limit);
}

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

/** A position at or past the end restarts from the top rather than ending at once. */
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
