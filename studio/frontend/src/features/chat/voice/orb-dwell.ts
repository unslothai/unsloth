// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import type { VoiceOrbState } from "./orb-state";

export const ORB_DWELL_MS = 450;

const INSTANT: ReadonlySet<VoiceOrbState> = new Set(["hearing", "speaking"]);

export interface OrbDwellTimers {
  now: () => number;
  setTimer: (fn: () => void, ms: number) => unknown;
  clearTimer: (handle: unknown) => void;
}

// States hold ORB_DWELL_MS so sub-100 ms phases do not flash; "hearing", "speaking" and null land at once.
export interface OrbDwell {
  push: (state: VoiceOrbState | null) => void;
  shown: () => VoiceOrbState | null;
  dispose: () => void;
}

const browserTimers: OrbDwellTimers = {
  now: () => performance.now(),
  setTimer: (fn, ms) => setTimeout(fn, ms),
  clearTimer: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export function createOrbDwell(
  emit: (state: VoiceOrbState | null) => void,
  timers: OrbDwellTimers = browserTimers,
  dwellMs: number = ORB_DWELL_MS,
): OrbDwell {
  let displayed: VoiceOrbState | null = null;
  let shownAt = -Infinity;
  let pending: unknown = null;

  const cancel = () => {
    if (pending !== null) {
      timers.clearTimer(pending);
      pending = null;
    }
  };
  const show = (state: VoiceOrbState | null) => {
    cancel();
    displayed = state;
    // A blank orb has nothing to protect, so whatever comes next lands at once.
    shownAt = state === null ? -Infinity : timers.now();
    emit(state);
  };

  return {
    push(state) {
      if (state === displayed) {
        cancel();
        return;
      }
      const elapsed = timers.now() - shownAt;
      if (state === null || INSTANT.has(state) || elapsed >= dwellMs) {
        show(state);
        return;
      }
      cancel();
      pending = timers.setTimer(() => {
        pending = null;
        show(state);
      }, dwellMs - elapsed);
    },
    shown: () => displayed,
    dispose: cancel,
  };
}
