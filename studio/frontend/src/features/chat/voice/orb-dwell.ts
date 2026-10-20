// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Minimum time each orb state stays on screen.
 *
 * deriveOrbState is exact about the loop's phase, and some phases are over in
 * under 100 ms (a short transcription, a thinking model that answers at once).
 * Shown raw, the icon and colour flash for a frame or two, which reads as a
 * glitch. So a displayed state holds for ORB_DWELL_MS before the next raw state
 * replaces it, with the exceptions that matter to the ear: "hearing" and
 * "speaking" always land at once, since they track something the user can hear,
 * and the loop stopping (null) clears at once.
 *
 * Pure and timer-injected so the ordering is unit-testable without a browser.
 */

import type { VoiceOrbState } from "./orb-state";

export const ORB_DWELL_MS = 450;

const INSTANT: ReadonlySet<VoiceOrbState> = new Set(["hearing", "speaking"]);

export interface OrbDwellTimers {
  now: () => number;
  setTimer: (fn: () => void, ms: number) => unknown;
  clearTimer: (handle: unknown) => void;
}

export interface OrbDwell {
  /** Feed the newest raw state; emits the displayed state now or once the dwell is up. */
  push: (state: VoiceOrbState | null) => void;
  /** What is on screen right now. */
  shown: () => VoiceOrbState | null;
  /** Drop any pending switch. The displayed state is left as is. */
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
        // Back to what is on screen: any switch away from it is off.
        cancel();
        return;
      }
      const elapsed = timers.now() - shownAt;
      if (state === null || INSTANT.has(state) || elapsed >= dwellMs) {
        show(state);
        return;
      }
      // Only the newest raw state is ever pending; an older one is already stale.
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
