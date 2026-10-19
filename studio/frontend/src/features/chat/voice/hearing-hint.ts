// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * While a reply is under way but nothing is audible yet (thinking, generating, a voice
 * still synthesizing), the mic listens in barge-in mode, and "hearing" only lit once the
 * barge gate and the speech detector had both agreed: up to a second late, or never for a
 * soft voice. This lights it on the first loud frame instead and lets it go once the
 * voice has been quiet for `holdMs`, or the detector says it was not a voice. Only the orb
 * changes; whether the reply is cut is still the barge gate's call.
 */

/** Orb states where the reply makes no sound, so a loud mic frame can only be the user. */
const SILENT = new Set([
  "listening",
  "transcribing",
  "thinking",
  "generating",
  "synthesizing",
  "hearing",
]);

export function createHearingHint(holdMs = 500) {
  let on = false;
  let lastLoudAt = Number.NEGATIVE_INFINITY;
  return {
    /** Returns true or false when the orb's hearing flag should change, null otherwise. */
    step(loud: boolean, orbState: string | null, now: number): boolean | null {
      if (loud && orbState !== null && SILENT.has(orbState)) {
        lastLoudAt = now;
        if (!on) return (on = true);
        return null;
      }
      if (on && now - lastLoudAt > holdMs) {
        on = false;
        return false;
      }
      return null;
    },
    /** The detector heard no voice: drop the hint now. */
    reject(): boolean | null {
      if (!on) return null;
      on = false;
      return false;
    },
    /** The barge-in committed and owns the flag from here. */
    commit(): void {
      on = false;
    },
    get on() {
      return on;
    },
  };
}
