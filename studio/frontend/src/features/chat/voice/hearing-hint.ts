// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


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
    reject(): boolean | null {
      if (!on) return null;
      on = false;
      return false;
    },
    commit(): void {
      on = false;
    },
    get on() {
      return on;
    },
  };
}
