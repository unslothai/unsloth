// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Barge-in while the reply plays. The mic is echo-cancelled, but what leaks
// through still varies by speaker and room, so the gate sits above the leak it has
// measured during this reply as well as above a fixed minimum.

const BARGE_IN_RMS = 0.045;
const BARGE_OVER_BLEED = 3;
const BARGE_IN_MS = 220;

export interface BargeGate {
  bleed: number;
  loudMs: number;
}

export function createBargeGate(): BargeGate {
  return { bleed: -1, loudMs: 0 };
}

/** Feed one mic frame heard during playback; true once a burst has held long enough to cut in. */
export function stepBargeGate(gate: BargeGate, rms: number, frameMs: number): boolean {
  const threshold = Math.max(BARGE_IN_RMS, gate.bleed * BARGE_OVER_BLEED);
  if (rms >= threshold) {
    gate.loudMs += frameMs;
    return gate.loudMs >= BARGE_IN_MS;
  }
  gate.loudMs = 0;
  // Falls fast, climbs slowly: a quiet stretch resets it, and a burst of speech
  // can't raise its own bar before it has held long enough to count.
  if (gate.bleed < 0) gate.bleed = rms;
  else gate.bleed += (rms - gate.bleed) * (rms < gate.bleed ? 0.3 : 0.05);
  return false;
}
