// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Barge-in while the reply plays. The mic is echo-cancelled, but what leaks
// through still varies by speaker and room, so the gate sits above the leak it has
// measured during this reply as well as above a fixed minimum.

const BARGE_IN_RMS = 0.045;
// Over the reply's bleed into the mic. 3x needed a raised voice on speakers; the detector rejects bleed anyway.
const BARGE_OVER_BLEED = 2;
// Long enough that a cough or a dropped object (one impulsive burst that decays
// inside ~300 ms) does not cut the reply; "wait, stop" still clears it.
const BARGE_IN_MS = 450;

export interface BargeGate {
  bleed: number;
  loudMs: number;
  /** How long a burst must hold; shorter when a speech detector confirms it afterwards. */
  holdMs: number;
}

export function createBargeGate(holdMs: number = BARGE_IN_MS): BargeGate {
  return { bleed: -1, loudMs: 0, holdMs };
}

/** Feed one mic frame heard during playback; true once a burst has held long enough to cut in. */
export function stepBargeGate(
  gate: BargeGate,
  rms: number,
  frameMs: number,
): boolean {
  const threshold = Math.max(BARGE_IN_RMS, gate.bleed * BARGE_OVER_BLEED);
  if (rms >= threshold) {
    gate.loudMs += frameMs;
    return gate.loudMs >= gate.holdMs;
  }
  gate.loudMs = 0;
  // Falls fast, climbs slowly: a quiet stretch resets it, and a burst of speech
  // can't raise its own bar before it has held long enough to count.
  if (gate.bleed < 0) gate.bleed = rms;
  else gate.bleed += (rms - gate.bleed) * (rms < gate.bleed ? 0.3 : 0.05);
  return false;
}
