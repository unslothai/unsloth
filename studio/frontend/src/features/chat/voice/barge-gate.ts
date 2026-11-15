// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


const BARGE_IN_RMS = 0.045;
// 3x needed a raised voice on speakers; the detector rejects bleed anyway.
const BARGE_OVER_BLEED = 2;
// A cough or dropped object decays inside ~300 ms and must not cut the reply.
const BARGE_IN_MS = 450;

export interface BargeGate {
  bleed: number;
  loudMs: number;
  holdMs: number;
}

export function createBargeGate(holdMs: number = BARGE_IN_MS): BargeGate {
  return { bleed: -1, loudMs: 0, holdMs };
}

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
  if (gate.bleed < 0) gate.bleed = rms;
  else gate.bleed += (rms - gate.bleed) * (rms < gate.bleed ? 0.3 : 0.05);
  return false;
}
