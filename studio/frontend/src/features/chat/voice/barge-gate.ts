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
  /** The last frame stepped was over the bar. */
  wasLoud: boolean;
}

export function createBargeGate(holdMs: number = BARGE_IN_MS): BargeGate {
  return { bleed: -1, loudMs: 0, holdMs, wasLoud: false };
}

export function stepBargeGate(
  gate: BargeGate,
  rms: number,
  frameMs: number,
): boolean {
  const threshold = Math.max(BARGE_IN_RMS, gate.bleed * BARGE_OVER_BLEED);
  gate.wasLoud = rms >= threshold;
  if (gate.wasLoud) {
    gate.loudMs += frameMs;
    return gate.loudMs >= gate.holdMs;
  }
  gate.loudMs = 0;
  if (gate.bleed < 0) gate.bleed = rms;
  else gate.bleed += (rms - gate.bleed) * (rms < gate.bleed ? 0.3 : 0.05);
  return false;
}

/** Backs over short pauses to the lead-in words; the gate needs an unbroken loud run. */
export function bargeOnset(
  loud: readonly boolean[],
  maxGap: number,
  min: number,
): number {
  const floor = Math.max(0, loud.length - min);
  const last = loud.lastIndexOf(true);
  if (last < 0) return floor;
  let first = last;
  let gap = 0;
  for (let i = last - 1; i >= 0; i--) {
    if (loud[i]) {
      first = i;
      gap = 0;
    } else if (++gap > maxGap) break;
  }
  return Math.max(0, Math.min(first - 1, floor));
}
