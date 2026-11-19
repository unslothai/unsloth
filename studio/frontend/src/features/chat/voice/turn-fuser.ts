// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export type TurnVerdict = "confirmed" | "today";

export interface TurnScores {
  pText: number | null;
  pAudio: number | null;
}

// Text uses JUDGE_SEND_THRESHOLD; audio is a placeholder until a labelled turn set retunes it.
export const TURN_THRESHOLDS = { text: 0.65, audio: 0.5 } as const;

export function fuseTurn(
  { pText, pAudio }: TurnScores,
  t: { text: number; audio: number } = TURN_THRESHOLDS,
): TurnVerdict {
  if (pAudio === null || pAudio < t.audio) return "today";
  // With no judge score, the turn counts as finished, as today's path already does.
  return pText === null || pText >= t.text ? "confirmed" : "today";
}
