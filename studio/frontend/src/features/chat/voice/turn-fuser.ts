// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure: Smart Turn can only end a turn sooner. Anything short of both scores keeps today's path.

export type TurnVerdict = "confirmed" | "today";

export interface TurnScores {
  /** The turn judge's probability that the text is a finished turn; null when it did not answer
   * (its models are not installed by default), and then the audio decides alone. */
  pText: number | null;
  /** Smart Turn's end-of-turn probability on the audio; null when it is off or failed. */
  pAudio: number | null;
}

// text is the judge's send threshold (JUDGE_SEND_THRESHOLD); audio is a placeholder until a labelled turn set retunes it.
export const TURN_THRESHOLDS = { text: 0.65, audio: 0.5 } as const;

export function fuseTurn(
  { pText, pAudio }: TurnScores,
  t: { text: number; audio: number } = TURN_THRESHOLDS,
): TurnVerdict {
  if (pAudio === null || pAudio < t.audio) return "today";
  // With no judge, today's path sends the turn as finished too; Smart Turn only makes it sooner.
  return pText === null || pText >= t.text ? "confirmed" : "today";
}
