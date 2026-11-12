// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Inception Mercury diffusion streaming: every text of a provider turn (each `_diffusionReplace`
 * and the closing delta.content) is the whole turn so far, so it replaces the previous snapshot
 * instead of appending. Shared by the live stream and the reload recovery follower.
 */
export function createDiffusionReplace() {
  let snapshot: string | null = null;
  let start = 0;
  return {
    /** True once this turn has had a snapshot: every later text is one too. */
    get active(): boolean {
      return snapshot !== null;
    },
    /**
     * Where to cut the buffer before appending `text`, or null when `text` is the current
     * snapshot again (the closing delta.content), so the reply holds it once.
     * `firstStart` is where this turn's text begins; it defaults to the buffer's end.
     */
    cutAt(bufferLength: number, text: string, firstStart = bufferLength): number | null {
      if (snapshot === null) {
        start = Math.min(Math.max(firstStart, 0), bufferLength);
      } else if (snapshot === text) {
        return null;
      }
      snapshot = text;
      return start;
    },
    /** A tool round ended the turn; the next turn's snapshots start after everything so far. */
    endTurn(): void {
      snapshot = null;
    },
  };
}

/**
 * Where the turn being recovered begins in a checkpointed raw buffer, which may already end with a
 * saved snapshot: after the last carried part (a tool card), the last closed reasoning block and
 * the continued assistant partial, whichever is latest.
 */
export function recoveredDiffusionStart(
  raw: string,
  carriedAts: readonly number[],
  continuedPartial?: string,
): number {
  const thinkClose = raw.lastIndexOf("</think>");
  return Math.min(
    raw.length,
    Math.max(
      0,
      ...carriedAts,
      thinkClose === -1 ? 0 : thinkClose + "</think>".length,
      continuedPartial && raw.startsWith(continuedPartial) ? continuedPartial.length : 0,
    ),
  );
}
