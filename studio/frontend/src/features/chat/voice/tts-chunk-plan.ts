// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How many pending sentences the TTS player sends in one synth request. Each
// request costs a few seconds of fixed overhead on top of the audio itself, so
// one short sentence per request leaves a gap after every clip. The first chunk
// stays one sentence for a fast start; later ones merge sentences up to a
// character budget, never splitting a sentence.

export const TTS_CHUNK_TARGET_CHARS = 220;
export const TTS_CHUNK_MAX_CHARS = 320;
// A short chunk may wait for more sentences only this long while nothing plays.
export const TTS_CHUNK_WAIT_MS = 400;

export interface TtsChunkPlanInput {
  /** Sentences not yet requested, already stripped for speech, in order. */
  pending: string[];
  /** Nothing has been requested yet this stream. */
  first: boolean;
  /** The reply text is complete; no more sentences will arrive. */
  final: boolean;
  /** A clip is scheduled or audible. */
  playing: boolean;
  /** How long the head of `pending` has been held back waiting for more. */
  waitedMs: number;
}

/** Sentences to send now, from the head of `pending`; 0 means wait for more. */
export function planTtsChunk(input: TtsChunkPlanInput): number {
  const { pending, first, final, playing, waitedMs } = input;
  if (pending.length === 0) return 0;
  if (first) return 1;
  let take = 0;
  let chars = 0;
  for (const sentence of pending) {
    const add = chars === 0 ? sentence.length : sentence.length + 1;
    if (chars > 0 && (chars >= TTS_CHUNK_TARGET_CHARS || chars + add > TTS_CHUNK_MAX_CHARS)) break;
    chars += add;
    take++;
  }
  if (take < pending.length || chars >= TTS_CHUNK_TARGET_CHARS || final) return take;
  // The list ran dry under budget while the LLM is still writing: wait for
  // more, but an idle player must not sit on a sentence it could be speaking.
  if (!playing && waitedMs >= TTS_CHUNK_WAIT_MS) return take;
  return 0;
}
