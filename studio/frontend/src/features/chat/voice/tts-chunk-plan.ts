// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// How many pending sentences the TTS player sends in one synth request. Each
// request costs a fixed overhead on top of the audio itself (/audio/generate reports it
// as resolve_ms), so one short sentence per request
// leaves a gap after every clip. The first chunk
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

// The first clause of an unsettled first sentence can go out before the sentence ends.
export const CLAUSE_MIN_CHARS = 20;
export const CLAUSE_MAX_CHARS = 50;
// Only punctuation marks a clause: a cut before a bare "and" breaks the phrase mid-flow.
// Commas, semicolons and colons count only before a space, so "1,000" and "14:30" never cut,
// and a hyphen only between spaces; a sentence end means the sentence settles on its own.
const CLAUSE_BOUNDARY_RE = /[,;:](?=\s)|[—–]|\s-(?=\s)|[.!?]["'”’»)\]]*(?=\s)/g;

/**
 * Where to cut the head off `text` (the splitter's unsettled partial): after the first
 * `,` `;` `:` or dash that has at least CLAUSE_MIN_CHARS before it, with the head no longer
 * than CLAUSE_MAX_CHARS. null when there is no such cut.
 */
export function firstClauseCut(text: string): number | null {
  for (const match of text.matchAll(CLAUSE_BOUNDARY_RE)) {
    const at = match.index;
    if (/^[.!?]/.test(match[0])) return null;
    const cut = at + match[0].length;
    if (cut > CLAUSE_MAX_CHARS) return null;
    if (at >= CLAUSE_MIN_CHARS) return cut;
  }
  return null;
}

// A clause sent alone loses the pause the voice puts after a comma mid-sentence (Kokoro: 350-420 ms).
// Clause-first with 150 ms back rated as natural as the whole sentence (6 of 6); with none, 4.0 of 5.
export const CLAUSE_PAUSE_MS = 150;
const CLAUSE_END_RE = /(?:[,;:—–]|\s-)$/;

/** Silence to leave after `chunk` before the next clip: CLAUSE_PAUSE_MS when it ends on a clause mark. */
export function pauseAfterChunk(chunk: string): number {
  return CLAUSE_END_RE.test(chunk.trimEnd()) ? CLAUSE_PAUSE_MS : 0;
}

// A sentence that ended at a line break (a list item, a heading) has no stop of its own.
const SPOKEN_STOP_RE = /[.!?…:;,—–-]["'”’»)\]]*$/;

/** One synth request's text: sentences joined by spaces, each without a stop given a period so the voice pauses. */
export function spokenChunk(sentences: readonly string[]): string {
  return sentences
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => (SPOKEN_STOP_RE.test(s) ? s : `${s}.`))
    .join(" ");
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
