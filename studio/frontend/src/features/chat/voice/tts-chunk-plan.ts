// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

export const CLAUSE_MIN_CHARS = 20;
export const CLAUSE_MAX_CHARS = 50;
// Punctuation only: a cut before a bare "and" breaks the phrase.
const CLAUSE_BOUNDARY_RE = /[,;:](?=\s)|[—–]|\s-(?=\s)|[.!?]["'”’»)\]]*(?=\s)/g;

export function firstClauseCut(text: string): number | null {
  for (const match of text.matchAll(CLAUSE_BOUNDARY_RE)) {
    const at = match.index;
    if (/^[.!?]/.test(match[0])) return null;
    if (match[0] === "," && /\d$/.test(text.slice(0, at)) && /^,\s+\d{4}\b/.test(text.slice(at))) continue;
    const cut = at + match[0].length;
    if (cut > CLAUSE_MAX_CHARS) return null;
    if (at >= CLAUSE_MIN_CHARS) return cut;
  }
  return null;
}

// A lone clause loses 350-420 ms of Kokoro's pause; 150 ms sounds natural.
export const CLAUSE_PAUSE_MS = 150;
const CLAUSE_END_RE = /(?:[,;:—–]|\s-)$/;

export function pauseAfterChunk(chunk: string): number {
  return CLAUSE_END_RE.test(chunk.trimEnd()) ? CLAUSE_PAUSE_MS : 0;
}

const SPOKEN_STOP_RE = /[.!?…:;,—–-]["'”’»)\]]*$/;

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
