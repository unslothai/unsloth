// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

/**
 * Hearing the end of a turn. At a short pause the loop asks a small audio model whether the
 * speaker sounds finished; a yes ends the turn after a shorter silence than the fixed wait.
 * It can only shorten that wait: no answer, a late answer or a "not finished" all leave it as it was.
 */

// Silence before the loop asks. Mirrors ASK_AFTER_MS in turn_audiocpp_sidecar.py.
export const TURN_ASK_AFTER_MS = 250;
// A turn that sounded finished still ends only after this much silence. A pause after a clause can
// sound finished too: ending at the answer (about 350 ms) split a two-clause test sentence at its
// 570 ms comma pause in 4 turns of 5. The answer and the transcript are both in by then.
export const TURN_END_AFTER_MS = 700;

/**
 * The early transcript ends on a question mark: the turn is over even when the end-of-turn model
 * disagrees. It scored "When did the first moon landing happen, and what time was it?" at p = 0.06,
 * which left the reply waiting for the full silence hang.
 */
export function readsAsQuestion(text: string): boolean {
  return /[?？؟]["”’)\]]*$/.test(text.trim());
}
// Speech heard while a reply waits to start is still being recorded for up to the silence wait after
// it goes quiet. Starting the reply in that gap answered half a sentence and restarted for the rest.
export const TURN_CONTINUE_GRACE_MS = 1200;
// The model reads at most this much, from the end of the turn: the rest is not worth uploading.
export const TURN_CLIP_MAX_S = 8;

/** Whether a reply should still wait: the speaker is talking, or went quiet too recently to be done. */
export function stillInTurn(hearing: boolean, lastHeardAt: number, now: number): boolean {
  return hearing || (lastHeardAt > 0 && now - lastHeardAt < TURN_CONTINUE_GRACE_MS);
}

/** The end of `samples` the model reads. */
export function turnClipTail(samples: Float32Array, sampleRate: number): Float32Array {
  const keep = Math.round(TURN_CLIP_MAX_S * sampleRate);
  return samples.length > keep ? samples.subarray(samples.length - keep) : samples;
}

export interface TurnEndAnswer {
  finished: boolean;
  /** Null when the model is not asked for this language. */
  p: number | null;
}

/**
 * The clip the model hears: from where the utterance begins to the last frame that was not silent.
 * The quiet end of the last word stays in (cutting at the last loud frame loses finished turns), and
 * the silence after it stays out (it makes unfinished turns sound finished).
 */
export function turnClipSpan(
  voiced: readonly boolean[],
  notSilent: readonly boolean[],
): { start: number; end: number } | null {
  let start = voiced.indexOf(true);
  if (start < 0) return null;
  let end = voiced.lastIndexOf(true);
  while (start > 0 && notSilent[start - 1]) start--;
  while (end < voiced.length - 1 && notSilent[end + 1]) end++;
  return { start, end };
}

// A last word no sentence ends on, whatever mark the transcriber put after it.
const OPEN_ENDINGS = new Set(
  "and or but because cause if while than the a an my your our their um uh er erm".split(" "),
);
const CLOSING_MARK = /[.?!。？！؟।]["'”’)\]]*$/u;

/** Whether the transcriber wrote a closing mark here: proof that it writes them at all. */
export function hasClosingMark(transcript: string): boolean {
  return CLOSING_MARK.test(transcript.trim());
}

/**
 * Whether a transcript reads as cut off mid-sentence: it ends on a word that leaves it open, or it
 * has no closing mark although this transcriber writes them (`punctuates`). A wider form of the
 * rule (any open last word) called 266 of 700 labelled English clips unfinished and was right about
 * 258. It says nothing the other way round: a transcript that reads finished is often not.
 */
export function readsUnfinished(transcript: string, punctuates: boolean): boolean {
  const text = transcript.trim();
  if (!text) return false;
  const words = text.toLowerCase().replace(/[^\p{L}\p{N}'\s]/gu, " ").split(/\s+/).filter(Boolean);
  const last = words[words.length - 1] ?? "";
  if (OPEN_ENDINGS.has(last) && !/[?!？！؟]["'”’)\]]*$/u.test(text)) return true;
  if (!punctuates || CLOSING_MARK.test(text)) return false;
  // A word or two is an answer ("Yes", "Room 12"), and transcribers often leave those unmarked.
  return words.length >= 3;
}

// One load for the page: the mic re-arms after every turn, and the model stays up between them.
let loading: Promise<boolean> | null = null;

/** Starts the end-of-turn model (a first use downloads it). False when this install cannot run it. */
export function loadTurnModel(): Promise<boolean> {
  loading ??= authFetch("/api/inference/audio/turn/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  })
    .then((response) => response.ok)
    .catch(() => false);
  return loading;
}

export async function unloadTurnModel(): Promise<void> {
  const pending = loading;
  loading = null;
  // A first load downloads before starting the server, so the unload waits for it.
  await pending;
  await authFetch("/api/inference/audio/turn/unload", { method: "POST" }).catch(() => {});
}

/** `language` is the language being spoken (a BCP 47 tag), when the user set one: the bar depends on it. */
export async function askTurnEnd(
  wav: Blob,
  signal?: AbortSignal,
  language?: string,
): Promise<TurnEndAnswer> {
  const query = language && language !== "auto" ? `?${new URLSearchParams({ language })}` : "";
  const response = await authFetch(`/api/inference/audio/turn/detect${query}`, {
    method: "POST",
    headers: { "Content-Type": "audio/wav" },
    body: wav,
    signal,
  });
  if (!response.ok) throw new Error(response.statusText || `HTTP ${response.status}`);
  const body = (await response.json()) as { finished?: boolean; p_complete?: number | null };
  return { finished: body.finished === true, p: body.p_complete ?? null };
}

/**
 * One question per pause. `key` names the speech heard so far, so an answer counts only for the
 * speech it was asked about: once the speaker goes on, the key changes and the old answer is dropped.
 */
export function createTurnEndCheck(
  ask: (wav: Blob, signal: AbortSignal) => Promise<TurnEndAnswer>,
  onAnswer: (answer: TurnEndAnswer, ms: number) => void = () => {},
) {
  let check: { key: string; finished: boolean; abort: AbortController } | null = null;
  return {
    ask(key: string, wav: () => Blob): void {
      if (check?.key === key) return;
      check?.abort.abort();
      const mine = { key, finished: false, abort: new AbortController() };
      check = mine;
      const at = performance.now();
      ask(wav(), mine.abort.signal).then(
        (answer) => {
          if (check !== mine) return;
          mine.finished = answer.finished;
          onAnswer(answer, Math.round(performance.now() - at));
        },
        () => {},
      );
    },
    finished(key: string): boolean {
      return check?.key === key && check.finished;
    },
    clear(): void {
      check?.abort.abort();
      check = null;
    },
  };
}
