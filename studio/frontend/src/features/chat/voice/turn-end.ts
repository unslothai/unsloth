// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

// Mirrors ASK_AFTER_MS in turn_audiocpp_sidecar.py.
export const TURN_ASK_AFTER_MS = 250;
// Ending at ~350 ms split 4 of 5 turns at a 570 ms comma pause.
export const TURN_END_AFTER_MS = 700;

/** A question mark ends the turn even when the model disagrees. */
export function readsAsQuestion(text: string): boolean {
  return /[?？؟]["”’)\]]*$/.test(text.trim());
}
// Grace after quiet: a reply in this gap answers half a sentence.
export const TURN_CONTINUE_GRACE_MS = 1200;
// The model reads at most this many seconds, from the end.
const TURN_CLIP_MAX_S = 8;

export function stillInTurn(hearing: boolean, lastHeardAt: number, now: number): boolean {
  return hearing || (lastHeardAt > 0 && now - lastHeardAt < TURN_CONTINUE_GRACE_MS);
}

export function turnClipTail(samples: Float32Array, sampleRate: number): Float32Array {
  const keep = Math.round(TURN_CLIP_MAX_S * sampleRate);
  return samples.length > keep ? samples.subarray(samples.length - keep) : samples;
}

export interface TurnEndAnswer {
  finished: boolean;
  /** Null when the model is not asked for this language. */
  p: number | null;
}

/** End at the last non-silent frame, not the last loud one. */
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

const OPEN_ENDINGS = new Set(
  "and or but because cause if while than the a an my your our their um uh er erm".split(" "),
);
const CLOSING_MARK = /[.?!。？！؟।]["'”’)\]]*$/u;

export function hasClosingMark(transcript: string): boolean {
  return CLOSING_MARK.test(transcript.trim());
}

/** A false here does not mean finished. */
export function readsUnfinished(transcript: string, punctuates: boolean): boolean {
  const text = transcript.trim();
  if (!text) return false;
  const words = text.toLowerCase().replace(/[^\p{L}\p{N}'\s]/gu, " ").split(/\s+/).filter(Boolean);
  const last = words[words.length - 1] ?? "";
  if (OPEN_ENDINGS.has(last) && !/[?!？！؟]["'”’)\]]*$/u.test(text)) return true;
  if (!punctuates || CLOSING_MARK.test(text)) return false;
  // One or two words are often unmarked answers.
  return words.length >= 3;
}

let loading: Promise<boolean> | null = null;

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
  // Wait out a first load: it downloads first.
  await pending;
  await authFetch("/api/inference/audio/turn/unload", { method: "POST" }).catch(() => {});
}

/** BCP 47 language, when the user set one. */
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

/** An answer counts only for the key it was asked for. */
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
