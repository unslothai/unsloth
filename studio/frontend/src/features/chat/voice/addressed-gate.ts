// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The pure half of the "only answer when I'm talking to it" gate, import-free so it
// runs under node: the name rule, the overheard buffer and its context block, and
// the outcome the loop reports once the next utterance is known.

import { REPEAT_WINDOW_MS, sharesMostWords } from "./laya-outcomes.ts";

export type AddressedLabel = "assistant" | "other" | "reading";
/** What came before this utterance; "assistant" also means the assistant just replied. */
export type AddressedPrev = "none" | AddressedLabel;

/** Overheard speech older than this is not worth quoting back. */
export const OVERHEARD_WINDOW_MS = 120_000;
export const OVERHEARD_MAX_CHARS = 600;

// A turn that names the app is for it whatever the model thinks; the app has no wake
// word, so this is the handful of ways people address it out loud.
const ADDRESS_WORDS =
  /(^|[^\p{L}])(hey|ok|okay|yo|hi|hello|excuse me)[,!]?\s+(assistant|unsloth|computer|app|ai)(?![\p{L}])|^(assistant|unsloth)[,:!]/iu;

export function isAddressedByName(text: string): boolean {
  return ADDRESS_WORDS.test(text.trim());
}

export interface OverheardEntry {
  text: string;
  label: AddressedLabel;
  at: number;
}

/** Append an overheard utterance and drop what fell out of the window. */
export function pushOverheard(
  list: readonly OverheardEntry[],
  entry: OverheardEntry,
): OverheardEntry[] {
  return [...list, entry].filter((e) => entry.at - e.at <= OVERHEARD_WINDOW_MS);
}

/** The newest overheard speech inside the window, oldest first, capped at the char budget. */
export function overheardText(list: readonly OverheardEntry[], now: number): string {
  const recent = list.filter((e) => now - e.at <= OVERHEARD_WINDOW_MS && e.text.trim());
  const picked: string[] = [];
  let used = 0;
  for (let i = recent.length - 1; i >= 0; i--) {
    const t = recent[i].text.trim();
    if (used + t.length + 1 > OVERHEARD_MAX_CHARS) break;
    picked.unshift(t);
    used += t.length + 1;
  }
  return picked.join(" ");
}

/** The user's turn with the overheard speech quoted in front of it, or the turn alone. */
export function withOverheardContext(
  text: string,
  list: readonly OverheardEntry[],
  now: number,
): string {
  const heard = overheardText(list, now);
  if (!heard) return text;
  return `(Heard earlier, not addressed to you: "${heard.replace(/"/g, "'")}")\n\n${text}`;
}

/** A turn held back as other / reading, then said again inside the window, was for the assistant. */
export function addressedOutcome(
  earlier: string,
  later: string | null,
  gapMs: number,
): "repeated" | "ok" {
  if (later === null || gapMs > REPEAT_WINDOW_MS) return "ok";
  return sharesMostWords(earlier, later) ? "repeated" : "ok";
}

/** "not you", "I wasn't talking to you": the answered turn was not for the assistant after all. */
export function disownsReply(text: string): boolean {
  return /\b(not you|wasn'?t talking to you|not talking to you|wasn'?t asking you|ignore that)\b/i.test(text);
}
