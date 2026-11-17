// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure: how much of a spoken reply the user heard before cutting in, and that reply as the model
// then sees it. The message text is never rewritten; the cut is an offset applied on the way out.

import { splitSpoken, stripForSpeech, type SpokenSentence } from "./speech-text.ts";

/** One piece of a scheduled clip. `from`/`to` are offsets into the stream's text; `chars` weighs its share. */
export interface PlayedEntry {
  source: number;
  from: number;
  to: number;
  chars: number;
}

/** A clip on the AudioContext clock. Fillers and asides carry no entries. */
export interface PlayedChunk {
  entries: readonly PlayedEntry[];
  start: number;
  duration: number;
}

export interface HeardCut {
  offset: number;
}

type Part = { type: string; text?: string };

/** The reply's text parts as voice reads them, "\n" between parts, and where each part starts. */
export function joinTextParts(content: readonly Part[]): { text: string; partStarts: number[] } {
  let text = "";
  const partStarts: number[] = [];
  for (const part of content) {
    if (part.type === "text" && part.text) {
      if (text) text += "\n";
      partStarts.push(text.length);
      text += part.text;
    } else {
      partStarts.push(text.length);
    }
  }
  return { text, partStarts };
}

/**
 * Kokoro gives no word timings, so the unit is the splitter sentence: one is heard once the
 * audible position passes the midpoint of the piece that ends it, a chunk's time shared among its
 * pieces by characters. No log (the chat model spoke the reply itself) is no cut.
 */
export function heardCut(
  sentences: readonly SpokenSentence[],
  log: readonly PlayedChunk[] | null,
  position: number,
  base: number,
): HeardCut | null {
  if (log === null) return null;
  const ends = new Set(sentences.map((s) => s.end));
  let offset = 0;
  for (const chunk of log) {
    const total = chunk.entries.reduce((n, e) => n + e.chars, 0);
    let at = chunk.start;
    for (const entry of chunk.entries) {
      const span = total > 0 ? (chunk.duration * entry.chars) / total : 0;
      const end = base + entry.to;
      if (position >= at + span / 2 && ends.has(end)) offset = Math.max(offset, end);
      at += span;
    }
  }
  return { offset };
}

/** A second barge-in on the same reply keeps whichever cut heard more. */
export function laterCut(a: HeardCut | null, b: HeardCut | null): HeardCut | null {
  if (!a) return b;
  if (!b) return a;
  return a.offset >= b.offset ? a : b;
}

/** The cut saved on the message (`metadata.custom.voiceHeard`), if any. */
export function savedHeardCut(metadata: unknown): HeardCut | null {
  const heard = (metadata as { custom?: { voiceHeard?: { offset?: unknown } } } | null | undefined)
    ?.custom?.voiceHeard;
  return typeof heard?.offset === "number" && heard.offset >= 0 ? { offset: heard.offset } : null;
}

/** Text up to the cut stays, with the parts before it; everything after is hidden from the model. */
export function applyHeardCut<T extends { content: readonly Part[] }>(message: T, cut: HeardCut): T {
  const { partStarts } = joinTextParts(message.content);
  let at = -1;
  message.content.forEach((part, i) => {
    if (part.type === "text" && part.text && partStarts[i] <= cut.offset) at = i;
  });
  if (at < 0) return message;
  const part = message.content[at];
  const text = (part.text ?? "").slice(0, cut.offset - partStarts[at]);
  if (at === message.content.length - 1 && text === part.text) return message;
  return { ...message, content: [...message.content.slice(0, at), { ...part, text }] };
}

/**
 * Metadata a writer built from an older copy of the reply, with the heard cut the open reply carries now.
 * Only the open reply sets or clears a cut ("Use full reply"); a copy taken before either would undo it.
 * No open copy (`live` undefined): the writer's own is all there is.
 */
export function keepVoiceHeard(
  next: Record<string, unknown>,
  live: Record<string, unknown> | undefined,
): Record<string, unknown> {
  if (live === undefined || next.voiceHeard === live.voiceHeard) return next;
  if (live.voiceHeard === undefined) {
    const rest = { ...next };
    delete rest.voiceHeard;
    return rest;
  }
  return { ...next, voiceHeard: live.voiceHeard };
}

/** The last words the user heard, for the note under an interrupted reply; "" when none were. */
// Built on first use, then reused.
let graphemeSegmenter: Intl.Segmenter | undefined;

export function heardTail(content: readonly Part[], cut: HeardCut, words = 6): string {
  // An inline player tag carries its audio as base64: blanked, or the note would show it.
  const heard = splitSpoken(joinTextParts(content).text.slice(0, cut.offset).replace(/<audio-player\b[^>]*\/>/g, " "))
    .map((s) => stripForSpeech(s.text))
    .filter(Boolean)
    .join(" ")
    .split(/\s+/)
    .filter(Boolean);
  // An unspaced script (ja, zh, th) has no words to count: the note keeps to a short tail by characters too.
  // By graphemes, so neither an emoji nor a letter with its marks is split.
  graphemeSegmenter ??= new Intl.Segmenter(undefined, { granularity: "grapheme" });
  const graphemes = graphemeSegmenter.segment(heard.slice(-words).join(" "));
  return Array.from(graphemes, (g) => g.segment).slice(-80).join("");
}
