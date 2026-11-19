// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure. The message text is never rewritten; the cut is an offset applied on the way out.

import { splitSpoken, stripForSpeech, type SpokenSentence } from "./speech-text.ts";

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

/** Kokoro has no word timings: a sentence counts as heard past its last piece's midpoint. */
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

export function laterCut(a: HeardCut | null, b: HeardCut | null): HeardCut | null {
  if (!a) return b;
  if (!b) return a;
  return a.offset >= b.offset ? a : b;
}

export function savedHeardCut(metadata: unknown): HeardCut | null {
  const heard = (metadata as { custom?: { voiceHeard?: { offset?: unknown } } } | null | undefined)
    ?.custom?.voiceHeard;
  return typeof heard?.offset === "number" && heard.offset >= 0 ? { offset: heard.offset } : null;
}

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

/** The open copy's heard cut wins, even when cleared; with no open copy, the writer's stands. */
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

let graphemeSegmenter: Intl.Segmenter | undefined;

export function heardTail(content: readonly Part[], cut: HeardCut, words = 6): string {
  // Inline player tags carry base64 audio: blank them, or the note shows it.
  const heard = splitSpoken(joinTextParts(content).text.slice(0, cut.offset).replace(/<audio-player\b[^>]*\/>/g, " "))
    .map((s) => stripForSpeech(s.text))
    .filter(Boolean)
    .join(" ")
    .split(/\s+/)
    .filter(Boolean);
  // Unspaced scripts (ja, zh, th) have no words: the tail is cut by graphemes.
  graphemeSegmenter ??= new Intl.Segmenter(undefined, { granularity: "grapheme" });
  const graphemes = graphemeSegmenter.segment(heard.slice(-words).join(" "));
  return Array.from(graphemes, (g) => g.segment).slice(-80).join("");
}
