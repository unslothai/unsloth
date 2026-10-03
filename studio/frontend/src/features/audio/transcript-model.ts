// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure transcript helpers: no imports, so node tests can load this file directly.

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  /** The model's raw speaker id (S01, 0), when it told speakers apart. */
  speaker?: string;
}

export interface TranscriptWord {
  start: number;
  end: number;
  word: string;
}

export interface TranscriptSpeaker {
  id: string;
  /** The default name, "Speaker 1" by first appearance. */
  label: string;
}

/** Everything beyond the plain text: the timing, the speakers and where the audio came from. */
export interface TranscriptDetails {
  segments: TranscriptSegment[];
  words: TranscriptWord[];
  speakers: TranscriptSpeaker[];
  source: { kind: "input" | "clip" | "voice"; id: string; name: string } | null;
  language: string | null;
  duration: number | null;
}

export const EMPTY_TRANSCRIPT_DETAILS: TranscriptDetails = {
  segments: [],
  words: [],
  speakers: [],
  source: null,
  language: null,
  duration: null,
};

export const SPEAKER_NAME_MAX_LENGTH = 40;

/** The segment playing at `seconds`, or -1. In a gap the previous segment stays current. */
export function activeSegmentIndex(
  seconds: number,
  segments: readonly TranscriptSegment[],
): number {
  if (segments.length === 0 || !(seconds >= segments[0].start)) return -1;
  let low = 0;
  let high = segments.length - 1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (segments[middle].start <= seconds) low = middle;
    else high = middle - 1;
  }
  return low;
}

/** What a speaker is called: the user's name for them, else the default label, else the raw id. */
export function speakerLabel(
  id: string,
  speakers: readonly TranscriptSpeaker[],
  names: Readonly<Record<string, string>>,
): string {
  const named = names[id]?.trim();
  if (named) return named;
  return speakers.find((speaker) => speaker.id === id)?.label ?? id;
}

/** 1-based position of a speaker, for its colour (cycled by the caller). */
export function speakerIndex(
  id: string,
  speakers: readonly TranscriptSpeaker[],
): number {
  const index = speakers.findIndex((speaker) => speaker.id === id);
  return index < 0 ? 1 : index + 1;
}

export interface TranscriptParagraph {
  speaker?: string;
  start: number;
  text: string;
}

/** Consecutive segments by one speaker joined into one paragraph, for the Text view. */
export function paragraphs(
  segments: readonly TranscriptSegment[],
  speakersOn: boolean,
): TranscriptParagraph[] {
  const result: TranscriptParagraph[] = [];
  for (const segment of segments) {
    const speaker = speakersOn ? segment.speaker : undefined;
    const last = result[result.length - 1];
    if (last && speakersOn && last.speaker === speaker) {
      last.text = `${last.text} ${segment.text}`.trim();
    } else {
      result.push({ speaker, start: segment.start, text: segment.text.trim() });
    }
  }
  return speakersOn ? result : [];
}

/** `m:ss`, or `h:mm:ss` past an hour. */
export function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}`
    : `${minutes}:${secs}`;
}

export function hasTimestamps(details: TranscriptDetails | null): boolean {
  return Boolean(details && details.segments.length > 0);
}

/** A trimmed single-line name, capped; empty means "use the default label". */
export function sanitizeSpeakerName(name: string): string {
  return name.replace(/\s+/g, " ").trim().slice(0, SPEAKER_NAME_MAX_LENGTH);
}

function finiteSeconds(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

/** Details from a stream result, saved record or draft; malformed entries are dropped. */
export function detailsFrom(value: {
  segments?: unknown;
  words?: unknown;
  speakers?: unknown;
  source?: unknown;
  language?: unknown;
  duration?: unknown;
}): TranscriptDetails {
  const segments: TranscriptSegment[] = [];
  if (Array.isArray(value.segments)) {
    for (const item of value.segments) {
      if (!item || typeof item !== "object") continue;
      const start = finiteSeconds(item.start);
      const end = finiteSeconds(item.end);
      if (start === null || end === null || end < start) continue;
      if (typeof item.text !== "string") continue;
      const segment: TranscriptSegment = { start, end, text: item.text };
      if (typeof item.speaker === "string" && item.speaker)
        segment.speaker = item.speaker;
      segments.push(segment);
    }
  }
  const words: TranscriptWord[] = [];
  if (Array.isArray(value.words)) {
    for (const item of value.words) {
      if (!item || typeof item !== "object") continue;
      const start = finiteSeconds(item.start);
      const end = finiteSeconds(item.end);
      if (start === null || end === null || end < start) continue;
      if (typeof item.word !== "string") continue;
      words.push({ start, end, word: item.word });
    }
  }
  const speakers: TranscriptSpeaker[] = [];
  if (Array.isArray(value.speakers)) {
    for (const item of value.speakers) {
      if (!item || typeof item !== "object") continue;
      if (typeof item.id !== "string" || typeof item.label !== "string")
        continue;
      speakers.push({ id: item.id, label: item.label });
    }
  }
  const raw = value.source as
    | { kind?: unknown; id?: unknown; name?: unknown }
    | null
    | undefined;
  const source: TranscriptDetails["source"] =
    raw &&
    typeof raw === "object" &&
    (raw.kind === "input" || raw.kind === "clip" || raw.kind === "voice") &&
    typeof raw.id === "string" &&
    typeof raw.name === "string"
      ? {
          kind: raw.kind as "input" | "clip" | "voice",
          id: raw.id,
          name: raw.name,
        }
      : null;
  return {
    segments,
    words,
    speakers,
    source,
    language: typeof value.language === "string" ? value.language : null,
    duration: finiteSeconds(value.duration),
  };
}
