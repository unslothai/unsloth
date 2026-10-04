// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No imports, so node tests can load this file directly.

export interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  speaker?: string;
}

export interface TranscriptWord {
  start: number;
  end: number;
  word: string;
}

export interface TranscriptSpeaker {
  id: string;
  label: string;
}

export interface TranscriptSource {
  kind: "input" | "clip" | "voice";
  id: string;
  name: string;
}

export interface TranscriptDetails {
  segments: TranscriptSegment[];
  words: TranscriptWord[];
  speakers: TranscriptSpeaker[];
  source: TranscriptSource | null;
  language: string | null;
  duration: number | null;
}

export const EMPTY_TRANSCRIPT_DETAILS = detailsFrom({});

export const SPEAKER_NAME_MAX_LENGTH = 40;

/** In a gap the previous segment stays current. */
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

export function speakerLabel(
  id: string,
  speakers: readonly TranscriptSpeaker[],
  names: Readonly<Record<string, string>>,
): string {
  const named = names[id]?.trim();
  if (named) return named;
  return speakers.find((speaker) => speaker.id === id)?.label ?? id;
}

export function hasSpeakers({
  speakers,
  segments,
}: TranscriptDetails): boolean {
  return (
    speakers.length > 0 && segments.some((segment) => Boolean(segment.speaker))
  );
}

export function paragraphs(
  segments: readonly TranscriptSegment[],
): { speaker?: string; text: string }[] {
  const result: { speaker?: string; text: string }[] = [];
  for (const { speaker, text: raw } of segments) {
    const text = raw.trim();
    if (!text) continue;
    const last = result[result.length - 1];
    if (last && last.speaker === speaker) last.text += ` ${text}`;
    else result.push({ speaker, text });
  }
  return result;
}

export function formatTimestamp(seconds: number): string {
  const total = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = String(total % 60).padStart(2, "0");
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}`
    : `${minutes}:${secs}`;
}

export function sanitizeSpeakerName(name: string): string {
  return name.replace(/\s+/g, " ").trim().slice(0, SPEAKER_NAME_MAX_LENGTH);
}

function finiteSeconds(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : null;
}

export function detailsFrom(
  value: Partial<Record<keyof TranscriptDetails, unknown>>,
): TranscriptDetails {
  const segments: TranscriptSegment[] = [];
  const words: TranscriptWord[] = [];
  for (const [list, key] of [
    [value.segments, "text"],
    [value.words, "word"],
  ] as const) {
    if (!Array.isArray(list)) continue;
    for (const item of list) {
      if (!item || typeof item !== "object") continue;
      const start = finiteSeconds(item.start);
      const end = finiteSeconds(item.end);
      if (start === null || end === null || end < start) continue;
      if (typeof item[key] !== "string") continue;
      if (key === "word") words.push({ start, end, word: item.word });
      else if (typeof item.speaker === "string" && item.speaker)
        segments.push({ start, end, text: item.text, speaker: item.speaker });
      else segments.push({ start, end, text: item.text });
    }
  }
  const speakers: TranscriptSpeaker[] = (
    Array.isArray(value.speakers) ? value.speakers : []
  )
    .filter(
      (item) => typeof item?.id === "string" && typeof item.label === "string",
    )
    .map(({ id, label }) => ({ id, label }));
  const raw = value.source as Partial<Record<string, unknown>> | null;
  const source: TranscriptSource | null =
    raw &&
    typeof raw === "object" &&
    (raw.kind === "input" || raw.kind === "clip" || raw.kind === "voice") &&
    typeof raw.id === "string" &&
    typeof raw.name === "string"
      ? { kind: raw.kind, id: raw.id, name: raw.name }
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
