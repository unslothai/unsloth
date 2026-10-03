// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Transcript file formats. Type imports only, so node tests can load this file directly.

import type {
  TranscriptDetails,
  TranscriptSegment,
  TranscriptSpeaker,
} from "./transcript-model.ts";

export type TranscriptExportFormat = "txt" | "srt" | "vtt" | "json";

export const TRANSCRIPT_EXPORT_FORMATS: readonly TranscriptExportFormat[] = [
  "txt",
  "srt",
  "vtt",
  "json",
];

/** Everything a transcript file is built from. */
export interface TranscriptExport {
  title: string;
  text: string;
  model: string;
  details: TranscriptDetails;
  /** The user's names for speakers, by raw id. */
  names: Readonly<Record<string, string>>;
}

const MIME: Record<TranscriptExportFormat, string> = {
  txt: "text/plain;charset=utf-8",
  srt: "application/x-subrip;charset=utf-8",
  vtt: "text/vtt;charset=utf-8",
  json: "application/json;charset=utf-8",
};

/** SRT and VTT need timed segments; without them they would be one cue for the whole clip. */
export function formatNeedsTimestamps(format: TranscriptExportFormat): boolean {
  return format === "srt" || format === "vtt";
}

// Local copy of transcript-model's rule, so this file stays free of runtime imports.
function nameOf(
  id: string,
  speakers: readonly TranscriptSpeaker[],
  names: Readonly<Record<string, string>>,
): string {
  const named = names[id]?.trim();
  if (named) return named;
  return speakers.find((speaker) => speaker.id === id)?.label ?? id;
}

/** Whether the file should say who is talking: the model told speakers apart. */
function speakersOn(input: TranscriptExport): boolean {
  return (
    input.details.speakers.length > 0 &&
    input.details.segments.some((segment) => Boolean(segment.speaker))
  );
}

/** A cue's text on as few lines as it needs: a blank line would end the cue early. */
function cueText(text: string): string {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .join("\n");
}

function clock(seconds: number, separator: "," | "."): string {
  const total = Math.max(
    0,
    Math.round((Number.isFinite(seconds) ? seconds : 0) * 1000),
  );
  const millis = total % 1000;
  const whole = Math.floor(total / 1000);
  const pad = (value: number, width = 2) => String(value).padStart(width, "0");
  return `${pad(Math.floor(whole / 3600))}:${pad(Math.floor((whole % 3600) / 60))}:${pad(whole % 60)}${separator}${pad(millis, 3)}`;
}

/** The timed cues, or one cue spanning the clip when the model gave no timing. */
function cues(input: TranscriptExport): TranscriptSegment[] {
  if (input.details.segments.length > 0) return input.details.segments;
  if (!input.text.trim()) return [];
  return [{ start: 0, end: input.details.duration ?? 0, text: input.text }];
}

export function toTxt(input: TranscriptExport): string {
  // Without speakers the file is the text exactly as shown, as before this PR.
  if (!speakersOn(input)) return input.text;
  const blocks: string[] = [];
  let previous: string | undefined;
  for (const segment of input.details.segments) {
    const text = segment.text.trim();
    if (!text) continue;
    if (blocks.length > 0 && segment.speaker === previous) {
      blocks[blocks.length - 1] += ` ${text}`;
      continue;
    }
    previous = segment.speaker;
    blocks.push(
      segment.speaker
        ? `${nameOf(segment.speaker, input.details.speakers, input.names)}: ${text}`
        : text,
    );
  }
  return `${blocks.join("\n\n")}\n`;
}

export function toSrt(input: TranscriptExport): string {
  const on = speakersOn(input);
  const blocks = cues(input)
    .map((segment) => ({ segment, text: cueText(segment.text) }))
    .filter(({ text }) => text)
    .map(({ segment, text }, index) => {
      const name =
        on && segment.speaker
          ? `${nameOf(segment.speaker, input.details.speakers, input.names)}: `
          : "";
      return `${index + 1}\n${clock(segment.start, ",")} --> ${clock(segment.end, ",")}\n${name}${text}`;
    });
  return blocks.length > 0 ? `${blocks.join("\n\n")}\n` : "";
}

function escapeVtt(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function toVtt(input: TranscriptExport): string {
  const on = speakersOn(input);
  const blocks = cues(input)
    .map((segment) => ({ segment, text: cueText(segment.text) }))
    .filter(({ text }) => text)
    .map(({ segment, text }) => {
      const voice =
        on && segment.speaker
          ? `<v ${escapeVtt(nameOf(segment.speaker, input.details.speakers, input.names))}>`
          : "";
      return `${clock(segment.start, ".")} --> ${clock(segment.end, ".")}\n${voice}${escapeVtt(text)}`;
    });
  return `${["WEBVTT", ...blocks].join("\n\n")}\n`;
}

export function toJson(input: TranscriptExport): string {
  const { details } = input;
  return `${JSON.stringify(
    {
      title: input.title,
      model: input.model,
      language: details.language,
      duration: details.duration,
      text: input.text,
      segments: details.segments,
      ...(details.words.length > 0 ? { words: details.words } : {}),
      speakers: details.speakers.map((speaker) => ({
        id: speaker.id,
        label: speaker.label,
        name: input.names[speaker.id]?.trim() || null,
      })),
    },
    null,
    2,
  )}\n`;
}

/** A safe file name from the transcript title: no extension, no path or control characters. */
export function exportFileName(title: string, ext: string): string {
  const base =
    title
      .replace(/\.[^.]+$/, "")
      .replace(/[<>:"/\\|?*]/g, "_")
      .replace(/\p{Cc}/gu, "_") || "transcript";
  return `${base}.${ext}`;
}

/** One format's file content and media type. */
export function exportTranscript(
  format: TranscriptExportFormat,
  input: TranscriptExport,
): { content: string; ext: TranscriptExportFormat; mime: string } {
  const content =
    format === "srt"
      ? toSrt(input)
      : format === "vtt"
        ? toVtt(input)
        : format === "json"
          ? toJson(input)
          : toTxt(input);
  return { content, ext: format, mime: MIME[format] };
}
