// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type TranscriptDetails,
  type TranscriptSegment,
  speakerLabel,
} from "./transcript-model.ts";

export type TranscriptExportFormat = "txt" | "srt" | "vtt" | "json";

export const TRANSCRIPT_EXPORT_FORMATS: readonly TranscriptExportFormat[] = [
  "txt",
  "srt",
  "vtt",
  "json",
];

export interface TranscriptExport {
  title: string;
  text: string;
  model: string;
  details: TranscriptDetails;
  names: Readonly<Record<string, string>>;
}

const MIME: Record<TranscriptExportFormat, string> = {
  txt: "text/plain;charset=utf-8",
  srt: "application/x-subrip;charset=utf-8",
  vtt: "text/vtt;charset=utf-8",
  json: "application/json;charset=utf-8",
};

export function formatNeedsTimestamps(format: TranscriptExportFormat): boolean {
  return format === "srt" || format === "vtt";
}

function nameOf(input: TranscriptExport, id: string): string {
  return speakerLabel(id, input.details.speakers, input.names);
}

function speakersOn(input: TranscriptExport): boolean {
  return (
    input.details.speakers.length > 0 &&
    input.details.segments.some((segment) => Boolean(segment.speaker))
  );
}

// A blank line would end the cue early.
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

/** Without timing, one cue spans the clip. */
function cues(input: TranscriptExport) {
  const { segments, duration } = input.details;
  const timed: TranscriptSegment[] =
    segments.length > 0
      ? segments
      : input.text.trim()
        ? [{ start: 0, end: duration ?? 0, text: input.text }]
        : [];
  const on = speakersOn(input);
  return timed
    .map((segment) => ({
      segment,
      text: cueText(segment.text),
      name: on && segment.speaker ? nameOf(input, segment.speaker) : null,
    }))
    .filter(({ text }) => text);
}

export function toTxt(input: TranscriptExport): string {
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
      segment.speaker ? `${nameOf(input, segment.speaker)}: ${text}` : text,
    );
  }
  return `${blocks.join("\n\n")}\n`;
}

export function toSrt(input: TranscriptExport): string {
  const blocks = cues(input).map(
    ({ segment, text, name }, index) =>
      `${index + 1}\n${clock(segment.start, ",")} --> ${clock(segment.end, ",")}\n${name === null ? "" : `${name}: `}${text}`,
  );
  return blocks.length > 0 ? `${blocks.join("\n\n")}\n` : "";
}

function escapeVtt(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function toVtt(input: TranscriptExport): string {
  const blocks = cues(input).map(
    ({ segment, text, name }) =>
      `${clock(segment.start, ".")} --> ${clock(segment.end, ".")}\n${name === null ? "" : `<v ${escapeVtt(name)}>`}${escapeVtt(text)}`,
  );
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

export function exportFileName(title: string, ext: string): string {
  const base =
    title
      .replace(/\.[^.]+$/, "")
      .replace(/[<>:"/\\|?*]/g, "_")
      .replace(/\p{Cc}/gu, "_") || "transcript";
  return `${base}.${ext}`;
}

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
