// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type TranscriptDetails,
  type TranscriptSegment,
  hasSpeakers,
  paragraphs,
  speakerLabel,
} from "./transcript-model.ts";

export type TranscriptExportFormat = "txt" | "srt" | "vtt" | "json";

export interface TranscriptExport {
  title: string;
  text: string;
  model: string;
  details: TranscriptDetails;
  names: Readonly<Record<string, string>>;
}

function nameOf(input: TranscriptExport, id: string): string {
  return speakerLabel(id, input.details.speakers, input.names);
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

function cues(input: TranscriptExport) {
  const { segments, duration } = input.details;
  const timed: TranscriptSegment[] =
    segments.length > 0
      ? segments
      : input.text.trim()
        ? [{ start: 0, end: duration ?? 0, text: input.text }]
        : [];
  const on = hasSpeakers(input.details);
  return timed
    .map((segment) => ({
      segment,
      text: cueText(segment.text),
      name: on && segment.speaker ? nameOf(input, segment.speaker) : null,
    }))
    .filter(({ text }) => text);
}

export function toTxt(input: TranscriptExport): string {
  if (!hasSpeakers(input.details)) return input.text;
  const blocks = paragraphs(input.details.segments).map(({ speaker, text }) =>
    speaker ? `${nameOf(input, speaker)}: ${text}` : text,
  );
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

const FORMATS: Record<
  TranscriptExportFormat,
  { mime: string; write: (input: TranscriptExport) => string }
> = {
  txt: { mime: "text/plain;charset=utf-8", write: toTxt },
  srt: { mime: "application/x-subrip;charset=utf-8", write: toSrt },
  vtt: { mime: "text/vtt;charset=utf-8", write: toVtt },
  json: { mime: "application/json;charset=utf-8", write: toJson },
};

export const TRANSCRIPT_EXPORT_FORMATS = Object.keys(
  FORMATS,
) as TranscriptExportFormat[];

export function exportTranscript(
  format: TranscriptExportFormat,
  input: TranscriptExport,
): { content: string; mime: string } {
  return { content: FORMATS[format].write(input), mime: FORMATS[format].mime };
}
