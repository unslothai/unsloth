// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Multi-format transcript export for the Studio Transcribe panel. Segment
// timing comes straight from the serving STT engine (see
// `transcribeAudioBlobDetailed`); formats that need timestamps are only
// offered when segments are present.

import type { SttSegment } from "@/features/chat/adapters/studio-model-dictation-adapter";

export type TranscriptExportFormat = "txt" | "timestamped-txt" | "srt" | "vtt" | "json" | "csv";

function pad(n: number, width: number): string {
  return String(Math.trunc(n)).padStart(width, "0");
}

/** `HH:MM:SS,mmm`, SRT's timestamp format. */
function srtTimestamp(seconds: number): string {
  const clamped = Math.max(0, seconds);
  const hours = Math.floor(clamped / 3600);
  const minutes = Math.floor((clamped % 3600) / 60);
  const secs = Math.floor(clamped % 60);
  const millis = Math.round((clamped - Math.floor(clamped)) * 1000);
  return `${pad(hours, 2)}:${pad(minutes, 2)}:${pad(secs, 2)},${pad(millis, 3)}`;
}

/** `HH:MM:SS.mmm`, WebVTT's timestamp format. */
function vttTimestamp(seconds: number): string {
  return srtTimestamp(seconds).replace(",", ".");
}

function csvField(value: string): string {
  return `"${value.replace(/"/g, '""')}"`;
}

/** Render a transcript in one export format. `segments` is required for every
 * format but plain and timestamped `.txt`. */
export function renderTranscriptExport(
  format: TranscriptExportFormat,
  transcript: string,
  segments: SttSegment[] | null,
): string {
  switch (format) {
    case "txt":
      return transcript;
    case "timestamped-txt":
      return (segments ?? [])
        .map((segment) => `[${vttTimestamp(segment.start)}] ${segment.text}`)
        .join("\n");
    case "srt":
      return (segments ?? [])
        .map((segment, index) =>
          [
            String(index + 1),
            `${srtTimestamp(segment.start)} --> ${srtTimestamp(segment.end)}`,
            segment.text,
            "",
          ].join("\n"),
        )
        .join("\n");
    case "vtt":
      return [
        "WEBVTT",
        "",
        ...(segments ?? []).map((segment) =>
          [
            `${vttTimestamp(segment.start)} --> ${vttTimestamp(segment.end)}`,
            segment.text,
            "",
          ].join("\n"),
        ),
      ].join("\n");
    case "json":
      return JSON.stringify({ text: transcript, segments: segments ?? [] }, null, 2);
    case "csv":
      return [
        "start,end,text",
        ...(segments ?? []).map(
          (segment) =>
            `${segment.start},${segment.end},${csvField(segment.text)}`,
        ),
      ].join("\n");
  }
}

export const TRANSCRIPT_EXPORT_MIME: Record<TranscriptExportFormat, string> = {
  txt: "text/plain",
  "timestamped-txt": "text/plain",
  srt: "text/plain",
  vtt: "text/vtt",
  json: "application/json",
  csv: "text/csv",
};

export const TRANSCRIPT_EXPORT_EXTENSION: Record<TranscriptExportFormat, string> = {
  txt: "txt",
  "timestamped-txt": "txt",
  srt: "srt",
  vtt: "vtt",
  json: "json",
  csv: "csv",
};

export const TRANSCRIPT_EXPORT_LABEL: Record<TranscriptExportFormat, string> = {
  txt: "Download .txt",
  "timestamped-txt": "Download timestamped .txt",
  srt: "Download .srt (subtitles)",
  vtt: "Download .vtt (subtitles)",
  json: "Download .json (with timestamps)",
  csv: "Download .csv (with timestamps)",
};
