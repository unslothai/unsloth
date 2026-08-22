// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { renderTranscriptExport } from "../src/features/audio/transcript-export.ts";

const segments = [
  { start: 0, end: 1.5, text: "Hello world." },
  { start: 1.5, end: 3.25, text: "Second line." },
];

test("plain txt export ignores segments and returns the raw transcript", () => {
  assert.equal(
    renderTranscriptExport("txt", "Hello world. Second line.", segments),
    "Hello world. Second line.",
  );
});

test("timestamped txt export prefixes each segment with its start time", () => {
  assert.equal(
    renderTranscriptExport("timestamped-txt", "unused", segments),
    "[00:00:00.000] Hello world.\n[00:00:01.500] Second line.",
  );
});

test("srt export numbers cues and uses comma-separated milliseconds", () => {
  const srt = renderTranscriptExport("srt", "unused", segments);
  assert.equal(
    srt,
    [
      "1",
      "00:00:00,000 --> 00:00:01,500",
      "Hello world.",
      "",
      "2",
      "00:00:01,500 --> 00:00:03,250",
      "Second line.",
      "",
    ].join("\n"),
  );
});

test("vtt export opens with the WEBVTT header and dot-separated milliseconds", () => {
  const vtt = renderTranscriptExport("vtt", "unused", segments);
  assert.match(vtt, /^WEBVTT\n\n/);
  assert.match(vtt, /00:00:00\.000 --> 00:00:01\.500\nHello world\./);
});

test("json export carries both the flat transcript and the segment array", () => {
  const parsed = JSON.parse(
    renderTranscriptExport("json", "Hello world. Second line.", segments),
  ) as { text: string; segments: typeof segments };
  assert.equal(parsed.text, "Hello world. Second line.");
  assert.deepEqual(parsed.segments, segments);
});

test("csv export quotes text fields and escapes embedded quotes", () => {
  const csv = renderTranscriptExport("csv", "unused", [
    { start: 0, end: 1, text: 'She said "hi".' },
  ]);
  assert.equal(csv, 'start,end,text\n0,1,"She said ""hi""."');
});

test("timestamp-dependent formats render nothing when segments are absent", () => {
  assert.equal(renderTranscriptExport("srt", "unused", null), "");
  assert.equal(renderTranscriptExport("vtt", "unused", null), "WEBVTT\n");
  assert.equal(renderTranscriptExport("csv", "unused", null), "start,end,text");
});
