// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  EMPTY_TRANSCRIPT_DETAILS,
  SPEAKER_NAME_MAX_LENGTH,
  activeSegmentIndex,
  detailsFrom,
  formatTimestamp,
  hasTimestamps,
  paragraphs,
  sanitizeSpeakerName,
  speakerIndex,
  speakerLabel,
} from "../src/features/audio/transcript-model.ts";

const segments = [
  { start: 0.5, end: 2, text: "One", speaker: "S01" },
  { start: 3, end: 5, text: "two", speaker: "S01" },
  { start: 4.5, end: 7, text: "Three", speaker: "S02" },
  { start: 7, end: 9, text: "four", speaker: "S01" },
];
const speakers = [
  { id: "S01", label: "Speaker 1" },
  { id: "S02", label: "Speaker 2" },
];

test("the active segment is found at the edges, in gaps and across overlaps", () => {
  assert.equal(activeSegmentIndex(0, segments), -1, "before the first line");
  assert.equal(activeSegmentIndex(0.5, segments), 0, "at a start");
  assert.equal(
    activeSegmentIndex(2.5, segments),
    0,
    "a gap keeps the previous line",
  );
  assert.equal(
    activeSegmentIndex(4.6, segments),
    2,
    "an overlap picks the later start",
  );
  assert.equal(
    activeSegmentIndex(7, segments),
    3,
    "a shared boundary moves on",
  );
  assert.equal(
    activeSegmentIndex(60, segments),
    3,
    "past the end stays on the last",
  );
  assert.equal(activeSegmentIndex(Number.NaN, segments), -1);
  assert.equal(activeSegmentIndex(1, []), -1);
});

test("speakers are named by rename, then default label, then raw id", () => {
  assert.equal(speakerLabel("S01", speakers, {}), "Speaker 1");
  assert.equal(speakerLabel("S01", speakers, { S01: " Alice " }), "Alice");
  assert.equal(speakerLabel("S01", speakers, { S01: "  " }), "Speaker 1");
  assert.equal(speakerLabel("S09", speakers, {}), "S09");
  assert.equal(speakerIndex("S02", speakers), 2);
  assert.equal(speakerIndex("S09", speakers), 1);
});

test("paragraphs join consecutive lines by one speaker", () => {
  assert.deepEqual(paragraphs(segments, true), [
    { speaker: "S01", start: 0.5, text: "One two" },
    { speaker: "S02", start: 4.5, text: "Three" },
    { speaker: "S01", start: 7, text: "four" },
  ]);
  assert.deepEqual(paragraphs(segments, false), []);
});

test("timestamps read m:ss, and h:mm:ss past an hour", () => {
  assert.equal(formatTimestamp(0), "0:00");
  assert.equal(formatTimestamp(83.9), "1:23");
  assert.equal(formatTimestamp(3599.999), "59:59");
  assert.equal(formatTimestamp(3600), "1:00:00");
  assert.equal(formatTimestamp(3725), "1:02:05");
  assert.equal(formatTimestamp(-4), "0:00");
  assert.equal(formatTimestamp(Number.POSITIVE_INFINITY), "0:00");
});

test("speaker names are one clean, capped line", () => {
  assert.equal(sanitizeSpeakerName("  Ada\n  Lovelace "), "Ada Lovelace");
  assert.equal(
    sanitizeSpeakerName("x".repeat(80)).length,
    SPEAKER_NAME_MAX_LENGTH,
  );
  assert.equal(sanitizeSpeakerName("   "), "");
});

test("details drop malformed entries and keep good ones", () => {
  const details = detailsFrom({
    segments: [
      { start: 1, end: 2, text: "ok", speaker: "S01" },
      { start: 3, end: 2, text: "backwards" },
      { start: "1", end: 2, text: "string time" },
      { start: 1, end: 2 },
      null,
    ],
    words: [
      { start: 0, end: 1, word: "ok" },
      { start: 0, end: 1 },
    ],
    speakers: [{ id: "S01", label: "Speaker 1" }, { id: 1 }],
    source: { kind: "input", id: "abc", name: "a.wav", path: "/etc/passwd" },
    language: "en",
    duration: 12,
  });
  assert.deepEqual(details, {
    segments: [{ start: 1, end: 2, text: "ok", speaker: "S01" }],
    words: [{ start: 0, end: 1, word: "ok" }],
    speakers: [{ id: "S01", label: "Speaker 1" }],
    source: { kind: "input", id: "abc", name: "a.wav" },
    language: "en",
    duration: 12,
  });
  assert.deepEqual(detailsFrom({}), EMPTY_TRANSCRIPT_DETAILS);
  assert.equal(
    detailsFrom({ source: { kind: "file", id: "x", name: "y" } }).source,
    null,
  );
  assert.equal(hasTimestamps(details), true);
  assert.equal(hasTimestamps(EMPTY_TRANSCRIPT_DETAILS), false);
  assert.equal(hasTimestamps(null), false);
});
