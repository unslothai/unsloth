// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  EMPTY_TRANSCRIPT_DETAILS,
  activeSegmentIndex,
  detailsFrom,
  formatTimestamp,
  paragraphs,
  sanitizeSpeakerName,
  speakerLabel,
} from "../src/features/audio/transcript-model.ts";

const segments = [
  { start: 0.5, end: 2, text: "One", speaker: "S01" },
  { start: 3, end: 5, text: "two ", speaker: "S01" },
  { start: 4.5, end: 7, text: "Three", speaker: "S02" },
  { start: 6, end: 7, text: "  ", speaker: "S01" },
  { start: 7, end: 9, text: "four", speaker: "S01" },
];

test("the active segment is found at the edges, in gaps and across overlaps", () => {
  const at = (seconds: number) => activeSegmentIndex(seconds, segments);
  assert.deepEqual(
    [0, 0.5, 2.5, 4.6, 7, 60, Number.NaN].map(at),
    [-1, 0, 0, 2, 4, 4, -1],
  );
  assert.equal(activeSegmentIndex(1, []), -1);
});

test("speakers are named by rename, then default label, then raw id, and grouped into paragraphs", () => {
  const speakers = [{ id: "S01", label: "Speaker 1" }];
  assert.equal(speakerLabel("S01", speakers, {}), "Speaker 1");
  assert.equal(speakerLabel("S01", speakers, { S01: " Alice " }), "Alice");
  assert.equal(speakerLabel("S01", speakers, { S01: "  " }), "Speaker 1");
  assert.equal(speakerLabel("S09", speakers, {}), "S09");
  assert.deepEqual(paragraphs(segments), [
    { speaker: "S01", text: "One two" },
    { speaker: "S02", text: "Three" },
    { speaker: "S01", text: "four" },
  ]);
});

test("timestamps read m:ss, h:mm:ss past an hour, and speaker names are one capped line", () => {
  assert.deepEqual(
    [0, 83.9, 3599.999, 3725, -4, Number.POSITIVE_INFINITY].map(
      formatTimestamp,
    ),
    ["0:00", "1:23", "59:59", "1:02:05", "0:00", "0:00"],
  );
  assert.equal(sanitizeSpeakerName("  Ada\n  Lovelace "), "Ada Lovelace");
  assert.equal(sanitizeSpeakerName("x".repeat(80)), "x".repeat(40));
});

test("details drop malformed entries, server paths and unknown source kinds", () => {
  const good = {
    segments: [{ start: 1, end: 2, text: "ok", speaker: "S01" }],
    words: [{ start: 0, end: 1, word: "ok" }],
    speakers: [{ id: "S01", label: "Speaker 1" }],
    source: { kind: "input", id: "abc", name: "a.wav" },
    language: "en",
    duration: 12,
  };
  const details = detailsFrom({
    ...good,
    segments: [
      ...good.segments,
      { start: 3, end: 2, text: "backwards" },
      { start: "1", end: 2, text: "string time" },
      { start: 1, end: 2 },
      null,
    ],
    words: [...good.words, { start: 0, end: 1 }],
    speakers: [...good.speakers, { id: 1 }],
    source: { ...good.source, path: "/etc/passwd" },
  });
  assert.deepEqual(details, good);
  assert.deepEqual(
    detailsFrom({ ...good, source: { ...good.source, kind: "file" } }).source,
    null,
  );
  assert.deepEqual(detailsFrom({}), EMPTY_TRANSCRIPT_DETAILS);
});
