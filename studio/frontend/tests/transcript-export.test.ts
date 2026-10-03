// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type TranscriptExport,
  exportFileName,
  exportTranscript,
  formatNeedsTimestamps,
  toJson,
  toSrt,
  toTxt,
  toVtt,
} from "../src/features/audio/transcript-export.ts";
import { EMPTY_TRANSCRIPT_DETAILS } from "../src/features/audio/transcript-model.ts";

const timed: TranscriptExport = {
  title: "meeting.wav",
  text: "Hello there. Hi! <b>Tom & Jerry</b>",
  model: "MOSS-Transcribe-Diarize",
  names: { S02: "Bea" },
  details: {
    segments: [
      { start: 1.234, end: 3, text: "Hello there.", speaker: "S01" },
      { start: 3, end: 3.5, text: "Hi!", speaker: "S02" },
      {
        start: 3661.001,
        end: 3662.25,
        text: "<b>Tom & Jerry</b>",
        speaker: "S02",
      },
    ],
    words: [{ start: 1.234, end: 1.6, word: "Hello" }],
    speakers: [
      { id: "S01", label: "Speaker 1" },
      { id: "S02", label: "Speaker 2" },
    ],
    source: { kind: "input", id: "abc", name: "meeting.wav" },
    language: "en",
    duration: 3663,
  },
};

const plain: TranscriptExport = {
  title: "note.m4a",
  text: "Just text.",
  model: "whisper-small",
  names: {},
  details: { ...EMPTY_TRANSCRIPT_DETAILS, duration: 4.5 },
};

test("SRT numbers cues from 1 with comma milliseconds and speaker names", () => {
  assert.equal(
    toSrt(timed),
    [
      "1",
      "00:00:01,234 --> 00:00:03,000",
      "Speaker 1: Hello there.",
      "",
      "2",
      "00:00:03,000 --> 00:00:03,500",
      "Bea: Hi!",
      "",
      "3",
      "01:01:01,001 --> 01:01:02,250",
      "Bea: <b>Tom & Jerry</b>",
      "",
    ].join("\n"),
  );
});

test("VTT has a header, dot milliseconds, voice tags and escaped text", () => {
  assert.equal(
    toVtt(timed),
    [
      "WEBVTT",
      "",
      "00:00:01.234 --> 00:00:03.000",
      "<v Speaker 1>Hello there.",
      "",
      "00:00:03.000 --> 00:00:03.500",
      "<v Bea>Hi!",
      "",
      "01:01:01.001 --> 01:01:02.250",
      "<v Bea>&lt;b&gt;Tom &amp; Jerry&lt;/b&gt;",
      "",
    ].join("\n"),
  );
  const noSpeakers: TranscriptExport = {
    ...timed,
    details: {
      ...timed.details,
      speakers: [],
      segments: timed.details.segments.map(({ speaker: _, ...rest }) => rest),
    },
  };
  assert.doesNotMatch(toVtt(noSpeakers), /<v /);
  assert.doesNotMatch(toSrt(noSpeakers), /Speaker 1:/);
});

test("a blank line inside a cue cannot end it early, and empty cues are dropped", () => {
  const input: TranscriptExport = {
    ...plain,
    details: {
      ...EMPTY_TRANSCRIPT_DETAILS,
      segments: [
        { start: 0, end: 1, text: "line one\n\n  line two" },
        { start: 1, end: 2, text: "   " },
        { start: 2, end: 3, text: "x --> y" },
      ],
    },
  };
  assert.equal(
    toSrt(input),
    "1\n00:00:00,000 --> 00:00:01,000\nline one\nline two\n\n2\n00:00:02,000 --> 00:00:03,000\nx --> y\n",
  );
  assert.match(toVtt(input), /\nx --&gt; y\n$/);
});

test("without segments SRT and VTT fall back to one cue over the whole clip", () => {
  assert.equal(toSrt(plain), "1\n00:00:00,000 --> 00:00:04,500\nJust text.\n");
  assert.equal(
    toVtt(plain),
    "WEBVTT\n\n00:00:00.000 --> 00:00:04.500\nJust text.\n",
  );
  const unknown = { ...plain, details: EMPTY_TRANSCRIPT_DETAILS };
  assert.equal(
    toSrt(unknown),
    "1\n00:00:00,000 --> 00:00:00,000\nJust text.\n",
  );
  const empty = { ...unknown, text: "" };
  assert.equal(toSrt(empty), "");
  assert.equal(toVtt(empty), "WEBVTT\n");
  assert.equal(formatNeedsTimestamps("srt"), true);
  assert.equal(formatNeedsTimestamps("vtt"), true);
  assert.equal(formatNeedsTimestamps("txt"), false);
  assert.equal(formatNeedsTimestamps("json"), false);
});

test("TXT is the plain text, or paragraphs led by the speaker's current name", () => {
  assert.equal(toTxt(plain), "Just text.");
  assert.equal(
    toTxt(timed),
    "Speaker 1: Hello there.\n\nBea: Hi! <b>Tom & Jerry</b>\n",
  );
  assert.equal(
    toTxt({ ...timed, names: { S01: "Al", S02: "Bea" } }).split("\n")[0],
    "Al: Hello there.",
  );
});

test("JSON carries the documented keys, names, and words only when present", () => {
  const json = JSON.parse(toJson(timed));
  assert.deepEqual(Object.keys(json), [
    "title",
    "model",
    "language",
    "duration",
    "text",
    "segments",
    "words",
    "speakers",
  ]);
  assert.deepEqual(json.speakers, [
    { id: "S01", label: "Speaker 1", name: null },
    { id: "S02", label: "Speaker 2", name: "Bea" },
  ]);
  assert.equal(json.segments.length, 3);
  const bare = JSON.parse(toJson(plain));
  assert.equal("words" in bare, false);
  assert.deepEqual(bare.segments, []);
  assert.deepEqual(bare.speakers, []);
  assert.equal(bare.duration, 4.5);
});

test("file names drop the extension and unsafe characters", () => {
  assert.equal(exportFileName("meeting.wav", "srt"), "meeting.srt");
  assert.equal(
    exportFileName('a/b\\c:d*e?"f<g>h|i.mp3', "vtt"),
    "a_b_c_d_e__f_g_h_i.vtt",
  );
  assert.equal(exportFileName("tab\there", "txt"), "tab_here.txt");
  assert.equal(exportFileName("", "json"), "transcript.json");
  assert.equal(exportFileName(".wav", "txt"), "transcript.txt");
});

test("each format has its media type", () => {
  assert.equal(exportTranscript("txt", plain).mime, "text/plain;charset=utf-8");
  assert.equal(
    exportTranscript("srt", plain).mime,
    "application/x-subrip;charset=utf-8",
  );
  assert.equal(exportTranscript("vtt", plain).mime, "text/vtt;charset=utf-8");
  assert.equal(
    exportTranscript("json", plain).mime,
    "application/json;charset=utf-8",
  );
  assert.equal(exportTranscript("vtt", timed).content, toVtt(timed));
});
