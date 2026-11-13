// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type TranscriptExport,
  exportFileName,
  exportTranscript,
  toJson,
  toSrt,
  toTxt,
  toVtt,
} from "../src/features/audio/transcript-export.ts";
import { EMPTY_TRANSCRIPT_DETAILS } from "../src/features/audio/transcript-model.ts";

const seg = (start: number, end: number, text: string, speaker?: string) => ({
  start,
  end,
  text,
  speaker,
});
const plain: TranscriptExport = {
  title: "note.m4a",
  text: "Just text.",
  model: "whisper-small",
  names: {},
  details: { ...EMPTY_TRANSCRIPT_DETAILS, duration: 4.5 },
};
const timed: TranscriptExport = {
  ...plain,
  names: { S02: "Bea" },
  details: {
    ...EMPTY_TRANSCRIPT_DETAILS,
    segments: [
      seg(1.234, 3, "Hello there.", "S01"),
      seg(3, 3.5, "Hi!", "S02"),
      seg(3661.001, 3662.25, "<b>Tom & Jerry</b>", "S02"),
    ],
    words: [{ start: 1.234, end: 1.6, word: "Hello" }],
    speakers: [
      { id: "S01", label: "Speaker 1" },
      { id: "S02", label: "Speaker 2" },
    ],
  },
};

test("SRT numbers cues with comma milliseconds; VTT has a header, voice tags and escaped text", () => {
  assert.equal(
    toSrt(timed),
    "1\n00:00:01,234 --> 00:00:03,000\nSpeaker 1: Hello there.\n\n" +
      "2\n00:00:03,000 --> 00:00:03,500\nBea: Hi!\n\n" +
      "3\n01:01:01,001 --> 01:01:02,250\nBea: <b>Tom & Jerry</b>\n",
  );
  assert.equal(
    toVtt(timed),
    "WEBVTT\n\n00:00:01.234 --> 00:00:03.000\n<v Speaker 1>Hello there.\n\n" +
      "00:00:03.000 --> 00:00:03.500\n<v Bea>Hi!\n\n" +
      "01:01:01.001 --> 01:01:02.250\n<v Bea>&lt;b&gt;Tom &amp; Jerry&lt;/b&gt;\n",
  );
});

test("cues: no names without speakers, no early end on a blank line, one cue over the clip without timing", () => {
  const unnamed = { ...timed, details: { ...timed.details, speakers: [] } };
  assert.doesNotMatch(toSrt(unnamed) + toVtt(unnamed), /Speaker 1|<v /);
  const segments = [seg(0, 1, "one\n\n  two"), seg(1, 2, "  "), seg(2, 3, "x")];
  const gaps = { ...plain, details: { ...plain.details, segments } };
  assert.equal(
    toSrt(gaps),
    "1\n00:00:00,000 --> 00:00:01,000\none\ntwo\n\n2\n00:00:02,000 --> 00:00:03,000\nx\n",
  );
  assert.equal(toSrt(plain), "1\n00:00:00,000 --> 00:00:04,500\nJust text.\n");
  assert.equal(
    toVtt(plain),
    "WEBVTT\n\n00:00:00.000 --> 00:00:04.500\nJust text.\n",
  );
  assert.equal(toSrt({ ...plain, text: "" }), "");
  assert.equal(toVtt({ ...plain, text: "" }), "WEBVTT\n");
});

test("TXT is the plain text or paragraphs led by the current names; JSON keeps keys, names and words", () => {
  assert.equal(toTxt(plain), "Just text.");
  assert.equal(
    toTxt(timed),
    "Speaker 1: Hello there.\n\nBea: Hi! <b>Tom & Jerry</b>\n",
  );
  const json = JSON.parse(toJson(timed));
  assert.deepEqual(
    Object.keys(json).join(),
    "title,model,language,duration,text,segments,words,speakers",
  );
  assert.deepEqual(json.speakers, [
    { id: "S01", label: "Speaker 1", name: null },
    { id: "S02", label: "Speaker 2", name: "Bea" },
  ]);
  assert.equal("words" in JSON.parse(toJson(plain)), false);
});

test("each format has its media type, and file names drop the extension and unsafe characters", () => {
  for (const [format, mime, write] of [
    ["txt", "text/plain;charset=utf-8", toTxt],
    ["srt", "application/x-subrip;charset=utf-8", toSrt],
    ["vtt", "text/vtt;charset=utf-8", toVtt],
    ["json", "application/json;charset=utf-8", toJson],
  ] as const)
    assert.deepEqual(exportTranscript(format, timed), {
      content: write(timed),
      mime,
    });
  for (const [title, name] of [
    ["meeting.wav", "meeting.srt"],
    ['a/b\\c:d*e?"f<g>h|i.mp3', "a_b_c_d_e__f_g_h_i.srt"],
    ["tab\there", "tab_here.srt"],
    ["", "transcript.srt"],
    [".wav", "transcript.srt"],
  ])
    assert.equal(exportFileName(title, "srt"), name);
});
