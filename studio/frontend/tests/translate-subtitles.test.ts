// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  numberedBatch,
  parseSubtitles,
  rebuildSubtitles,
  splitNumberedBatch,
  subtitleBatches,
  subtitleFileName,
  translateSubtitles,
} from "../src/features/translate/translate-subtitles.ts";

import { readSrc } from "./helpers/kit.ts";

const SRT = `1
00:00:01,000 --> 00:00:02,500
Hello there.

2
00:00:03,000 --> 00:00:05,000
- How are you?
- Fine.

3
00:00:06,000 --> 00:00:07,000
Goodbye.
`;

const VTT = `WEBVTT - Example
Kind: captions

NOTE a comment that is not a cue

intro
00:01.000 --> 00:02.000 align:start
<v Anna>Hello there.

00:00:03.000 --> 00:00:04.000
Goodbye.`;

const upper = (doc: NonNullable<ReturnType<typeof parseSubtitles>>) =>
  doc.cues.map((cue) => cue.text.toUpperCase());

test("an SRT file round trips exactly and keeps numbers and timings", () => {
  const doc = parseSubtitles(SRT);
  assert.ok(doc);
  assert.equal(doc.format, "srt");
  assert.deepEqual(
    doc.cues.map((cue) => cue.text),
    ["Hello there.", "- How are you?\n- Fine.", "Goodbye."],
  );
  assert.equal(rebuildSubtitles(doc, []), SRT);
  assert.equal(
    rebuildSubtitles(doc, upper(doc)),
    SRT.replace("Hello there.", "HELLO THERE.")
      .replace("- How are you?\n- Fine.", "- HOW ARE YOU?\n- FINE.")
      .replace("Goodbye.", "GOODBYE."),
  );
  assert.equal(
    rebuildSubtitles(doc, ["Hola."], 1),
    "1\n00:00:01,000 --> 00:00:02,500\nHola.",
  );
});

test("CRLF, a BOM and an SRT without cue numbers round trip", () => {
  const crlf = `\uFEFF${SRT.replaceAll("\n", "\r\n")}`;
  const doc = parseSubtitles(crlf);
  assert.ok(doc);
  assert.equal(rebuildSubtitles(doc, []), crlf);
  assert.equal(
    rebuildSubtitles(doc, ["A", "B\r\nC", "D"]).split("\r\n\r\n")[1],
    "2\r\n00:00:03,000 --> 00:00:05,000\r\nB\r\nC",
  );
  const bare =
    "00:00:01,000 --> 00:00:02,000\nHi\n\n00:00:03,000 --> 00:00:04,000\nBye";
  assert.equal(parseSubtitles(bare)?.cues.length, 2);
});

test("a WebVTT file keeps its header, notes, identifiers and cue settings", () => {
  const doc = parseSubtitles(VTT);
  assert.ok(doc);
  assert.equal(doc.format, "vtt");
  assert.deepEqual(
    doc.cues.map((cue) => cue.text),
    ["<v Anna>Hello there.", "Goodbye."],
  );
  assert.equal(rebuildSubtitles(doc, []), VTT);
  const out = rebuildSubtitles(doc, ["<v Anna>Hola.", "Adiós."]);
  assert.match(out, /^WEBVTT - Example\nKind: captions\n\nNOTE a comment/);
  assert.match(
    out,
    /intro\n00:01\.000 --> 00:02\.000 align:start\n<v Anna>Hola\./,
  );
});

test("plain text, prose with a timestamp and broken subtitles stay plain text", () => {
  for (const text of [
    "",
    "Hello world",
    "Meeting notes\n\nStarts 00:00:01,000 --> 00:00:02,000 sharp",
    "1\n00:00:01,000 --> 00:00:02,000\nHi\n\nA stray paragraph\n\n2\n00:00:03,000 --> 00:00:04,000\nBye",
    "WEBVTT\n\nno cues here",
  ]) {
    assert.equal(parseSubtitles(text), null, JSON.stringify(text));
  }
});

test("an SRT clipped inside its last block still parses and keeps the tail as is", () => {
  const clipped = `${SRT}\n4\n00:00:0`;
  const doc = parseSubtitles(clipped);
  assert.ok(doc);
  assert.equal(doc.cues.length, 3);
  assert.equal(rebuildSubtitles(doc, []), clipped);
});

test("batches hold consecutive cues with text within the passage size", () => {
  const texts = ["a".repeat(400), "", "b".repeat(400), "c".repeat(400), "d"];
  assert.deepEqual(subtitleBatches(texts, 1000), [
    [0, 2],
    [3, 4],
  ]);
  assert.deepEqual(subtitleBatches(["x".repeat(2000), "y"], 1000), [[0], [1]]);
});

test("a numbered reply splits back into its cues, or is refused", () => {
  assert.equal(
    numberedBatch(["Hello.", "- How are you?\n\n- Fine."]),
    "1. Hello.\n2. - How are you?\n- Fine.",
  );
  assert.deepEqual(
    splitNumberedBatch("1. Hola.\n\n2. - ¿Cómo estás?\n- Bien.", 2),
    ["Hola.", "- ¿Cómo estás?\n- Bien."],
  );
  assert.deepEqual(splitNumberedBatch("1、你好。\n2．再见。", 2), [
    "你好。",
    "再见。",
  ]);
  assert.equal(splitNumberedBatch("1. Hola. 2. Adiós.", 2), null);
  assert.equal(splitNumberedBatch("Hola.\nAdiós.", 2), null);
  assert.equal(splitNumberedBatch("1. Hola.\n3. Adiós.", 2), null);
  assert.equal(splitNumberedBatch("1. A\n2. B\n3. C", 2), null);
});

test("a batch the model merged is translated again cue by cue", async () => {
  const doc = parseSubtitles(SRT);
  assert.ok(doc);
  const sent: string[] = [];
  const outputs: string[] = [];
  const parts: string[] = [];
  const out = await translateSubtitles(
    doc,
    async (source) => {
      sent.push(source);
      // Merges the numbered batch into one line; single cues come back translated.
      return source.startsWith("1. ")
        ? "Hola. ¿Cómo estás? Bien. Adiós."
        : `ES ${source}`;
    },
    {
      signal: new AbortController().signal,
      onPhase: (phase) => parts.push(`${phase.part}/${phase.parts}`),
      onOutput: (output) => outputs.push(output),
    },
  );
  assert.equal(sent.length, 4);
  assert.deepEqual(sent.slice(1), [
    "Hello there.",
    "- How are you?\n- Fine.",
    "Goodbye.",
  ]);
  assert.deepEqual(parts, ["1/1"]);
  assert.equal(
    out,
    SRT.replace("Hello there.", "ES Hello there.")
      .replace("- How are you?", "ES - How are you?")
      .replace("Goodbye.", "ES Goodbye."),
  );
  assert.equal(outputs.at(-1), out);
});

test("an aligned batch is one request and keeps every timing", async () => {
  const doc = parseSubtitles(VTT);
  assert.ok(doc);
  let calls = 0;
  const out = await translateSubtitles(
    doc,
    async () => {
      calls += 1;
      return "1. <v Anna>Hola.\n2. Adiós.";
    },
    {
      signal: new AbortController().signal,
      onPhase: () => {},
      onOutput: () => {},
    },
  );
  assert.equal(calls, 1);
  assert.equal(
    out,
    VTT.replace("Hello there.", "Hola.").replace("Goodbye.", "Adiós."),
  );
});

test("the download is named after the opened file and the target language", () => {
  assert.equal(
    subtitleFileName("Movie.EN.srt", "fr", "srt"),
    "Movie.EN.fr.srt",
  );
  assert.equal(subtitleFileName("talk.vtt", "ja", "vtt"), "talk.ja.vtt");
  assert.equal(subtitleFileName(null, "de", "srt"), "subtitles.de.srt");
});

test("only text that parses as subtitles takes the cue path; other text is chunked as before", () => {
  const runner = readSrc("features/translate/translate-runner.ts");
  assert.match(
    runner,
    /const subtitles = parseSubtitles\(text\);\s*let output = "";\s*if \(subtitles\) \{\s*output = await translateSubtitles\(subtitles, complete,/,
  );
  assert.match(
    runner,
    /\} else \{\s*const chunks = splitForTranslation\(text\);/,
  );
  const page = readSrc("features/translate/translate-page.tsx");
  // .srt and .vtt still open through the text reader, so the box shows the file as saved.
  assert.match(page, /const TEXT_FILE_EXTENSIONS =\s*"[^"]*\.srt,\.vtt,/);
  assert.match(page, /loadText\(text, file\.name\);/);
  assert.match(
    page,
    /result && !stale && !busy\s*\?\s*\(parseSubtitles\(output\)\?\.format \?\? null\)/,
  );
});
