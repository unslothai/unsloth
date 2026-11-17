// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Phase 4 parity: splitSpoken, and the wrappers over it, give the sentences the old splitter gave
// on every string the speech tests use and on every streamed prefix of them.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import * as next from "../src/features/chat/voice/speech-text.ts";
import * as v1 from "./fixtures/voice-speech-text-v1.ts";

const SOURCES = [
  "voice-split-streaming.test.ts",
  "voice-sentence-chunker.test.ts",
  "voice-speakable-text.test.ts",
  "voice-speech-numbers-json.test.ts",
  "read-aloud-sentence-chunks.test.ts",
  "voice-tts-first-clause.test.ts",
  "voice-tts-chunk-plan.test.ts",
];

function literals(source: string): string[] {
  const out: string[] = [];
  for (const m of source.matchAll(/"((?:[^"\\\n]|\\.)*)"|'((?:[^'\\\n]|\\.)*)'|`((?:[^`\\$]|\\.)*)`/g)) {
    const body = m[1] ?? m[2] ?? m[3] ?? "";
    try {
      out.push(JSON.parse(`"${body.replace(/\\'/g, "'").replace(/"/g, '\\"').replace(/\n/g, '\\n')}"`));
    } catch {
      out.push(body);
    }
  }
  return out;
}

const EXTRA = [
  "Check the logs.<think>maybe the disk</think> The disk is full. Free some space.",
  "Run this:\n```bash\npip install unsloth\n```\nThen restart. Done.",
  "Inline ```x = 1``` works. Next one.",
  '{"a": 1,\n "b": [1, 2]}\nThe JSON is above. Read it.',
  "1. First step.\n2. Second step!\n- a bullet\n* another one",
  "<|channel|>analysis<|message|>hidden<|channel|>final<|message|>Shown text here. More.",
  "Dr. Smith went to Washington D.C. on Jan. 5. He left at 3.5 p.m. Then etc. Then he slept.",
  "   \n\n  ",
  "",
  "... -- !!",
  "Trailing spaces.   \n\nNext line   ",
  "<think>unfinished thought",
  "A " + "very ".repeat(60) + "long sentence without an end",
];

const corpus = [...new Set([
  ...SOURCES.flatMap((f) => literals(readFileSync(new URL(f, import.meta.url), "utf8"))),
  ...EXTRA,
])].filter((t) => t.length < 4000);

const prefixes = (t: string) => {
  const step = Math.max(1, Math.floor(t.length / 60));
  const out: string[] = [];
  for (let i = 0; i <= t.length; i += step) out.push(t.slice(0, i));
  out.push(t);
  return out;
};

test("the corpus covers the speech tests", () => {
  assert.ok(corpus.length > 150, String(corpus.length));
});

// Whitespace-only is nothing to say either way; the old splitter returned the leftover whitespace.
const blankAsEmpty = (parts: string[]) => parts.map((p) => (/\S/.test(p) ? p : ""));
const streamed = (r: { complete: string[]; partial: string }) => ({
  complete: blankAsEmpty(r.complete),
  partial: /\S/.test(r.partial) ? r.partial : "",
});

test("splitSpoken gives the old sentences, on every case and every streamed prefix", () => {
  let checked = 0;
  for (const text of corpus) {
    for (const t of prefixes(text)) {
      assert.deepEqual(
        next.splitSpoken(t).map((s) => s.text),
        blankAsEmpty(v1.splitIntoSentences(v1.speakableReply(t))),
        JSON.stringify(t),
      );
      assert.deepEqual(streamed(next.splitStreaming(t)), streamed(v1.splitStreaming(t)), JSON.stringify(t));
      assert.deepEqual(next.splitIntoSentences(t), v1.splitIntoSentences(t), JSON.stringify(t));
      checked++;
    }
  }
  assert.ok(checked > 1000, String(checked));
});

test("offsets point into the reply and never into a blanked region", () => {
  for (const text of corpus) {
    const blanked = next.speakableReply(text);
    assert.equal(blanked.length, text.length);
    for (const s of next.splitSpoken(text)) {
      if (!/\S/.test(s.text)) continue;
      assert.ok(0 <= s.start && s.start < s.end && s.end <= text.length, JSON.stringify([text, s]));
      // The first and last characters are spoken text, kept exactly as written.
      assert.notEqual(blanked[s.start], " ");
      assert.equal(blanked[s.start], text[s.start]);
      assert.equal(blanked[s.end - 1], text[s.end - 1]);
      assert.ok(s.text.startsWith(text[s.start]) && s.text.endsWith(text[s.end - 1]), JSON.stringify([text, s]));
    }
  }
});

test("a cut at a sentence end never lands inside a code fence or a think block", () => {
  const text = "First part.<think>a. b. c.</think> Second part. ```\nx. y.\n```\nThird part.";
  const ends = next.splitSpoken(text).map((s) => s.end);
  for (const end of ends) {
    const before = text.slice(0, end);
    assert.equal((before.match(/<think>/g) ?? []).length, (before.match(/<\/think>/g) ?? []).length);
    assert.equal((before.match(/```/g) ?? []).length % 2, 0);
  }
  assert.deepEqual(next.splitSpoken(text).map((s) => s.text), ["First part.", "Second part.", "Third part."]);
});
