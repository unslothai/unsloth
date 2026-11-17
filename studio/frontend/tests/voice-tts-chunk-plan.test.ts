// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  CLAUSE_MAX_CHARS,
  CLAUSE_MIN_CHARS,
  CLAUSE_PAUSE_MS,
  firstClauseCut,
  pauseAfterChunk,
  planTtsChunk,
  TTS_CHUNK_MAX_CHARS,
  TTS_CHUNK_TARGET_CHARS,
  TTS_CHUNK_WAIT_MS,
} from "../src/features/chat/voice/tts-chunk-plan.ts";

const s = (n: number) => "x".repeat(n - 1) + ".";
const plan = (over: Partial<Parameters<typeof planTtsChunk>[0]>) =>
  planTtsChunk({ pending: [], first: false, final: false, playing: false, waitedMs: 0, ...over });

test("nothing pending is a wait", () => {
  assert.equal(plan({ pending: [] }), 0);
  assert.equal(plan({ pending: [], final: true }), 0);
});

test("the first chunk is the first sentence alone", () => {
  assert.equal(plan({ pending: [s(10), s(10), s(10)], first: true }), 1);
  assert.equal(plan({ pending: [s(400)], first: true, final: true }), 1);
});

test("later sentences merge up to the target", () => {
  assert.equal(plan({ pending: [s(100), s(100), s(100), s(100)] }), 3);
  assert.equal(plan({ pending: [s(TTS_CHUNK_TARGET_CHARS), s(10), s(10)] }), 1);
});

test("the hard cap closes a chunk and never splits a sentence", () => {
  assert.equal(plan({ pending: [s(200), s(150), s(10)] }), 1);
  assert.equal(plan({ pending: [s(TTS_CHUNK_MAX_CHARS + 50), s(10)] }), 1);
  assert.equal(plan({ pending: [s(TTS_CHUNK_MAX_CHARS + 50)], final: true }), 1);
});

test("a short tail is sent once the stream is final", () => {
  assert.equal(plan({ pending: [s(20), s(20)], final: true, playing: true }), 2);
  assert.equal(plan({ pending: [s(20)], final: true }), 1);
});

test("a short chunk waits for more while a clip is playing", () => {
  assert.equal(plan({ pending: [s(20), s(20)], playing: true }), 0);
  assert.equal(plan({ pending: [s(20)], playing: true, waitedMs: 10_000 }), 0);
});

test("an idle player waits at most TTS_CHUNK_WAIT_MS for a bigger chunk", () => {
  assert.equal(plan({ pending: [s(20)], waitedMs: 0 }), 0);
  assert.equal(plan({ pending: [s(20)], waitedMs: TTS_CHUNK_WAIT_MS - 1 }), 0);
  assert.equal(plan({ pending: [s(20), s(20)], waitedMs: TTS_CHUNK_WAIT_MS }), 2);
});

test("sentences stripped to nothing count but add no text", () => {
  assert.equal(plan({ pending: ["", s(20), ""], final: true }), 3);
});

const head = (text: string) => {
  const cut = firstClauseCut(text);
  return cut === null ? null : text.slice(0, cut);
};

test("the first clause is cut at each kind of boundary", () => {
  assert.equal(head("I checked the forecast for Paris, and it looks"), "I checked the forecast for Paris,");
  assert.equal(head("There are two ways to do this; the first is"), "There are two ways to do this;");
  assert.equal(head("Here is what I found for you: three"), "Here is what I found for you:");
  assert.equal(head("The train leaves at noon—unless it rains"), "The train leaves at noon—");
  assert.equal(head("The train leaves at noon – unless it rains"), "The train leaves at noon –");
  assert.equal(head("The train leaves at noon - unless it rains"), "The train leaves at noon -");
});

test("a bare conjunction is never a cut, only punctuation is", () => {
  assert.equal(head("I looked at the weather in Paris and it is sunny"), null);
  assert.equal(head("That model is a good choice but it needs more"), null);
  assert.equal(head("The download finished a minute ago so you can"), null);
  assert.equal(head("The ocean is salty because rocks and minerals, over time"), "The ocean is salty because rocks and minerals,");
  // A hyphen inside a word is not a dash.
  assert.equal(head("This is a well-known and widely used approach"), null);
});

test("a boundary needs the minimum before it, so the first one too early is passed over", () => {
  assert.equal(head("Sure, I checked the forecast for you, and"), "Sure, I checked the forecast for you,");
  const atMin = `${"a".repeat(CLAUSE_MIN_CHARS)}, then more`;
  assert.equal(head(atMin), `${"a".repeat(CLAUSE_MIN_CHARS)},`);
  assert.equal(head(`${"a".repeat(CLAUSE_MIN_CHARS - 1)}, then more`), null);
});

test("the head is capped, so a long first clause waits for its sentence", () => {
  assert.equal(firstClauseCut(`${"a".repeat(CLAUSE_MAX_CHARS - 1)}, more`), CLAUSE_MAX_CHARS);
  assert.equal(firstClauseCut(`${"a".repeat(CLAUSE_MAX_CHARS)}, more`), null);
  assert.equal(head(`${"word ".repeat(12)}, and then the rest`), null);
});

test("no cut inside numbers, times, words, a finished sentence, or an unsettled boundary", () => {
  assert.equal(head("The total comes to about 1,000 dollars in all"), null);
  assert.equal(head("The meeting is scheduled for 14:30 tomorrow"), null);
  assert.equal(head("This works on android phones and tablets"), null);
  assert.equal(head("That is the right answer. Then, after that"), null);
  // The comma is the last character so far: what follows might still be a digit.
  assert.equal(head("The total for the first quarter is 1,"), null);
  assert.equal(head("Short."), null);
  assert.equal(head(""), null);
});

test("a clause head is followed by the comma pause; a sentence is not", () => {
  for (const head of ["If you want a faster reply,", "That error usually means the path is wrong;", "One thing first:", "It was fast —", "It was fast -"]) {
    assert.equal(pauseAfterChunk(head), CLAUSE_PAUSE_MS, head);
  }
  for (const text of ["The capital is Canberra.", "Really?", "Wait!", "1,000", "Done", ""]) {
    assert.equal(pauseAfterChunk(text), 0, text);
  }
});

test("the player adds the pause after the clip it just scheduled", () => {
  const player = readFileSync(new URL("../src/features/chat/hooks/use-tts-player.ts", import.meta.url), "utf8");
  assert.match(player, /const start = schedule\(buffer, text, entries\);\s*\/\/[^\n]*\n\s*if \(start !== null\) playHeadRef\.current \+= pauseAfterChunk\(text\) \/ 1000;/);
});
