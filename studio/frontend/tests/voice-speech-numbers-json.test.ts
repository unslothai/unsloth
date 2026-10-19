// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import {
  speakableReply,
  splitIntoSentences,
  splitStreaming,
  stripForSpeech,
} from "../src/features/chat/voice/speech-text.ts";

// use-tts-player's feedText/endStream: queue each newly complete sentence by index, flush the rest.
function speak(reply: string, step = 1): string[] {
  const queued: string[] = [];
  for (let i = step; i < reply.length + step; i += step) {
    const { complete } = splitStreaming(reply.slice(0, i));
    while (queued.length < complete.length) queued.push(complete[queued.length] ?? "");
  }
  const all = splitIntoSentences(speakableReply(reply));
  while (queued.length < all.length) queued.push(all[queued.length] ?? "");
  return queued.map(stripForSpeech).filter(Boolean);
}

test("a sentence ending in a year then one starting with a digit are both spoken", () => {
  const reply = "It opened in 2020. 30 people came. It was busy.";
  const expected = ["It opened in 2020.", "30 people came.", "It was busy."];
  assert.deepEqual(splitIntoSentences(reply), expected);
  assert.deepEqual(speak(reply), expected);
});

test("every streamed prefix agrees with the finished split across a digit boundary", () => {
  const reply = "It opened in 2020. 30 people came. It was busy.";
  const final = splitStreaming(reply).complete;
  for (let i = 1; i <= reply.length; i++) {
    const { complete } = splitStreaming(reply.slice(0, i));
    assert.deepEqual(complete, final.slice(0, complete.length), `prefix ${i}`);
  }
});

test("a decimal with no space still reads as one number, and holds mid-stream", () => {
  const reply = "It is about 2.5 centimeters long. Small.";
  assert.deepEqual(splitIntoSentences(reply), ["It is about 2.5 centimeters long.", "Small."]);
  assert.deepEqual(splitStreaming("It is about 2.").complete, []);
  assert.deepEqual(speak(reply), ["It is about 2.5 centimeters long.", "Small."]);
});

test("a number ending a line is not glued to a number starting the next", () => {
  assert.deepEqual(splitIntoSentences("Founded in 2020.\n30 people came."), [
    "Founded in 2020.",
    "30 people came.",
  ]);
});

test("an unfenced JSON block is not voiced, streamed or whole", () => {
  const reply = 'Here it is:\n{\n  "name": "Ada",\n  "tags": ["a.b", "c"],\n  "nested": {\n    "on": true\n  }\n}\nDone.';
  const spoken = speak(reply);
  assert.deepEqual(spoken, ["Here it is:", "Done."]);
  const array = '[\n  {"id": 1},\n  {"id": 2}\n]\nThat is all.';
  assert.deepEqual(speak(array), ["That is all."]);
});

test("lone key/value lines are dropped, streamed char by char", () => {
  const reply = 'Settings:\n"name": "Ada",\n"count": 3.5,\n"enabled": false\nAll set.';
  assert.deepEqual(speak(reply), ["Settings:", "All set."]);
});

test("prose with quotes and a colon is still voiced", () => {
  const reply = '"Rule one": "Be kind," she said. Then she left.\nHe said "no": that was it.\n"It works. Try it."';
  const spoken = speak(reply);
  assert.deepEqual(spoken, [
    '"Rule one": "Be kind," she said.',
    "Then she left.",
    'He said "no": that was it.',
    '"It works.',
    'Try it."',
  ]);
});
