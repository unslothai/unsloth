// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import {
  splitIntoSentences,
  stripForSpeech,
  splitStreaming,
} from "../src/features/chat/voice/speech-text.ts";

test("splits on sentence punctuation", () => {
  assert.deepEqual(splitIntoSentences("Hello! How can I assist you today?"), [
    "Hello!",
    "How can I assist you today?",
  ]);
});

test("apostrophes are not boundaries", () => {
  assert.deepEqual(
    splitIntoSentences("I'm sorry, but I can't ask questions."),
    ["I'm sorry, but I can't ask questions."],
  );
  assert.deepEqual(splitIntoSentences("Don't worry. It's fine!"), [
    "Don't worry.",
    "It's fine!",
  ]);
});

test("a decimal point is not a sentence end", () => {
  assert.deepEqual(splitIntoSentences("It cost $3.50 today."), [
    "It cost $3.50 today.",
  ]);
});

test("abbreviations keep their sentence", () => {
  assert.deepEqual(splitIntoSentences("Dr. Smith went home."), [
    "Dr. Smith went home.",
  ]);
});

test("a closing quote stays with the sentence it closes", () => {
  assert.deepEqual(splitIntoSentences('He said "stop." Then he left.'), [
    'He said "stop."',
    "Then he left.",
  ]);
});

test("an opening quote still starts a new chunk", () => {
  assert.deepEqual(splitIntoSentences('Ready?\n"Lice, lice, lice."'), [
    "Ready?",
    '"Lice, lice, lice."',
  ]);
});

test("newlines are boundaries, so a list speaks a line at a time", () => {
  assert.deepEqual(splitIntoSentences("Here is a list:\nitem one\nitem two"), [
    "Here is a list:",
    "item one",
    "item two",
  ]);
});

test("stripForSpeech keeps the newlines the splitter needs", () => {
  // Regression: collapsing every \s+ merged paragraphs into one synth request.
  const reply = 'Sure! Here it is:\n\n**"Lice, lice, lice."**\n\nLet me know!';
  const spoken = stripForSpeech(reply);
  assert.ok(spoken.includes("\n"), "paragraph breaks must survive stripping");
  assert.deepEqual(splitIntoSentences(spoken), [
    "Sure!",
    "Here it is:",
    '"Lice, lice, lice."',
    "Let me know!",
  ]);
});

test("stripForSpeech still drops markup and emoji", () => {
  assert.equal(stripForSpeech("**bold** and `code` 😀"), "bold and code");
});

test("a Unicode minus before a number survives as a minus", () => {
  assert.equal(stripForSpeech("The temperature is −5 degrees."), "The temperature is -5 degrees.");
  assert.equal(stripForSpeech("The result is −2."), "The result is -2.");
  assert.equal(stripForSpeech("Paris — the capital."), "Paris the capital.");
});

test("a streamed reply never queues a sentence the finished text splits differently", () => {
  const replies = [
    "Dr. Smith went home. He slept.",
    "The price is 3.5 dollars. That is cheap.",
    "Ask J. Smith about it. Then call me.",
    "Pick option A. Then wait.",
    "We met in the U.S. last year. It rained!",
    "Wait... what? Yes.",
  ];
  for (const reply of replies) {
    const final = splitIntoSentences(reply);
    for (let i = 1; i <= reply.length; i++) {
      const { complete } = splitStreaming(reply.slice(0, i));
      assert.deepEqual(complete, final.slice(0, complete.length), `${reply} @ ${i}`);
    }
  }
});

test("the paused player does not resume itself when the next clip is scheduled", () => {
  const player = readFileSync(new URL("../src/features/chat/hooks/use-tts-player.ts", import.meta.url), "utf8");
  const schedule = player.slice(player.indexOf("const schedule = useCallback("));
  assert.match(schedule.slice(0, 600), /if \(ctx\.state !== "running" && !pausedRef\.current\) void ctx\.resume\(\)/);
});

