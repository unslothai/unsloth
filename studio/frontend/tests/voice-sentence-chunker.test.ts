// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import { test } from "node:test";

import {
  splitIntoSentences,
  stripForSpeech,
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
