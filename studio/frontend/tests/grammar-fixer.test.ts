// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { fixGrammar } from "../src/features/voice-typing/grammar-fixer.ts";

const cases: [string, string][] = [
  ["i think the the model is a good idea", "I think the model is a good idea"],
  ["This is an unsloth model", "This is an unsloth model"],
  ["I have a apple and an banana", "I have an apple and a banana"],
  ["It was an hour ago and a university", "It was an hour ago and a university"],
  ["We use a LLM and an GPU", "We use an LLM and a GPU"],
  ["i dont know, i cant say", "I don't know, I can't say"],
  ["You should of called", "You should have called"],
  ["your welcome and your going to love it", "You're welcome and you're going to love it"],
  ["their is a problem", "There is a problem"],
  ["its a good day", "It's a good day"],
  ["this is better then that", "This is better than that"],
  ["hello  world ,this is fine.Thanks", "Hello world, this is fine. Thanks"],
  ["open github and youtube", "Open GitHub and YouTube"],
  ["visit https://example.com/the_the now", "Visit https://example.com/the_the now"],
  ["he don't care and they was late", "He doesn't care and they were late"],
  ["I had had enough", "I had had enough"],
];

for (const [input, expected] of cases) {
  test(`fixGrammar: ${input}`, () => {
    assert.equal(fixGrammar(input, { language: "en" }), expected);
  });
}

test("fixGrammar leaves other languages untouched", () => {
  const german = "das ist ein test der der";
  assert.equal(fixGrammar(german, { language: "de" }), german);
  const arabic = "هذا اختبار";
  assert.equal(fixGrammar(arabic), arabic);
});
