// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { parseAssistantContent } from "../src/features/chat/utils/parse-assistant-content.ts";

const reasoning = (raw: string): string =>
  parseAssistantContent(raw)
    .filter((part) => part.type === "reasoning")
    .map((part) => (part as { text: string }).text)
    .join("");

const answer = (raw: string): string =>
  parseAssistantContent(raw)
    .filter((part) => part.type === "text")
    .map((part) => (part as { text: string }).text)
    .join("");

// A serialized quotation escapes both quotes, so both are excluded from the
// parity count and the mention read as the structural close: the drawer shut on
// the first tag and the rest of the thought was rendered as the answer (#7334).
// The backend extractor has the same escaped-pair case (_is_literal_think_close).
test("a symmetric escaped pair stays inside the reasoning drawer", () => {
  const raw = '<think>serialized \\"</think>\\" still reasoning</think>answer';
  assert.equal(reasoning(raw), 'serialized \\"</think>\\" still reasoning');
  assert.equal(answer(raw), "answer");
});

test("an unescaped quoted pair still reads as a mention", () => {
  const raw = '<think>quoted "</think>" still reasoning</think>answer';
  assert.equal(reasoning(raw), 'quoted "</think>" still reasoning');
  assert.equal(answer(raw), "answer");
});

// The escaped-pair case must not swallow real closes: the checks that already
// resolve a quoted tag as structural still win.
test("a bare close tag is still structural", () => {
  assert.equal(reasoning("<think>draft</think>answer"), "draft");
  assert.equal(answer("<think>draft</think>answer"), "answer");
});

test("an escaped closing quote running into a word still opens the answer", () => {
  const raw = '<think>a \\"</think>\\"The answer is 42.';
  assert.equal(reasoning(raw), 'a \\"');
  assert.equal(answer(raw), '\\"The answer is 42.');
});

test("mismatched delimiter runs are still structural", () => {
  const raw = "<think>`</think>```python\ncode";
  assert.equal(reasoning(raw), "`");
  assert.equal(answer(raw), "```python\ncode");
});
