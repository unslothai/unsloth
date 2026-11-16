// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { markLiteralThinkTags, parseAssistantContent } = await import(
  "../src/features/chat/utils/parse-assistant-content.ts"
);
const { createSegmentedAssistantText } = await import(
  "../src/features/chat/utils/incremental-assistant-content.ts"
);

const TRACE = "Match <think> up to the first </think> lazily.";
const ANSWER = 'Use re.sub(r"<think>.*?</think>", "", text).';

function streamed(): string[] {
  return [
    "<think>",
    ...[...TRACE].map(markLiteralThinkTags),
    "</think>",
    ...[...ANSWER].map(markLiteralThinkTags),
  ];
}

test("literal tags in split-out reasoning and answer stay as written", () => {
  const segmented = createSegmentedAssistantText();
  let raw = "";
  for (const arrival of streamed()) {
    raw += arrival;
    segmented.appendText(arrival);
  }
  const expected = [
    { type: "reasoning", text: TRACE },
    { type: "text", text: ANSWER },
  ];
  assert.deepEqual(segmented.runs(raw, []), [expected]);
  assert.deepEqual(parseAssistantContent(raw), expected);
});

test("a tag split across arrivals is still literal", () => {
  const raw = `<think>plan</think>${markLiteralThinkTags("a <")}${markLiteralThinkTags("/think> b")}`;
  assert.deepEqual(parseAssistantContent(raw), [
    { type: "reasoning", text: "plan" },
    { type: "text", text: "a </think> b" },
  ]);
});

test("text without tags is left alone", () => {
  for (const text of ["a < b", "<div>", "x <thinking>", "<tool_call>"]) {
    assert.equal(markLiteralThinkTags(text), text);
  }
});
