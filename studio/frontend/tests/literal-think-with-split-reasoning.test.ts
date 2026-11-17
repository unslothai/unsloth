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
    ...[...TRACE].map((char) => markLiteralThinkTags(char)),
    "</think>",
    ...[...ANSWER].map((char) => markLiteralThinkTags(char)),
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
  for (const cut of ["a <", "a <th"]) {
    const rest = "a <think> b".slice(cut.length);
    const raw = `<think>plan</think>${markLiteralThinkTags(cut)}${markLiteralThinkTags(rest)}`;
    assert.deepEqual(parseAssistantContent(raw), [
      { type: "reasoning", text: "plan" },
      { type: "text", text: "a <think> b" },
    ]);
  }
});

test("text without tags is left alone", () => {
  for (const text of ["a < b", "<div>", "x <thinking>", "<tool_call>"]) {
    assert.equal(markLiteralThinkTags(text), text);
  }
});

const { generationRawContent, restoreCarriedPartsFromRaw, restoreCarriedParts } =
  await import("../src/features/chat/utils/chat-generation-recovery.ts");
const { continuationSeed } = await import("../src/features/chat/utils/continuation.ts");
const { extractDeltaText } = await import(
  "../src/features/chat/utils/parse-assistant-content.ts"
);

const SAVED = [
  { type: "reasoning", text: TRACE },
  { type: "text", text: "See " },
  { type: "tool-call", toolCallId: "c1" },
  { type: "text", text: ANSWER },
];

test("saved parts rebuilt for recovery keep their literal tags", () => {
  const { raw, carried } = generationRawContent(SAVED);
  assert.deepEqual(restoreCarriedPartsFromRaw(raw, carried), SAVED);
  assert.deepEqual(
    restoreCarriedParts(
      SAVED.filter((part) => part.type !== "tool-call"),
      carried,
    ),
    SAVED,
  );
});

test("a continuation seed keeps the saved thought and answer as written", () => {
  assert.deepEqual(parseAssistantContent(continuationSeed(ANSWER, TRACE)), [
    { type: "reasoning", text: TRACE },
    { type: "text", text: ANSWER },
  ]);
  assert.deepEqual(parseAssistantContent(`${continuationSeed("", TRACE)}more`), [
    { type: "reasoning", text: `${TRACE}more` },
  ]);
});

test("a structured thinking part keeps a literal closing tag inside it", () => {
  const { text } = extractDeltaText([{ type: "thinking", thinking: TRACE }]);
  assert.deepEqual(parseAssistantContent(text), [{ type: "reasoning", text: TRACE }]);
});

test("a replay overlapping or restarting a marked continuation seed is still recognised", async () => {
  const { joinContinuation } = await import("../src/features/chat/utils/continuation.ts");
  const partial = "Use literal <think> tags to find the block you want removed.";
  const seed = continuationSeed(partial, "");
  assert.equal(
    joinContinuation(seed, "<think> tags to find the block you want removed. Then", {
      streaming: true,
    }),
    markLiteralThinkTags(`${partial} Then`),
  );
  const restart = markLiteralThinkTags(`${partial.slice(0, 52)}rewritten.`);
  assert.equal(joinContinuation(seed, restart), restart);
});
