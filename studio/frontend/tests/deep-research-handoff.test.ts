// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The handoff is read off the tool events every loop publishes, and those events also close a
// call that never ran. Reading a run out of one spends the chat's single Deep Research on a
// question the loop refused to pass on, and hiding a gated call's card hangs the turn on a
// verdict the user is never asked for.

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEEP_RESEARCH_QUESTION_MAX_CHARS,
  DEEP_RESEARCH_STARTED_MARKER,
  newDeepResearchHandoff,
  readDeepResearchToolEvent,
} from "../src/features/chat/utils/deep-research-handoff.ts";

const STARTED = `${DEEP_RESEARCH_STARTED_MARKER} on that question. Reply with one short sentence.`;
const QUESTION = "Which small dog breeds suit a flat with no garden?";

const start = (over: Record<string, unknown> = {}) => ({
  type: "tool_start",
  tool_call_id: "call_r",
  arguments: { question: QUESTION },
  ...over,
});

const end = (over: Record<string, unknown> = {}) => ({
  type: "tool_end",
  tool_call_id: "call_r",
  result: STARTED,
  ...over,
});

test("a call that ran hands off its question and draws no tool card", () => {
  const handoff = newDeepResearchHandoff();
  assert.equal(readDeepResearchToolEvent(handoff, start()), true);
  assert.equal(handoff.question, null);
  assert.equal(readDeepResearchToolEvent(handoff, end()), true);
  assert.equal(handoff.question, QUESTION);
});

test("the provisional start the local loop paints first is ignored", () => {
  const handoff = newDeepResearchHandoff();
  readDeepResearchToolEvent(
    handoff,
    start({ arguments: {}, tool_call_id: "" }),
  );
  readDeepResearchToolEvent(handoff, start());
  readDeepResearchToolEvent(handoff, end());
  assert.equal(handoff.question, QUESTION);
});

test("a start whose question could not be read falls back to the user's message", () => {
  const handoff = newDeepResearchHandoff();
  readDeepResearchToolEvent(handoff, start({ arguments: undefined }));
  readDeepResearchToolEvent(handoff, end());
  assert.equal(handoff.question, "");
});

test("a denied call is not a handoff, and keeps the card that asked", () => {
  const handoff = newDeepResearchHandoff();
  // A gated start has to reach the renderer, or the Allow / Deny buttons never paint and the
  // loop blocks on a verdict for the rest of the turn.
  assert.equal(
    readDeepResearchToolEvent(handoff, start({ awaiting_confirmation: true })),
    false,
  );
  assert.equal(
    readDeepResearchToolEvent(
      handoff,
      end({ result: "Tool call rejected by user." }),
    ),
    false,
  );
  assert.equal(handoff.question, null);
});

test("an approved gated call hands off and closes its own card", () => {
  const handoff = newDeepResearchHandoff();
  readDeepResearchToolEvent(handoff, start({ awaiting_confirmation: true }));
  assert.equal(readDeepResearchToolEvent(handoff, end()), false);
  assert.equal(handoff.question, QUESTION);
});

test("a call the loop announced but never ran is not a handoff", () => {
  for (const result of [
    "Tool call budget for this message is exhausted.",
    "Studio did not run this call.",
    "Tool call was cut off mid-write.",
  ]) {
    const handoff = newDeepResearchHandoff();
    readDeepResearchToolEvent(handoff, start());
    readDeepResearchToolEvent(handoff, end({ result }));
    assert.equal(handoff.question, null, result);
  }
});

test("the question is clamped to what the endpoint accepts", () => {
  const handoff = newDeepResearchHandoff();
  const long = "x".repeat(DEEP_RESEARCH_QUESTION_MAX_CHARS + 500);
  readDeepResearchToolEvent(handoff, start({ arguments: { question: long } }));
  readDeepResearchToolEvent(handoff, end());
  assert.equal(handoff.question?.length, DEEP_RESEARCH_QUESTION_MAX_CHARS);
});

test("the question clamp does not split an astral Unicode character", () => {
  const handoff = newDeepResearchHandoff();
  const prefix = "x".repeat(DEEP_RESEARCH_QUESTION_MAX_CHARS - 1);
  readDeepResearchToolEvent(
    handoff,
    start({ arguments: { question: `${prefix}\u{1f600}tail` } }),
  );
  readDeepResearchToolEvent(handoff, end());
  assert.equal(handoff.question, `${prefix}\u{1f600}`);
  assert.equal(
    Array.from(handoff.question ?? "").length,
    DEEP_RESEARCH_QUESTION_MAX_CHARS,
  );
});

test("a second call in the same turn is the model repeating itself", () => {
  const handoff = newDeepResearchHandoff();
  readDeepResearchToolEvent(handoff, start());
  readDeepResearchToolEvent(handoff, end());
  readDeepResearchToolEvent(
    handoff,
    start({
      tool_call_id: "call_2",
      arguments: { question: "something else" },
    }),
  );
  readDeepResearchToolEvent(handoff, end({ tool_call_id: "call_2" }));
  assert.equal(handoff.question, QUESTION);
});
