// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  isToolCallCancelled,
  isToolCallRunning,
  knowledgeBaseToolName,
  toolFallbackLabel,
} from "../src/components/assistant-ui/tool-arg-text.ts";

test("a call still in flight says it is being used", () => {
  assert.equal(toolFallbackLabel({ type: "running" }), "Using tool");
  assert.equal(isToolCallRunning({ type: "running" }), true);
});

// A resultless part inherits the message's terminal status; restored history has none.
test("every settled status says the call is over", () => {
  assert.equal(toolFallbackLabel({ type: "complete" }), "Used tool");
  assert.equal(toolFallbackLabel(undefined), "Used tool");
  assert.equal(
    toolFallbackLabel({ type: "requires-action", reason: "interrupt" }),
    "Used tool",
  );
  assert.equal(
    toolFallbackLabel({ type: "incomplete", reason: "error" }),
    "Used tool",
  );
  assert.equal(
    toolFallbackLabel({ type: "incomplete", reason: "length" }),
    "Used tool",
  );
  assert.equal(isToolCallRunning({ type: "complete" }), false);
  assert.equal(isToolCallRunning(undefined), false);
});

test("cancellation outranks the in-flight wording", () => {
  assert.equal(
    toolFallbackLabel({ type: "incomplete", reason: "cancelled" }),
    "Cancelled tool",
  );
  assert.equal(
    isToolCallCancelled({ type: "incomplete", reason: "cancelled" }),
    true,
  );
  // The other incomplete reasons are failures, which keep the plain label.
  assert.equal(
    isToolCallCancelled({ type: "incomplete", reason: "error" }),
    false,
  );
  assert.equal(isToolCallCancelled({ type: "running" }), false);
});

test("a running search is never named in the past tense", () => {
  assert.equal(
    knowledgeBaseToolName({ isRunning: true, query: "adapters" }),
    'Searching documents for "adapters"…',
  );
  assert.equal(
    knowledgeBaseToolName({ isRunning: true, query: "" }),
    "Searching documents…",
  );
});

test("a finished search keeps the wording it shipped with", () => {
  assert.equal(
    knowledgeBaseToolName({ isRunning: false, query: "adapters" }),
    'Searched documents for "adapters"',
  );
  assert.equal(
    knowledgeBaseToolName({ isRunning: false, query: "" }),
    "Knowledge search",
  );
});

test("the query reaches the header, whichever tense it is in", () => {
  for (const isRunning of [true, false]) {
    assert.match(
      knowledgeBaseToolName({ isRunning, query: "quantisation" }),
      /quantisation/,
    );
    assert.doesNotMatch(
      knowledgeBaseToolName({ isRunning, query: "" }),
      /""|undefined/,
    );
  }
});
