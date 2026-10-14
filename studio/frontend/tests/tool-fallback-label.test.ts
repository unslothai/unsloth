// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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

// A document search can carry a whole page as its query; wrapping it into the open header pushed the
// icon and chevron to the middle of a tall block and the retrieval summary below it.
test("the document search header stays one line, with the summary first and the long query below", () => {
  const kb = readFileSync(
    new URL(
      "../src/components/assistant-ui/tool-ui-knowledge-base.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(kb, /singleLine=\{true\}/);
  assert.match(kb, /icon=\{LibraryGlyph\}/);
  assert.match(kb, /icon=\{LibrariesIcon\}/);
  const summary = kb.indexOf("Retrieved {citations.length} passage");
  const fullQuery = kb.indexOf('data-slot="knowledge-base-query"');
  assert.ok(
    summary >= 0 && fullQuery >= 0,
    "both the summary and the full query render",
  );
  assert.ok(summary < fullQuery);
  // The query renders after every outcome (hits, raw result, no hits), not only with citations.
  assert.ok(kb.indexOf("No matching passages.") < fullQuery);
  const fallback = readFileSync(
    new URL(
      "../src/components/assistant-ui/tool-fallback.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(fallback, /const wrapWhenOpen = !singleLine;/);
});
