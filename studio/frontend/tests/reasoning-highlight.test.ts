// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  orderHighlightRequests,
  reasoningHighlightFailure,
  reasoningHighlightReply,
  reasoningHighlightSource,
} from "../src/components/assistant-ui/reasoning-highlight.ts";

test("worker sources accept ordered deltas, viewport-only requests, and resets", () => {
  let source = reasoningHighlightSource("", "const bird = ");
  source = reasoningHighlightSource(source, {
    from: source.length,
    text: "true;",
  });
  assert.equal(source, "const bird = true;");
  assert.equal(
    reasoningHighlightSource(source, { from: source.length, text: "" }),
    source,
  );
  assert.equal(reasoningHighlightSource(source, "new source"), "new source");
});

test("highlighting transfers only requested lines, preserving grammar tokens", () => {
  const tokens = Array.from({ length: 10000 }, (_, i) => [
    { content: `line ${i}`, color: "red", offset: i },
  ]);
  const reply = reasoningHighlightReply(
    {
      client: 1,
      revision: 8,
      source: "unused",
      language: "js",
      lines: [4, 9999, 10000],
    },
    { tokens },
  );
  assert.deepEqual(
    reply.lines.map((line) => line.line),
    [4, 9999],
  );
  assert.equal(reply.lines[1].tokens, tokens[9999]);
  assert.equal(reply.revision, 8);
});

test("unavailable grammar produces no stale highlighted lines", () => {
  assert.deepEqual(
    reasoningHighlightReply(
      { client: 3, revision: 4, source: "code", language: null, lines: [0] },
      null,
    ),
    { client: 3, revision: 4, lines: [] },
  );
});

test("a failed request is reported without tokens and never by the ordinary reply", () => {
  const failure = reasoningHighlightFailure(3, 4);
  assert.deepEqual(failure, { client: 3, revision: 4, lines: [], failed: true });
  assert.equal("result" in failure, false);
  const reply = reasoningHighlightReply(
    { client: 1, revision: 1, source: "x", language: null, lines: [], full: true },
    null,
  );
  assert.equal("failed" in reply, false);
});

test("demanded requests go before speculative ones, each group in arrival order", () => {
  const ordered = orderHighlightRequests([
    { client: 1, speculative: true },
    { client: 2 },
    { client: 3, speculative: true },
    { client: 4, speculative: false },
  ]);
  assert.deepEqual(ordered.map((request) => request.client), [2, 4, 1, 3]);
  assert.deepEqual(orderHighlightRequests([]), []);
});
