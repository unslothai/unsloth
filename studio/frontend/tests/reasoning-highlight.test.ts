// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { reasoningHighlightReply } from "../src/components/assistant-ui/reasoning-highlight.ts";

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
