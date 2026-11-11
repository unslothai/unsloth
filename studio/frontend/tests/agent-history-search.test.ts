// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { findAgentTextMatches } from "../src/components/assistant-ui/agent-history-search.ts";

const transcript = Array.from({ length: 400 }, (_, i) => ({
  content: [{ type: "text", text: `agent-history-e2e-row-${String(i).padStart(5, "0")} unique-fixture` }],
}));

test("matches a historical message absent from the final 32 mounted rows", () => {
  assert.deepEqual(findAgentTextMatches(transcript, "row-00005"), {
    matches: [{ row: 5, offset: 18 }], capped: false,
  });
});

test("searches every row, counts repeated occurrences and excludes non-text parts", () => {
  const entries = [
    { content: [{ type: "text", text: "Needle, needle!" }, { type: "reasoning", text: "needle" }] },
    { content: [{ type: "tool-call", text: "needle" }] },
    { content: [{ type: "text", text: "NEEDLE" }] },
  ];
  assert.deepEqual(findAgentTextMatches(entries, "needle").matches, [
    { row: 0, offset: 0 }, { row: 0, offset: 8 }, { row: 2, offset: 0 },
  ]);
});

test("empty query and match cap do not scan indefinitely or report a false total", () => {
  assert.deepEqual(findAgentTextMatches(transcript, ""), { matches: [], capped: false });
  const limited = findAgentTextMatches([{ content: "a a a a a" }], "a", 2);
  assert.equal(limited.matches.length, 2);
  assert.equal(limited.capped, true);
});
