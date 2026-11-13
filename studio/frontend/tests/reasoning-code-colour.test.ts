// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  cachedSettledColour,
  clipLineTokens,
  keepUnchangedLines,
  rememberSettledColour,
  spellLines,
} from "../src/components/assistant-ui/reasoning-code-colour.ts";

const token = (content: string, color = "#111") => ({ content, offset: 0, color });

test("a row that starts mid-line takes only the tokens it shows", () => {
  const line = [token("const "), token("value"), token(" = 1;")];
  const clipped = clipLineTokens(line as never, 3, "st value");
  assert.deepEqual(clipped?.map((t) => t.content), ["st ", "value"]);
  assert.equal(clipped?.[1], line[1], "an untouched token keeps its identity");
});

test("tokens that do not spell the row are refused, so stale colours never show", () => {
  assert.equal(clipLineTokens([token("old")] as never, 0, "new"), null);
  assert.equal(clipLineTokens(undefined, 0, "x"), null);
});

test("settled colours outlive their rows and are keyed on language and source", () => {
  const result = { tokens: [[token("a")]] } as never;
  rememberSettledColour("python", "a", result);
  assert.equal(cachedSettledColour("python", "a"), result);
  assert.equal(cachedSettledColour("python", "a "), null);
  assert.equal(cachedSettledColour("ts", "a"), null);
});

test("a re-coloured growing fence keeps every unchanged line's identity", () => {
  const first = { tokens: [[token("a")], [token("b", "#222")]] } as never;
  const second = {
    tokens: [[token("a")], [token("b", "#333")], [token("c")]],
  } as never;
  const kept = keepUnchangedLines(first, second) as { tokens: unknown[][] };
  const before = first as { tokens: unknown[][] };
  const after = second as { tokens: unknown[][] };
  assert.equal(kept.tokens[0], before.tokens[0], "same tokens: the old line is reused");
  assert.equal(kept.tokens[1], after.tokens[1], "a recoloured line is replaced");
  assert.equal(kept.tokens[2], after.tokens[2]);
  assert.equal(keepUnchangedLines(null, second), second);
});

test("colours for a shorter source are kept only on the lines they still spell", () => {
  const line0 = [token("def "), token("f():")];
  const result = { tokens: [line0, [token("  ret")]] } as never;
  const spelled = spellLines(result, ["def f():", "  return 1", "x"]) as {
    tokens: { content: string; color?: string }[][];
  };
  assert.equal(spelled.tokens[0], line0, "a finished line keeps its colours and its identity");
  assert.deepEqual(spelled.tokens[1].map((t) => [t.content, t.color]), [["  return 1", undefined]], "the growing line is plain");
  assert.deepEqual(spelled.tokens[2].map((t) => t.content), ["x"], "a line the worker has not seen is plain");
});

test("evicting a settled colour returns exactly what it added", () => {
  for (let i = 0; i < 200; i += 1) rememberSettledColour("ts", `${i}`.padStart(9_000, "x"), { tokens: [] } as never);
  assert.notEqual(cachedSettledColour("ts", "199".padStart(9_000, "x")), null, "the newest entry survives");
  assert.equal(cachedSettledColour("ts", "0".padStart(9_000, "x")), null, "the oldest is gone");
  rememberSettledColour("ts", "small", { tokens: [] } as never);
  assert.notEqual(cachedSettledColour("ts", "small"), null);
});
