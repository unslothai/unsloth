// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  cachedSettledColour,
  clipLineTokens,
  rememberSettledColour,
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
