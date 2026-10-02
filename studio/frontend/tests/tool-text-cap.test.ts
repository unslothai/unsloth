// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_TOOL_TEXT_CHARS,
  capToolText,
} from "../src/features/chat/api/tool-text-cap.ts";

const NOTICE =
  "\n\n... (tool result truncated to 256,000 chars for the model; the full output is not retained in model context.)";

test("results at or under the floor pass through uncut", () => {
  assert.equal(capToolText("short result"), "short result");
  const exact = "x".repeat(MAX_TOOL_TEXT_CHARS);
  assert.equal(capToolText(exact), exact);
});

test("an oversized result is cut at a nearby line break and the notice is appended", () => {
  const line = `${"x".repeat(100)}\n`;
  const big = line.repeat(MAX_TOOL_TEXT_CHARS / line.length + 100);

  const out = capToolText(big);
  assert.ok(out.endsWith(NOTICE));
  const body = out.slice(0, -NOTICE.length);
  assert.ok(big.startsWith(body));
  assert.ok(body.length < MAX_TOOL_TEXT_CHARS);
  assert.equal(big[body.length], "\n");
  assert.ok(out.length <= MAX_TOOL_TEXT_CHARS + NOTICE.length);
});

test("a single line is cut mid-line when no break is nearby", () => {
  const big = "y".repeat(MAX_TOOL_TEXT_CHARS + 500);
  const out = capToolText(big);
  assert.ok(out.endsWith(NOTICE));
  const body = out.slice(0, -NOTICE.length);
  assert.equal(body, big.slice(0, MAX_TOOL_TEXT_CHARS));
});

test("capping is idempotent", () => {
  const big = Array.from({ length: 250_000 }, (_, i) => `line-${i}`).join("\n");
  const once = capToolText(big);
  assert.equal(capToolText(once), once);
});
