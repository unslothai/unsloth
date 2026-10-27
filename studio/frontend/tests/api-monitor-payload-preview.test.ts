// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PAYLOAD_PREVIEW_CHARS,
  payloadPreview,
} from "../src/features/api-monitor/payload-preview.ts";

const LONE_HIGH = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])/;
const LONE_LOW = /(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

test("short payloads are shown whole", () => {
  for (const body of ["", "hi", "x".repeat(PAYLOAD_PREVIEW_CHARS)]) {
    assert.equal(payloadPreview(body), body);
  }
});

test("an agent prompt keeps the system head and the latest user turn", () => {
  const system = `system: ${"You are a coding agent. ".repeat(2_000)}`;
  const body = `${system}\n\nuser: fix the failing test in parser.py`;
  const preview = payloadPreview(body);

  assert.ok(preview.startsWith("system: You are a coding agent."));
  assert.ok(preview.endsWith("user: fix the failing test in parser.py"));
  const omitted = body.length - PAYLOAD_PREVIEW_CHARS;
  assert.ok(
    preview.includes(
      `[... ${omitted.toLocaleString("en-US")} characters omitted ...]`,
    ),
  );
  assert.ok(preview.length < PAYLOAD_PREVIEW_CHARS + 64);
});

test("neither cut splits a surrogate pair", () => {
  // "a" lands the head cut inside a pair, "b" the tail cut.
  for (const [prefix, suffix] of [
    ["", ""],
    ["a", ""],
    ["", "b"],
  ]) {
    const body = prefix + "\u{1F9A5}".repeat(20_000) + suffix;
    const preview = payloadPreview(body);
    const label = JSON.stringify([prefix, suffix]);
    assert.ok(!LONE_HIGH.test(preview), label);
    assert.ok(!LONE_LOW.test(preview), label);
  }
});
