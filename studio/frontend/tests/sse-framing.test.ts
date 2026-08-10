// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { takeSseFrame } from "../src/features/training/api/sse-framing.ts";

test("preserves the next SSE frame across every supported delimiter", () => {
  for (const separator of ["\n\n", "\r\n\r\n", "\n\r\n", "\r\n\n"]) {
    assert.deepEqual(takeSseFrame(`event: progress${separator}event: next`), {
      event: "event: progress",
      remainder: "event: next",
    });
  }
});

test("waits for a complete SSE frame delimiter", () => {
  assert.equal(takeSseFrame("event: progress\n"), null);
  assert.equal(takeSseFrame("event: progress\n\r"), null);
});
