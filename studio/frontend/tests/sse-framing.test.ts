// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { takeSseFrame } from "../src/lib/sse-framing.ts";

test("preserves the next SSE frame across every supported delimiter", () => {
  for (const separator of [
    "\n\n", "\r\n\r\n", "\n\r\n", "\r\n\n",
    "\r\r", "\r\r\n", "\r\n\r", "\n\r",
  ]) {
    assert.deepEqual(takeSseFrame(`event: progress${separator}event: next`), {
      event: "event: progress",
      remainder: "event: next",
    });
  }
});

test("waits for a complete SSE frame delimiter", () => {
  assert.equal(takeSseFrame("event: progress\n"), null);
  assert.equal(takeSseFrame("event: progress\r"), null);
  assert.equal(takeSseFrame("event: progress\r\n"), null);
});

test("normalizes bare CR lines without changing CRLF lines", () => {
  assert.deepEqual(
    takeSseFrame('event: progress\rid: 1\r\ndata: {"n": 1}\r\rnext'),
    {
      event: 'event: progress\nid: 1\r\ndata: {"n": 1}',
      remainder: "next",
    },
  );
});
