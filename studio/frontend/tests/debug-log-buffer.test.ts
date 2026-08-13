// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_REFRESH_MODE,
  EMPTY_BUFFER,
  MAX_CLIENT_LINES,
  applyLogChunk,
  parseRefreshMode,
  pollDelayMs,
  trimBuffer,
} from "../src/features/settings/lib/debug-log-buffer.ts";

test("three seconds is the default refresh mode", () => {
  assert.equal(DEFAULT_REFRESH_MODE, "3s");
  assert.equal(parseRefreshMode(null), "3s");
  assert.equal(parseRefreshMode("nonsense"), "3s");
  assert.equal(parseRefreshMode("live"), "live");
  assert.equal(parseRefreshMode("manual"), "manual");
});

test("each mode maps to its poll delay, manual to none", () => {
  assert.equal(pollDelayMs("live"), 1000);
  assert.equal(pollDelayMs("3s"), 3000);
  assert.equal(pollDelayMs("manual"), null);
});

test("a chunk appends to what is already there", () => {
  const first = applyLogChunk(EMPTY_BUFFER, {
    lines: ["a", "b"],
    cursor: "c1",
    reset: true,
  });
  const second = applyLogChunk(first, {
    lines: ["c"],
    cursor: "c2",
    reset: false,
  });
  assert.deepEqual(second.lines, ["a", "b", "c"]);
  assert.equal(second.cursor, "c2");
});

test("a reset replaces the buffer rather than appending to it", () => {
  const first = applyLogChunk(EMPTY_BUFFER, {
    lines: ["old"],
    cursor: "c1",
    reset: true,
  });
  const second = applyLogChunk(first, {
    lines: ["fresh"],
    cursor: "c2",
    reset: true,
  });
  assert.deepEqual(second.lines, ["fresh"]);
});

test("an empty chunk returns the same object so React can skip the render", () => {
  const first = applyLogChunk(EMPTY_BUFFER, {
    lines: ["a"],
    cursor: "c1",
    reset: true,
  });
  const second = applyLogChunk(first, {
    lines: [],
    cursor: "c1",
    reset: false,
  });
  assert.equal(second, first);
});

test("the buffer is capped and keeps the newest lines", () => {
  const lines = Array.from(
    { length: MAX_CLIENT_LINES + 500 },
    (_, i) => `line${i}`,
  );
  const trimmed = trimBuffer(lines);
  assert.equal(trimmed.length, MAX_CLIENT_LINES);
  assert.equal(trimmed[trimmed.length - 1], `line${MAX_CLIENT_LINES + 499}`);
});

test("a few enormous lines are capped by characters, not just by count", () => {
  const lines = Array.from({ length: 40 }, () => "x".repeat(20_000));
  const trimmed = trimBuffer(lines);
  const chars = trimmed.reduce((total, line) => total + line.length + 1, 0);
  assert.ok(
    chars <= 400_000,
    `expected the buffer under the char cap, got ${chars}`,
  );
  assert.ok(trimmed.length < lines.length);
});

test("appending past the cap still keeps the tail", () => {
  let state = applyLogChunk(EMPTY_BUFFER, {
    lines: ["first"],
    cursor: "c1",
    reset: true,
  });
  for (let i = 0; i < MAX_CLIENT_LINES + 10; i += 1) {
    state = applyLogChunk(state, {
      lines: [`n${i}`],
      cursor: `c${i}`,
      reset: false,
    });
  }
  assert.equal(state.lines.length, MAX_CLIENT_LINES);
  assert.equal(state.lines[state.lines.length - 1], `n${MAX_CLIENT_LINES + 9}`);
  assert.ok(!state.lines.includes("first"));
});
