// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The stall rule behind the stream-pacing smoke's longestStallMs, which is the budget that
// catches the freeze class #7892 and #8845 fixed. A stall the harness fails to record is a
// harness that passes on a frozen renderer, so the interesting cases here are the ones where
// no later paint ever arrives to close the stall.

import assert from "node:assert/strict";
import test from "node:test";

import { stallInProgress } from "../smoke-stream-pacing-stall.ts";

const STARTED_AT = 1_000;

test("while text is still arriving, the stall runs to now", () => {
  assert.equal(stallInProgress(2_000, 3_500, STARTED_AT, null), 1_500);
});

test("a freeze spanning the end of the stream is still recorded in full", () => {
  // The regression this exists for. The frame loop is blocked across the moment the stream
  // ends, so the first frame afterwards (now = 9_000) already sees a non-null streamEndedAtMs.
  // Measuring only while the stream is live would skip the whole interval and report nothing,
  // and the missing tail can hide inside the harness's 90% workload floor.
  const streamEndedAtMs = 4_000; // absolute 5_000
  assert.equal(stallInProgress(1_500, 9_000, STARTED_AT, streamEndedAtMs), 3_500);
});

test("after the stream ends the stall stops growing with the settle window", () => {
  // The settle check needs 30 quiet frames by design. Measuring to now would count them, so
  // every healthy run would report a stall the length of its own settle window.
  const streamEndedAtMs = 4_000; // absolute 5_000
  const atEnd = stallInProgress(4_800, 5_000, STARTED_AT, streamEndedAtMs);
  const muchLater = stallInProgress(4_800, 60_000, STARTED_AT, streamEndedAtMs);
  assert.equal(atEnd, 200);
  assert.equal(muchLater, 200, "quiet frames after the stream ended are not a stall");
});

test("it is idempotent once the stream has ended", () => {
  const first = stallInProgress(2_000, 6_000, STARTED_AT, 4_000);
  const second = stallInProgress(2_000, 30_000, STARTED_AT, 4_000);
  assert.equal(first, second);
});

test("growth after the stream ended is not a negative stall", () => {
  // lastGrowthAt can sit past the stream end when the tail paints late, which would make the
  // capped interval negative. Report no stall rather than a negative one.
  assert.equal(stallInProgress(8_000, 9_000, STARTED_AT, 4_000), 0);
});
