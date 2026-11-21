// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  GENERATION_CANCELLED_SENTINEL,
  shouldContinueGenerating,
  shouldReportGenerateError,
  stopButtonLabel,
} from "../src/features/images/lib/generation-stop.ts";

test("a mounted page with no stop keeps generating", () => {
  assert.equal(
    shouldContinueGenerating({ mounted: true, stopRequested: false }),
    true,
  );
});

test("Stop breaks the run loop, so a count > 1 request does not start its next run", () => {
  // The backend cancel only reaches the denoise in flight; the page must not POST the next run.
  assert.equal(
    shouldContinueGenerating({ mounted: true, stopRequested: true }),
    false,
  );
});

test("an unmounted page stops generating regardless of the stop latch", () => {
  assert.equal(
    shouldContinueGenerating({ mounted: false, stopRequested: false }),
    false,
  );
  assert.equal(
    shouldContinueGenerating({ mounted: false, stopRequested: true }),
    false,
  );
});

test("the cancelled sentinel is not reported as an error", () => {
  assert.equal(
    shouldReportGenerateError({
      message: GENERATION_CANCELLED_SENTINEL,
      stopRequested: false,
    }),
    false,
  );
});

test("a stopped run reports nothing even when the message is not the sentinel", () => {
  assert.equal(
    shouldReportGenerateError({ message: "Bad Gateway", stopRequested: true }),
    false,
  );
});

test("a real failure is still reported", () => {
  assert.equal(
    shouldReportGenerateError({
      message:
        "The device ran out of memory. Try a smaller size, fewer steps, or a smaller batch.",
      stopRequested: false,
    }),
    true,
  );
});

test("a Stop the backend did not act on does not explain away a real failure", () => {
  // stopRequested is only passed once the backend answered {cancelled: true}.
  const stopRequested = true;
  for (const cancelAcked of [false]) {
    assert.equal(
      shouldReportGenerateError({
        message: "Failed to save the generated image",
        stopRequested: stopRequested && cancelAcked,
      }),
      true,
    );
  }
});

test("a Stop the backend confirmed still silences the run it stopped", () => {
  assert.equal(
    shouldReportGenerateError({ message: "Bad Gateway", stopRequested: true && true }),
    false,
  );
});

test("Stop shows Stopping… once clicked, so a cancel waiting on a long step does not look ignored", () => {
  assert.equal(stopButtonLabel({ stopping: false, done: null, count: 1 }), "Stop");
  assert.equal(stopButtonLabel({ stopping: false, done: 1, count: 4 }), "Stop (1/4)");
  assert.equal(stopButtonLabel({ stopping: true, done: 1, count: 4 }), "Stopping…");
  assert.equal(stopButtonLabel({ stopping: false, done: null, count: 1, idle: "Cancel" }), "Cancel");
  assert.equal(stopButtonLabel({ stopping: true, done: null, count: 1, idle: "Cancel" }), "Stopping…");
});
