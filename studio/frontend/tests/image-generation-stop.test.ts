// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  GENERATION_CANCELLED_SENTINEL,
  shouldContinueGenerating,
  shouldReportGenerateError,
} from "../src/features/images/lib/generation-stop.ts";

test("a mounted page with no stop keeps generating", () => {
  assert.equal(
    shouldContinueGenerating({ mounted: true, stopRequested: false }),
    true,
  );
});

test("Stop breaks the run loop, so a count > 1 request does not start its next run", () => {
  // The backend cancel only reaches the denoise in flight; without this the page would
  // immediately POST run 2 of 4 and the machine would keep generating after Stop.
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
  // A proxy can rewrite the 409 body, and the run can unwind some other way; the latch is what
  // makes the user's own Stop silent rather than a spurious red toast.
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
