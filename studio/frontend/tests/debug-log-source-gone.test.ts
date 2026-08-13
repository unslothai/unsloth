// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A source id stops being enumerated once the file is removed, or once a run of
// failed load attempts pushes it out of the per-family window. The backend
// answers that with a 404 (every other state is a 200 with a status) so the
// picker can rebuild itself. That only works if the status survives the client.

import assert from "node:assert/strict";
import test from "node:test";

import {
  DebugLogRequestError,
  isAbort,
  isLogSourceGone,
} from "../src/features/settings/lib/debug-log-error.ts";

test("a gone source is the one failure the picker can recover from", () => {
  const gone = new DebugLogRequestError("Unknown log source.", 404);
  assert.equal(isLogSourceGone(gone), true);
  assert.equal(gone.name, "DebugLogRequestError");
  assert.equal(gone.message, "Unknown log source.");
  assert.ok(gone instanceof Error);
});

test("every other failure is reported rather than retried", () => {
  for (const status of [400, 401, 403, 500, 502, 503]) {
    assert.equal(
      isLogSourceGone(new DebugLogRequestError("boom", status)),
      false,
    );
  }
  // A plain Error is what the sources call and anything unexpected throws.
  assert.equal(isLogSourceGone(new Error("Could not read the log.")), false);
  assert.equal(isLogSourceGone(undefined), false);
  assert.equal(isLogSourceGone(null), false);
  assert.equal(isLogSourceGone("404"), false);
});

test("an aborted poll is not a failure at all", () => {
  const abort = new Error("aborted");
  abort.name = "AbortError";
  assert.equal(isAbort(abort), true);
  assert.equal(isAbort(new Error("nope")), false);
  assert.equal(isAbort(undefined), false);
  // Unmounting the tab must not be reported as a log-source problem.
  assert.equal(isLogSourceGone(abort), false);
});
