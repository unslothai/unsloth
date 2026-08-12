// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  llamaUpdatePresentation,
  ownedLlamaSwitchOutcome,
} from "../src/lib/llama-job-lifecycle.ts";

const SWITCH_STARTED_AT = "2026-08-12T15:00:00Z";

test("an owned switch recognizes only its explicit running and terminal states", () => {
  for (const state of ["running", "success", "error"] as const) {
    assert.equal(
      ownedLlamaSwitchOutcome(
        { state, operation: "switch", startedAt: SWITCH_STARTED_AT },
        SWITCH_STARTED_AT,
      ),
      state,
    );
  }
});

test("a lost or replaced switch is interrupted rather than successful", () => {
  for (const job of [
    { state: "idle" as const, operation: null, startedAt: null },
    {
      state: "success" as const,
      operation: "update" as const,
      startedAt: SWITCH_STARTED_AT,
    },
    {
      state: "running" as const,
      operation: "switch" as const,
      startedAt: "a-different-job",
    },
    {
      state: "success" as const,
      operation: "switch" as const,
      startedAt: null,
    },
  ]) {
    assert.equal(
      ownedLlamaSwitchOutcome(job, SWITCH_STARTED_AT),
      "interrupted",
    );
  }
});

test("a running switch hides the update banner without showing update progress", () => {
  assert.deepEqual(
    llamaUpdatePresentation(true, {
      state: "running",
      operation: "switch",
    }),
    { applying: false, visible: false, running: true },
  );
});

test("every terminal switch status restores a pending update", () => {
  for (const state of ["success", "error", "idle"] as const) {
    assert.deepEqual(
      llamaUpdatePresentation(true, { state, operation: "switch" }),
      { applying: false, visible: true, running: false },
    );
  }
});

test("a completed update stays hidden when no update remains", () => {
  assert.deepEqual(
    llamaUpdatePresentation(false, {
      state: "success",
      operation: "update",
    }),
    { applying: false, visible: false, running: false },
  );
});
