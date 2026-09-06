// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  llamaReleaseChanged,
  llamaUpdateAdoptsRunningJob,
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

test("an apply adopts an already-running update but never a switch", () => {
  // Both share one job. Following a switch here would resolve the update action
  // as applied while the release it offered is still not installed.
  assert.equal(
    llamaUpdateAdoptsRunningJob("already_running", {
      state: "running",
      operation: "update",
    }),
    true,
  );
  assert.equal(
    llamaUpdateAdoptsRunningJob("already_running", {
      state: "running",
      operation: "switch",
    }),
    false,
  );
  assert.equal(
    llamaUpdateAdoptsRunningJob("up_to_date", {
      state: "success",
      operation: "update",
    }),
    false,
  );
});

test("a backend migration at the installed release reports no version change", () => {
  // What the server actually sends for a fork install whose release is current:
  // the display tag is normalized to its base and the latest tag is the full
  // release identity, so the two differ while naming the same release.
  assert.equal(
    llamaReleaseChanged(false, "b9596", "b9596-mix-4b653db"),
    false,
  );
  // And the same shape once the release really has moved.
  assert.equal(
    llamaReleaseChanged(true, "b9596", "b10715-mix-86bd2d3"),
    true,
  );
});

test("a release change still needs both tags to name it", () => {
  assert.equal(llamaReleaseChanged(true, null, "b10715-mix-86bd2d3"), false);
  assert.equal(llamaReleaseChanged(true, "b9596", null), false);
  assert.equal(llamaReleaseChanged(true, "b9596", "b9596"), false);
});

test("the banner asks the helper rather than comparing the two tags itself", () => {
  // Read from the source: the node suite has no DOM to render the banner in, and
  // the predicate is only worth fixing in one place if the banner uses that place.
  const banner = readFileSync(
    new URL("../src/components/llama-update-banner.tsx", import.meta.url),
    "utf8",
  );
  assert.match(banner, /const versionChanged = llamaReleaseChanged\(/);
  assert.doesNotMatch(banner, /installedTag !== latestTag/);
});
