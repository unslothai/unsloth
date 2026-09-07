// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  llamaReleaseChanged,
  llamaUpdateAdoptsRunningJob,
  llamaUpdatePresentation,
  llamaUpdateToastMessage,
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
  // What a fork install at the current release sends: the display tag is normalized and
  // the latest tag is the full identity, so they differ while naming one release.
  assert.equal(
    llamaReleaseChanged(false, "b9596", "b9596-mix-4b653db"),
    false,
  );
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
  const banner = readFileSync(
    new URL("../src/components/llama-update-banner.tsx", import.meta.url),
    "utf8",
  );
  assert.match(banner, /const versionChanged = llamaReleaseChanged\(/);
  assert.doesNotMatch(banner, /installedTag !== latestTag/);
});

test("a backend migration is reported by what the job did, not by the version fields", () => {
  // The migration runs at the release already installed, so composing the toast from the
  // tags announces an update that did not happen.
  assert.equal(
    llamaUpdateToastMessage({
      component: "whisper.cpp",
      migrating: true,
      jobMessage: "llama.cpp is now running on vulkan.",
      updatedTag: "b9596-mix-abc",
      reloadRequired: false,
    }),
    "llama.cpp is now running on vulkan.",
  );

  assert.equal(
    llamaUpdateToastMessage({
      component: "llama.cpp",
      migrating: true,
      jobMessage:
        "llama.cpp could not be moved to vulkan right now, so the existing rocm build was kept. Try again later.",
      updatedTag: "b9596-mix-abc",
      reloadRequired: false,
    }),
    "llama.cpp could not be moved to vulkan right now, so the existing rocm build was kept. Try again later.",
  );

  assert.equal(
    llamaUpdateToastMessage({
      component: "llama.cpp",
      migrating: true,
      jobMessage: "llama.cpp is now running on vulkan.",
      updatedTag: "b9596-mix-abc",
      reloadRequired: true,
    }),
    "llama.cpp is now running on vulkan. Reload your model to use it.",
  );
});

test("an ordinary update still reports the release it moved to", () => {
  // Control: always deferring to the job would drop the tag from every real update, and
  // a migration with nothing to say would print blank.
  assert.equal(
    llamaUpdateToastMessage({
      component: "llama.cpp",
      migrating: false,
      jobMessage: "llama.cpp is now running on vulkan.",
      updatedTag: "b9600-mix-def",
      reloadRequired: true,
    }),
    "llama.cpp updated to b9600-mix-def. Reload your model to use it.",
  );
  assert.equal(
    llamaUpdateToastMessage({
      component: "llama.cpp",
      migrating: true,
      jobMessage: "  ",
      updatedTag: "b9600-mix-def",
      reloadRequired: false,
    }),
    "llama.cpp updated to b9600-mix-def.",
  );
});
