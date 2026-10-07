// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  heldUpdateBannerPref,
  updateBannerComponent,
  updateToastTag,
  llamaReleaseChanged,
  llamaUpdateAdoptsRunningJob,
  llamaUpdatePresentation,
  llamaUpdateToastMessage,
  ownedLlamaSwitchOutcome,
} from "../src/lib/llama-job-lifecycle.ts";

import { readSrc } from "./helpers/kit.ts";

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
  // Both share one job, so following a switch would mark an uninstalled release applied.
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
  const banner = readSrc("components/llama-update-banner.tsx");
  assert.match(banner, /const versionChanged = llamaReleaseChanged\(/);
  assert.doesNotMatch(banner, /installedTag !== latestTag/);
});

test("a backend migration is reported by what the job did, not by the version fields", () => {
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

// A chained apply renames the card mid-job, so the starting switch must be held.
test("the switch a running card started under is held until the job is over", () => {
  let held = heldUpdateBannerPref(null, true, true);
  assert.equal(held, true);
  held = heldUpdateBannerPref(held, true, false);
  assert.equal(held, true, "the running update was taken off screen");
  held = heldUpdateBannerPref(held, true, false);
  assert.equal(held, true);
  assert.equal(heldUpdateBannerPref(held, false, false), null);
});

test("a muted card stays muted for a job it never showed", () => {
  assert.equal(heldUpdateBannerPref(null, true, false), false);
  assert.equal(heldUpdateBannerPref(null, false, true), null);
});

// The backend names only one pending component.
test("the card shows the offer the switches allow", () => {
  const on = { llama: true, whisper: true };
  const bothStale = { llama: true, whisper: true };
  assert.equal(updateBannerComponent("llama.cpp", bothStale, on), "llama.cpp");
  assert.equal(updateBannerComponent("whisper.cpp", bothStale, on), "whisper.cpp");
  assert.equal(
    updateBannerComponent("llama.cpp", bothStale, { llama: false, whisper: true }),
    "whisper.cpp",
  );
  assert.equal(
    updateBannerComponent("whisper.cpp", bothStale, { llama: true, whisper: false }),
    "llama.cpp",
  );
  assert.equal(
    updateBannerComponent(
      "llama.cpp",
      { llama: true, whisper: false },
      { llama: false, whisper: true },
    ),
    "llama.cpp",
  );
  assert.equal(
    updateBannerComponent("llama.cpp", bothStale, { llama: false, whisper: false }),
    "llama.cpp",
  );
});

test("a finished update reports the release its card advertised", () => {
  assert.equal(
    updateToastTag("whisper.cpp", "b11100", "v1.9.4-unsloth.4"),
    "v1.9.4-unsloth.4",
  );
  assert.equal(updateToastTag("llama.cpp", "b11100", "b11100"), "b11100");
  assert.equal(updateToastTag("whisper.cpp", "b11100", null), "b11100");
  assert.equal(updateToastTag("llama.cpp", null, "b11100"), "b11100");
  assert.equal(updateToastTag("llama.cpp", null, null), null);
});
