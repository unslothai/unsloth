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
  llamaUpdateSnapshotIsStale,
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

test("a late running poll for a finished job is stale", () => {
  const startedAt = "2026-08-18T13:02:21Z";
  assert.equal(
    llamaUpdateSnapshotIsStale(
      { state: "success", operation: "update", started_at: startedAt },
      { state: "running", operation: "update", started_at: startedAt },
    ),
    true,
  );
  assert.equal(
    llamaUpdateSnapshotIsStale(
      { state: "error", operation: "update", started_at: startedAt },
      { state: "running", operation: "update", started_at: startedAt },
    ),
    true,
  );
});

test("a running poll is kept when it is a different job or the current one is still running", () => {
  const startedAt = "2026-08-18T13:02:21Z";
  assert.equal(
    llamaUpdateSnapshotIsStale(
      { state: "success", operation: "update", started_at: startedAt },
      {
        state: "running",
        operation: "update",
        started_at: "2026-08-18T14:00:00Z",
      },
    ),
    false,
  );
  assert.equal(
    llamaUpdateSnapshotIsStale(
      { state: "running", operation: "update", started_at: startedAt },
      { state: "running", operation: "update", started_at: startedAt },
    ),
    false,
  );
  assert.equal(
    llamaUpdateSnapshotIsStale(
      { state: "success", operation: "update", started_at: startedAt },
      { state: "success", operation: "update", started_at: startedAt },
    ),
    false,
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
  const banner = readSrc("components/llama-update-banner.tsx");
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

// The card is muted per component, and a chained apply renames it when the
// llama.cpp phase lands and the whisper.cpp phase starts. Reading the live
// switch there hid a running update for the whole second phase.
test("the switch a running card started under is held until the job is over", () => {
  // llama.cpp notifications on, whisper.cpp off, chained update accepted.
  let held = heldUpdateBannerPref(null, true, true);
  assert.equal(held, true);
  // The llama phase lands and the status renames the card mid-job.
  held = heldUpdateBannerPref(held, true, false);
  assert.equal(held, true, "the running update was taken off screen");
  // The whisper phase fails: its retry has to stay reachable.
  held = heldUpdateBannerPref(held, true, false);
  assert.equal(held, true);
  // Job over: the live switch answers again.
  assert.equal(heldUpdateBannerPref(held, false, false), null);
});

test("a muted card stays muted for a job it never showed", () => {
  // Nothing held, whisper.cpp muted, a whisper job running from another surface.
  assert.equal(heldUpdateBannerPref(null, true, false), false);
  // And an offer with no job in flight always reads the live switch.
  assert.equal(heldUpdateBannerPref(null, false, true), null);
});

// Both components can be pending at once and the backend names only one, so the
// switch of the one it did not name had nothing to answer for.
test("the card shows the offer the switches allow", () => {
  const on = { llama: true, whisper: true };
  const bothStale = { llama: true, whisper: true };
  assert.equal(updateBannerComponent("llama.cpp", bothStale, on), "llama.cpp");
  assert.equal(updateBannerComponent("whisper.cpp", bothStale, on), "whisper.cpp");
  // llama.cpp muted, whisper.cpp on, both pending: show the one asked for.
  assert.equal(
    updateBannerComponent("llama.cpp", bothStale, { llama: false, whisper: true }),
    "whisper.cpp",
  );
  // And the reverse: a llama.cpp backend migration is named whisper.cpp when
  // whisper is stale as well, so it needs the same swap back.
  assert.equal(
    updateBannerComponent("whisper.cpp", bothStale, { llama: true, whisper: false }),
    "llama.cpp",
  );
  // Nothing to swap to when the other component has no offer.
  assert.equal(
    updateBannerComponent(
      "llama.cpp",
      { llama: true, whisper: false },
      { llama: false, whisper: true },
    ),
    "llama.cpp",
  );
  // Both muted: the name the backend gave stands, and the card stays hidden.
  assert.equal(
    updateBannerComponent("llama.cpp", bothStale, { llama: false, whisper: false }),
    "llama.cpp",
  );
});

// The job reports the llama.cpp build it installed, which is not what a card
// showing the whisper.cpp offer told the user it was getting.
test("a finished update reports the release its card advertised", () => {
  assert.equal(
    updateToastTag("whisper.cpp", "b11100", "v1.9.4-unsloth.4"),
    "v1.9.4-unsloth.4",
  );
  assert.equal(updateToastTag("llama.cpp", "b11100", "b11100"), "b11100");
  // Either side falls back to the other rather than reporting nothing.
  assert.equal(updateToastTag("whisper.cpp", "b11100", null), "b11100");
  assert.equal(updateToastTag("llama.cpp", null, "b11100"), "b11100");
  assert.equal(updateToastTag("llama.cpp", null, null), null);
});
