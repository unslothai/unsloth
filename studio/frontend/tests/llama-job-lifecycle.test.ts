// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  heldJobRows,
  heldUpdateBannerPref,
  updateBannerAllowed,
  updateBannerComponents,
  updateComponentKey,
  updateComponentsLabel,
  updateToastTag,
  llamaReleaseChanged,
  llamaUpdateAdoptsRunningJob,
  llamaUpdatePresentation,
  llamaUpdateToastMessage,
  ownedLlamaSwitchOutcome,
} from "../src/lib/llama-job-lifecycle.ts";
import {
  type NotificationFrequency,
  notificationDue,
} from "../src/hooks/use-notification-frequency.ts";

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
  assert.match(banner, /versionChanged: llamaReleaseChanged\(/);
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

// The job's tag is llama.cpp's, so it must not be pinned on every component a bundle names.
test("a bundle with nothing from the job names no single release", () => {
  assert.equal(
    llamaUpdateToastMessage({
      component: "llama.cpp, whisper.cpp and audio.cpp",
      migrating: true,
      jobMessage: "",
      updatedTag: null,
      reloadRequired: true,
    }),
    "llama.cpp, whisper.cpp and audio.cpp updated. Reload your model to use it.",
  );
});

// The card is muted per component, and a chained apply drops llama.cpp from the
// status when its phase lands and the whisper.cpp phase starts. Reading the live
// switch there hid a running update for the whole second phase.
test("the switch a running card started under is held until the job is over", () => {
  // llama.cpp notifications on, whisper.cpp off, chained update accepted.
  let held = heldUpdateBannerPref(null, true, true);
  assert.equal(held, true);
  // The llama phase lands and only the muted whisper.cpp is left pending.
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

// Update installs every pending component in one job, so the card lists them all.
test("the card lists every component Update will install", () => {
  const none = { llama: false, whisper: false, audio: false };
  assert.deepEqual(updateBannerComponents({ llama: true, whisper: true, audio: true }), [
    "llama.cpp",
    "whisper.cpp",
    "audio.cpp",
  ]);
  // One pending: the same single-component card as before.
  assert.deepEqual(
    updateBannerComponents({ llama: false, whisper: true, audio: false }),
    ["whisper.cpp"],
  );
  assert.deepEqual(updateBannerComponents(none), []);
});

test("the card names its components in install order", () => {
  assert.equal(updateComponentsLabel(["whisper.cpp"]), "whisper.cpp");
  assert.equal(
    updateComponentsLabel(["llama.cpp", "audio.cpp"]),
    "llama.cpp and audio.cpp",
  );
  assert.equal(
    updateComponentsLabel(["llama.cpp", "whisper.cpp", "audio.cpp"]),
    "llama.cpp, whisper.cpp and audio.cpp",
  );
});

test("one due component opens the card for all of them", () => {
  const listed = ["llama.cpp", "whisper.cpp"] as const;
  const noneMuted = { llama: false, whisper: false, audio: false };
  assert.equal(
    updateBannerAllowed(listed, { llama: false, whisper: true, audio: false }, noneMuted, false),
    true,
  );
  assert.equal(
    updateBannerAllowed(listed, { llama: false, whisper: false, audio: true }, noneMuted, false),
    false,
    "audio.cpp is due but has nothing listed",
  );
  // Open and recorded: nothing is due any more, and it stays up.
  assert.equal(
    updateBannerAllowed(listed, { llama: false, whisper: false, audio: false }, noneMuted, true),
    true,
  );
});

// Off means the component never asks for the card, not that the card hides it.
test("a muted component is listed but never opens or holds the card", () => {
  const listed = ["llama.cpp", "whisper.cpp"] as const;
  const whisperOff = { llama: false, whisper: true, audio: false };
  // whisper.cpp alone cannot open it, even when its clock would say due.
  assert.equal(
    updateBannerAllowed(listed, { llama: false, whisper: true, audio: false }, whisperOff, false),
    false,
  );
  // llama.cpp opens it, and whisper.cpp rides along on the same card.
  assert.equal(
    updateBannerAllowed(listed, { llama: true, whisper: false, audio: false }, whisperOff, false),
    true,
  );
  // Every listed component turned off while the card is open: it goes.
  assert.equal(
    updateBannerAllowed(
      listed,
      { llama: false, whisper: false, audio: false },
      { llama: true, whisper: true, audio: false },
      true,
    ),
    false,
  );
});

// llama.cpp and whisper.cpp land, audio.cpp fails: the retry is for audio.cpp alone.
test("a failed phase hands the card back to what is still pending", () => {
  const all = ["llama.cpp", "whisper.cpp", "audio.cpp"];
  let held = heldJobRows<string>(null, true, all);
  assert.deepEqual(held, all);
  // Phases land and leave the status mid-job; the card keeps naming the whole job.
  held = heldJobRows(held, true, ["whisper.cpp", "audio.cpp"]);
  held = heldJobRows(held, true, ["audio.cpp"]);
  assert.deepEqual(held, all);
  // audio.cpp fails: nothing is held, so the card shows the live rows.
  held = heldJobRows(held, false, ["audio.cpp"]);
  assert.equal(held, null);
  // The retry starts from what is left.
  held = heldJobRows(held, true, ["audio.cpp"]);
  assert.deepEqual(held, ["audio.cpp"]);
  // A job started elsewhere with nothing listed holds nothing.
  assert.equal(heldJobRows<string>(null, true, []), null);
});

// The reported annoyance: the card named one component per appearance and recorded
// only that one, so one pending job came back once per component, each on its own clock.
function appearances(
  listed: ReturnType<typeof updateBannerComponents>,
  frequency: Record<"llama" | "whisper" | "audio", NotificationFrequency>,
  launches: number[],
): number[] {
  const lastShown: Record<string, number | null> = {
    llama: null,
    whisper: null,
    audio: null,
  };
  const shownAt: number[] = [];
  for (const now of launches) {
    const due = {
      llama: notificationDue(frequency.llama, lastShown.llama, now),
      whisper: notificationDue(frequency.whisper, lastShown.whisper, now),
      audio: notificationDue(frequency.audio, lastShown.audio, now),
    };
    const noneMuted = { llama: false, whisper: false, audio: false };
    if (!updateBannerAllowed(listed, due, noneMuted, false)) continue;
    shownAt.push(now);
    for (const component of listed) {
      lastShown[updateComponentKey(component)] = now;
    }
  }
  return shownAt;
}

test("one pending job interrupts once per quiet period, not once per component", () => {
  const HOUR = 60 * 60 * 1000;
  const start = 1_000_000_000_000;
  const sameDay = [start, start + HOUR, start + 2 * HOUR, start + 3 * HOUR];
  // Defaults, llama.cpp current, whisper.cpp and audio.cpp both behind.
  assert.deepEqual(
    appearances(
      ["whisper.cpp", "audio.cpp"],
      { llama: "always", whisper: "biweekly", audio: "biweekly" },
      sameDay,
    ),
    [start],
  );
  // llama.cpp set to daily with all three behind.
  assert.deepEqual(
    appearances(
      ["llama.cpp", "whisper.cpp", "audio.cpp"],
      { llama: "daily", whisper: "biweekly", audio: "biweekly" },
      sameDay,
    ),
    [start],
  );
  // The shortest quiet period still governs when the card returns.
  assert.deepEqual(
    appearances(
      ["llama.cpp", "whisper.cpp"],
      { llama: "daily", whisper: "biweekly", audio: "biweekly" },
      [start, start + 25 * HOUR],
    ),
    [start, start + 25 * HOUR],
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
  assert.equal(
    updateToastTag("audio.cpp", "b11100", "v0.9.0-unsloth.1"),
    "v0.9.0-unsloth.1",
  );
  // Either side falls back to the other rather than reporting nothing.
  assert.equal(updateToastTag("whisper.cpp", "b11100", null), "b11100");
  assert.equal(updateToastTag("llama.cpp", null, "b11100"), "b11100");
  assert.equal(updateToastTag("llama.cpp", null, null), null);
});

// The chat "Update llama.cpp" button rides the shared update item, whose
// update_available is the union: a whisper.cpp or audio.cpp offer alone must not
// show it for a llama.cpp build that is current.
test("chat settings offer the llama.cpp update only for a llama.cpp release", () => {
  const sheet = readSrc("features/chat/chat-settings-sheet.tsx");
  assert.match(sheet, /mtpUpdatable && llamaUpdateStatus\?\.llama\.update_available/);
  assert.doesNotMatch(sheet, /llamaUpdateStatus\?\.update_available/);
});
