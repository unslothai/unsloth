// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Projects nav row repeats the Projects section, so it steps aside while that section is on
// screen, until the user decides its place in Customize sidebar.

import assert from "node:assert/strict";
import test from "node:test";
import { installLocalStorageFake, readSrcAsync } from "./helpers/kit.ts";

installLocalStorageFake();

const {
  DEFAULT_CUSTOMIZATION,
  sanitizeCustomization,
  sidebarNavAutoAfterChoice,
  sidebarNavRowPinned,
} = await import("../src/features/settings/stores/appearance-custom-store.ts");

const PROJECTS = { id: "projects", pinned: true } as const;

test("the row stands down only while the section is showing", () => {
  const auto = ["projects"] as const;
  assert.equal(
    sidebarNavRowPinned(PROJECTS, auto, { projectsSectionShowing: true }),
    false,
    "the row is still pinned beside the section that repeats it",
  );
  // No projects left, or the sidebar organised as one list, and the section is gone with them.
  assert.equal(
    sidebarNavRowPinned(PROJECTS, auto, { projectsSectionShowing: false }),
    true,
  );
});

test("a choice in Customize sidebar outlives the rule", () => {
  const decided = sidebarNavAutoAfterChoice(["projects"], "projects");
  assert.deepEqual(decided, []);
  // Pinned stays pinned beside the section…
  assert.equal(
    sidebarNavRowPinned(PROJECTS, decided, { projectsSectionShowing: true }),
    true,
  );
  // …and off stays off once every project is deleted.
  assert.equal(
    sidebarNavRowPinned({ id: "projects", pinned: false }, decided, {
      projectsSectionShowing: false,
    }),
    false,
  );
});

test("only Projects has a rule; every other row reads its own pref", () => {
  for (const id of ["hub", "images", "train", "api"] as const) {
    for (const showing of [true, false]) {
      assert.equal(
        sidebarNavRowPinned({ id, pinned: true }, ["projects"], {
          projectsSectionShowing: showing,
        }),
        true,
      );
    }
  }
  // And a stored list cannot smuggle another id into the rule.
  const sanitized = sanitizeCustomization({
    sidebarNavAuto: ["projects", "train", "nonsense"],
  });
  assert.deepEqual(sanitized.sidebarNavAuto, ["projects"]);
});

test("an install that never chose gets the rule, one that chose keeps its choice", () => {
  // Written before the field existed: no choice was ever made, so the rule applies.
  assert.deepEqual(sanitizeCustomization({}).sidebarNavAuto, ["projects"]);
  assert.deepEqual(DEFAULT_CUSTOMIZATION.sidebarNavAuto, ["projects"]);
  // An explicit empty list is a decision, and survives the round trip.
  assert.deepEqual(sanitizeCustomization({ sidebarNavAuto: [] }).sidebarNavAuto, []);
});

test("the sidebar and the customizer resolve the row the same way", async () => {
  const sidebar = await readSrcAsync("components/app-sidebar.tsx");
  const customizer = await readSrcAsync(
    "features/settings/components/sidebar-nav-customizer.tsx",
  );
  // Both split their rows with the shared resolver rather than reading `pinned` directly.
  assert.match(
    sidebar,
    /sidebarNavRowPinned\(item, sidebarNavAuto, \{ projectsSectionShowing \}\)/,
  );
  assert.match(sidebar, /\.filter\(\(item\) => !navRowPinned\(item\)\)/);
  assert.match(sidebar, /\.filter\(\(item\) => navRowPinned\(item\)\)/);
  assert.match(customizer, /checked=\{pinned\}/);
  // And the switch records the decision alongside the new placement.
  assert.match(
    customizer,
    /sidebarNavAuto: sidebarNavAutoAfterChoice\(sidebarNavAuto, item\.id\)/,
  );
});
