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
  // Written before the field existed, with a layout we shipped: no choice was ever made.
  assert.deepEqual(sanitizeCustomization({}).sidebarNavAuto, ["projects"]);
  assert.deepEqual(DEFAULT_CUSTOMIZATION.sidebarNavAuto, ["projects"]);
  assert.deepEqual(
    sanitizeCustomization({ sidebarNav: [...DEFAULT_CUSTOMIZATION.sidebarNav] })
      .sidebarNavAuto,
    ["projects"],
  );
  // An arranged layout already carries a Projects choice, so the rule must not overrule it.
  const unpinned = sanitizeCustomization({
    sidebarNav: DEFAULT_CUSTOMIZATION.sidebarNav.map((entry) =>
      entry.id === "projects" ? { ...entry, pinned: false } : entry,
    ),
  });
  assert.deepEqual(unpinned.sidebarNavAuto, []);
  assert.equal(
    sidebarNavRowPinned(
      unpinned.sidebarNav.find((entry) => entry.id === "projects")!,
      unpinned.sidebarNavAuto,
      { projectsSectionShowing: false },
    ),
    false,
    "an upgrade pinned a row the user had put away",
  );
  // Reordering counts as arranging too, even with Projects left where it was.
  const reordered = sanitizeCustomization({
    sidebarNav: [
      { id: "train", pinned: true },
      ...DEFAULT_CUSTOMIZATION.sidebarNav.filter((entry) => entry.id !== "train"),
    ],
  });
  assert.deepEqual(reordered.sidebarNavAuto, []);
  // An explicit empty list is a decision, and survives the round trip.
  assert.deepEqual(sanitizeCustomization({ sidebarNavAuto: [] }).sidebarNavAuto, []);
});

// The rail hides both folder sections in CSS without unmounting them, so the row has to stay on
// it: standing down there would bury the only way to reach projects behind More.
test("the rail keeps the Projects row, since the section is hidden there", async () => {
  const sidebar = await readSrcAsync("components/app-sidebar.tsx");
  assert.match(
    sidebar,
    /const projectsSectionShowing =\n\s*projectsSectionConfigured && \(isMobile \|\| sidebarState !== "collapsed"\);/,
  );
  // The section itself still mounts on the rail, as it did before, and CSS hides it.
  assert.match(sidebar, /\{projectsSectionRendered && \(/);
  assert.match(
    sidebar,
    /\{projectsSectionRendered && \(\n[\s\S]{0,400}?group-data-\[collapsible=icon\]:hidden/,
  );
});

// Pinned folders are rows of Pinned, whose ids are not the Projects list's, so a Shift-click
// that read the Projects list found neither endpoint and cleared the selection.
test("folders range-select within the list the row is in", async () => {
  const sidebar = await readSrcAsync("components/app-sidebar.tsx");
  assert.match(
    sidebar,
    /function handleProjectSelectionClick\(\n\s*event: React\.MouseEvent,\n\s*projectId: string,\n(?:\s*\/\/[^\n]*\n)*\s*orderedIds: string\[\],/,
  );
  assert.match(
    sidebar,
    /const sameList = anchorId !== null && orderedIds\.includes\(anchorId\);/,
  );
  assert.match(sidebar, /rangeBetween\(orderedIds, anchorId, projectId\)/);
  assert.match(
    sidebar,
    /handleProjectSelectionClick\(\n\s*event,\n\s*project\.id,\n\s*order\.selectionIds \?\? order\.orderedIds,\n\s*\)/,
  );
  // Nothing reaches for the Projects list from inside the handler any more.
  const handler = sidebar.slice(
    sidebar.indexOf("function handleProjectSelectionClick("),
    sidebar.indexOf("function selectProjectForContextMenu("),
  );
  assert.ok(!handler.includes("projectRowIds"), "the handler still reads projectRowIds");
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

// Train and Recipes list their runs where the folders go. The row read that as the section being
// gone and pinned itself back to the top, so walking to Train turned a row the user had put away
// back on, and walking back turned it off again.
test("a route that borrows the section does not bring the row back", async () => {
  const sidebar = await readSrcAsync("components/app-sidebar.tsx");
  assert.match(
    sidebar,
    /const projectsSectionConfigured =\n\s*organizeBy === "project" && projects\.length > 0;/,
  );
  // The section itself still stands down on those routes; only the row stopped following it.
  assert.match(
    sidebar,
    /const projectsSectionRendered =\n\s*!isStudioRoute && !showTrainingRecents && projectsSectionConfigured;/,
  );
  const showing = sidebar.slice(
    sidebar.indexOf("const projectsSectionShowing ="),
    sidebar.indexOf("const chatSort ="),
  );
  for (const route of ["isStudioRoute", "showTrainingRecents"]) {
    assert.ok(
      !showing.includes(route),
      `the row still reads ${route}`,
    );
  }
});

// Both the sidebar and Customize sidebar decide the row's place, and a switch that disagrees with
// the sidebar beside it is the bug in another form.
test("the customizer reads the setting the same way", async () => {
  const customizer = await readSrcAsync(
    "features/settings/components/sidebar-nav-customizer.tsx",
  );
  assert.match(
    customizer,
    /const projectsSectionShowing =\n\s*organizeBy === "project" && projects\.length > 0;/,
  );
});
