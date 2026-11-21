// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
  assert.equal(
    sidebarNavRowPinned(PROJECTS, auto, { projectsSectionShowing: false }),
    true,
  );
});

test("a choice in Customize sidebar outlives the rule", () => {
  const decided = sidebarNavAutoAfterChoice(["projects"], "projects");
  assert.deepEqual(decided, []);
  assert.equal(
    sidebarNavRowPinned(PROJECTS, decided, { projectsSectionShowing: true }),
    true,
  );
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
  const sanitized = sanitizeCustomization({
    sidebarNavAuto: ["projects", "train", "nonsense"],
  });
  assert.deepEqual(sanitized.sidebarNavAuto, ["projects"]);
});

test("an install that never chose gets the rule, one that chose keeps its choice", () => {
  assert.deepEqual(sanitizeCustomization({}).sidebarNavAuto, ["projects"]);
  assert.deepEqual(DEFAULT_CUSTOMIZATION.sidebarNavAuto, ["projects"]);
  assert.deepEqual(
    sanitizeCustomization({ sidebarNav: [...DEFAULT_CUSTOMIZATION.sidebarNav] })
      .sidebarNavAuto,
    ["projects"],
  );
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
  const reordered = sanitizeCustomization({
    sidebarNav: [
      { id: "train", pinned: true },
      ...DEFAULT_CUSTOMIZATION.sidebarNav.filter((entry) => entry.id !== "train"),
    ],
  });
  assert.deepEqual(reordered.sidebarNavAuto, []);
  assert.deepEqual(sanitizeCustomization({ sidebarNavAuto: [] }).sidebarNavAuto, []);
});

// The rail hides folder sections in CSS without unmounting, so the row must stay there.
test("the rail keeps the Projects row, since the section is hidden there", async () => {
  const sidebar = await readSrcAsync("components/app-sidebar.tsx");
  assert.match(
    sidebar,
    /const projectsSectionShowing =\n\s*projectsSectionConfigured && \(isMobile \|\| sidebarState !== "collapsed"\);/,
  );
  assert.match(sidebar, /if \(!projectsSectionRendered\) return null;/);
  assert.match(
    sidebar,
    /if \(!projectsSectionRendered\) return null;\n[\s\S]{0,400}?group-data-\[collapsible=icon\]:hidden/,
  );
});

// Pinned folder ids are not in the Projects list, so range-select uses the row's own list.
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
  assert.match(
    sidebar,
    /sidebarNavRowPinned\(item, sidebarNavAuto, \{ projectsSectionShowing \}\)/,
  );
  assert.match(
    sidebar,
    /placeNavRows\(\s*sidebarNav\.map\(\(item\) => \(\{ id: item\.id, pinned: navRowPinned\(item\) \}\)\),/,
  );
  assert.match(customizer, /checked=\{pinned\}/);
  assert.match(
    customizer,
    /sidebarNavAuto: sidebarNavAutoAfterChoice\(sidebarNavAuto, item\.id\)/,
  );
});

test("a route that borrows the section does not bring the row back", async () => {
  const sidebar = await readSrcAsync("components/app-sidebar.tsx");
  assert.match(
    sidebar,
    /const projectsSectionConfigured =\n\s*organizeBy === "project" &&\n\s*!projectsSectionHidden &&\n\s*\(projects\.length > 0 \|\| projectsLoaded\);/,
  );
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

test("the customizer reads the setting the same way", async () => {
  const customizer = await readSrcAsync(
    "features/settings/components/sidebar-nav-customizer.tsx",
  );
  assert.match(
    customizer,
    /const projectsSectionShowing =\n\s*organizeBy === "project" && projects\.length > 0;/,
  );
});
