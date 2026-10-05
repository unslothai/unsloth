// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The header's project switcher listed every project, so an account with thirty of them got a
// thirty-row menu with "View all projects" at the bottom of a scroll.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";

const SWITCHER = await readSrcAsync("features/chat/components/project-switcher.tsx");

test("the switcher lists the six most recent projects, not all of them", () => {
  assert.match(SWITCHER, /const RECENT_PROJECT_LIMIT = 6;/);
  assert.match(SWITCHER, /const recent = projects\.slice\(0, RECENT_PROJECT_LIMIT\);/);
  // The menu draws the capped list, not the prop.
  assert.match(SWITCHER, /\{recentProjects\.map\(\(project\) => \{/);
  assert.ok(
    !SWITCHER.includes("{projects.map("),
    "the menu still draws every project",
  );
  // The rest are one click away, so the cap hides nothing.
  assert.match(SWITCHER, /onSelect=\{onViewAllProjects\}/);
  assert.match(SWITCHER, /View all projects/);
});

// A project's own updatedAt only moves when it is edited, so the project being looked at is
// often not one of the six newest. Cutting it left the switcher showing no tick at all.
test("the open project keeps its row, without pushing the count past six", () => {
  assert.match(
    SWITCHER,
    /if \(!currentProject \|\| recent\.some\(\(p\) => p\.id === currentProject\.id\)\) \{\n\s*return recent;\n\s*\}/,
  );
  // Swapped for the oldest of the six, so the list is still six rows.
  assert.match(
    SWITCHER,
    /return \[\.\.\.recent\.slice\(0, RECENT_PROJECT_LIMIT - 1\), currentProject\];/,
  );
  // Recomputed when either side changes, or a switch would keep the last project's rows.
  assert.match(SWITCHER, /\}, \[projects, currentProject\]\);/);
});

// The loading and empty rows answer for the whole account, not for the slice: a capped list is
// never empty while projects exist, and would have said "No projects yet" over six of them.
test("the loading and empty rows still read the full list", () => {
  assert.match(SWITCHER, /const showLoadingRow = isLoading && projects\.length === 0;/);
  assert.match(SWITCHER, /const showEmptyRow = !isLoading && projects\.length === 0;/);
});
