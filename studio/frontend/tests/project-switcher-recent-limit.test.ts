// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";

const SWITCHER = await readSrcAsync("features/chat/components/project-switcher.tsx");

test("the switcher lists the six most recent projects, not all of them", () => {
  assert.match(SWITCHER, /const RECENT_PROJECT_LIMIT = 6;/);
  assert.match(SWITCHER, /const recent = projects\.slice\(0, RECENT_PROJECT_LIMIT\);/);
  assert.match(SWITCHER, /\{recentProjects\.map\(\(project\) => \{/);
  assert.ok(
    !SWITCHER.includes("{projects.map("),
    "the menu still draws every project",
  );
  assert.match(SWITCHER, /onSelect=\{onViewAllProjects\}/);
  assert.match(SWITCHER, /View all projects/);
});

// updatedAt moves only on edit, so the open project may not be among the newest six.
test("the open project keeps its row, without pushing the count past six", () => {
  assert.match(
    SWITCHER,
    /if \(!currentProject \|\| recent\.some\(\(p\) => p\.id === currentProject\.id\)\) \{\n\s*return recent;\n\s*\}/,
  );
  assert.match(
    SWITCHER,
    /return \[\.\.\.recent\.slice\(0, RECENT_PROJECT_LIMIT - 1\), currentProject\];/,
  );
  assert.match(SWITCHER, /\}, \[projects, currentProject\]\);/);
});

// A capped list is never empty while projects exist, so loading/empty read the full list.
test("the loading and empty rows still read the full list", () => {
  assert.match(SWITCHER, /const showLoadingRow = isLoading && projects\.length === 0;/);
  assert.match(SWITCHER, /const showEmptyRow = !isLoading && projects\.length === 0;/);
});
