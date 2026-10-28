// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const SIDEBAR = readSrc("components/app-sidebar.tsx");

function block(source: string, start: string, end: string): string {
  const from = source.indexOf(start);
  assert.notEqual(from, -1, `missing ${start}`);
  const to = source.indexOf(end, from + start.length);
  assert.notEqual(to, -1, `missing ${end} after ${start}`);
  return source.slice(from, to);
}

test("an unpinned Images row opens its workflows from More", () => {
  const more = block(SIDEBAR, "{overflowNavIds.map((id) => {", "<MoreMenuItem");
  assert.match(
    more,
    /<ImagesMoreSubmenu\s+key=\{id\}\s+\{\.\.\.submenu\}\s+onPick=\{pickImagesWorkflow\}/,
  );

  const images = block(
    SIDEBAR,
    "function ImagesMoreSubmenu(",
    "function AudioMoreSubmenu(",
  );
  assert.match(images, /<MediaMoreSubmenu/);
  assert.match(images, /tabs=\{WORKFLOW_TABS\}/);
  assert.match(
    images,
    /enabled=\{\(id\) => isWorkflowEnabled\(id, supported\)\}/,
  );
  assert.match(
    images,
    /const current = props\.active && pageMode === "create" \? workflow : null;/,
  );

  const submenu = block(
    SIDEBAR,
    "function MediaMoreSubmenu<",
    "function ImagesMoreSubmenu(",
  );
  assert.match(submenu, /disabled=\{!enabled\(tab\.id\)\}/);
  assert.match(
    submenu,
    /title=\{enabled\(tab\.id\) \? undefined : WORKFLOW_UNAVAILABLE\}/,
  );
});

test("an Images workflow picked from More or the sidebar selects it and opens Images", () => {
  const pick = block(SIDEBAR, "const pickImagesWorkflow =", "};");
  assert.match(
    pick,
    /useImageWorkflowStore\.getState\(\)\.setWorkflow\(workflowId\);/,
  );
  assert.match(pick, /navigate\(\{ to: "\/images" \}\);/);
  assert.match(pick, /closeMobileIfOpen\(\);/);
  assert.match(
    SIDEBAR,
    /<ImagesWorkflowList\s+active=\{row\.active\}\s+collapsed=\{!sidebarRowsLabelled\}\s+onPick=\{pickImagesWorkflow\}/,
  );
});

test("the open Images page keeps its row, and so its workflows, on the sidebar", () => {
  assert.match(
    SIDEBAR,
    /placeNavRows\([\s\S]*?navRows\.images\.active \? "images" : null,\s*\);/,
  );
});
