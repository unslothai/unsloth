// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

// The desktop collapse state must not hide workflows in the mobile sheet.
test("the Images workflow list treats the mobile sheet as expanded", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");

  assert.match(
    source,
    /const sidebarRowsLabelled = isMobile \|\| sidebarState !== "collapsed";/,
  );
  assert.match(
    source,
    /<ImagesWorkflowList\s+active=\{row\.active\}\s+collapsed=\{!sidebarRowsLabelled\}/,
  );
  assert.match(
    source,
    /const imagesWorkflowsListed =\s+sidebarRowsLabelled &&/,
  );

  const imagesRow = source.slice(
    source.indexOf('id === "images" && "group/images-item"'),
    source.indexOf("<ImagesWorkflowList"),
  );
  assert.ok(
    imagesRow.includes("<ImagesNavDisclosure />"),
    "could not find the Images row overlay",
  );
  assert.ok(
    !imagesRow.includes("sidebarState"),
    "the Images row overlay still reads the desktop pin state",
  );
});
