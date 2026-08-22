// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = () =>
  readFile(
    new URL("../src/components/tauri/window-titlebar.tsx", import.meta.url),
    "utf8",
  );
const DESKTOP_SURFACE_PATTERN =
  /const showDesktopSidebarSurface = showSidebarSurface && !isMobile;/;
const SIDEBAR_WIDTH_PATTERN =
  /const sidebarWidth = showDesktopSidebarSurface[\s\S]*?: "0px";/;
const NAVIGATION_WIDTH_PATTERN =
  /showSidebarSurface && \(!showDesktopSidebarSurface \|\| !pinned\)[\s\S]*?\? "7rem"/;
const CONTENT_BORDER_PATTERN =
  /showDesktopSidebarSurface && pinned[\s\S]*?`calc\(\$\{sidebarWidth\} \+ 12px\)`/;

test("mobile titlebar excludes desktop sidebar geometry", async () => {
  const titlebar = await source();
  const decorationIndex = titlebar.indexOf(
    'data-slot="window-titlebar-decoration"',
  );
  const decorationGateIndex = titlebar.lastIndexOf(
    "{showDesktopSidebarSurface && (",
    decorationIndex,
  );

  assert.match(titlebar, DESKTOP_SURFACE_PATTERN);
  assert.match(titlebar, SIDEBAR_WIDTH_PATTERN);
  assert.match(titlebar, NAVIGATION_WIDTH_PATTERN);
  assert.match(titlebar, CONTENT_BORDER_PATTERN);
  assert.notEqual(decorationIndex, -1);
  assert.notEqual(decorationGateIndex, -1);
});
