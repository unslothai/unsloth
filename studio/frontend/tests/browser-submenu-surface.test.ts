// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Dark submenus no longer copy their parent's fill, so each one carries its
// parent's surface class.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

test("every browser panel submenu uses the browser menu surface", async () => {
  const panel = await readSrcAsync("features/browser/browser-panel.tsx");
  const subs = [...panel.matchAll(/<DropdownMenuSubContent className="([^"]*)"/g)];
  assert.ok(subs.length > 0);
  for (const [, cls] of subs) assert.match(cls, /\bbrowser-menu\b/, cls);
});

test("pinned page menus hand their sidebar surface to the tab submenus", async () => {
  const row = await readSrcAsync("features/browser/pinned-page-row.tsx");
  const items = await readSrcAsync("features/browser/tab-menu.tsx");
  assert.match(row, /const SURFACE = "unsloth-plus-menu sidebar-row-menu sidebar-menu";/);
  assert.equal(row.match(/<TabMenuItems [^>]*subClassName=\{SURFACE\}/g)?.length, 2);
  assert.match(items, /<P\.SubContent className=\{cn\(subClassName, "w-48"\)\}>/);
});
