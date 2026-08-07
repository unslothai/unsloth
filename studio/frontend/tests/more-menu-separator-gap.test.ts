// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The More flyout ends with a rule and the Customize sidebar row. The rule's
// vertical margin has to equal the menu's own padding, or the gap above that
// last row reads as bigger than the one below it.
const TAILWIND_UNIT = 4;

function spacing(classes: string, prefix: string): number {
  const m = new RegExp(`(?:^| )${prefix}-([0-9.]+)!?(?: |$)`).exec(classes);
  assert.ok(m, `no ${prefix}-* in "${classes}"`);
  return Number(m[1]) * TAILWIND_UNIT;
}

test("the More flyout's rule sits as far from its rows as the menu's own edge", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );

  const menu =
    /onPointerLeave=\{closeMoreSoon\}\s*\n\s*className="([^"]*)"/.exec(source);
  assert.ok(menu, "could not find the More flyout's DropdownMenuContent");
  const rule = /<DropdownMenuSeparator className="(mx-1![^"]*)"/.exec(source);
  assert.ok(rule, "could not find the More flyout's separator");

  assert.equal(spacing(rule[1], "my"), spacing(menu[1], "p"));
});
