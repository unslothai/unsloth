// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

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
  const source = await readSrcAsync("components/app-sidebar.tsx");

  // Found by its placement props in any order: #12339 put `ref={moreContentRef}` first
  // and a regex pinned to `<DropdownMenuContent side="right"` stopped matching.
  let menu: { className: string; end: number } | null = null;
  for (const open of source.matchAll(/<DropdownMenuContent\b/g)) {
    // The opening tag closes on its own line (`>` after the props); arrow functions inside it
    // carry `=>`, so a bare `>` search would stop inside a handler.
    const close = /\n\s*>\n/.exec(source.slice(open.index));
    if (!close) continue;
    const tag = source.slice(open.index, open.index + close.index + close[0].length);
    const className = /\sclassName="([^"]*)"/.exec(tag);
    if (
      className &&
      /\sside="right"/.test(tag) &&
      /\salign="start"/.test(tag) &&
      /\ssideOffset=\{6\}/.test(tag)
    ) {
      menu = { className: className[1], end: open.index + tag.length };
      break;
    }
  }
  assert.ok(menu, "could not find the More flyout's DropdownMenuContent");
  // The rule inside that flyout, not the first styled separator anywhere in the file.
  const flyout = source.slice(menu.end, source.indexOf("</DropdownMenuContent>", menu.end));
  const rule = /<DropdownMenuSeparator className="(mx-1![^"]*)"/.exec(flyout);
  assert.ok(rule, "could not find the More flyout's separator");

  assert.equal(spacing(rule[1], "my"), spacing(menu.className, "p"));
});
