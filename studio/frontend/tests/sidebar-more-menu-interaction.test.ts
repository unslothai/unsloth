// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

async function moreFlyout() {
  const source = await readSrcAsync("components/app-sidebar.tsx");
  const start = source.indexOf(
    "{/* Unpinned destinations, behind one row. */}",
  );
  const end = source.indexOf("</DropdownMenuContent>", start);
  // Empty when the marker moves, so every assertion on it fails.
  return { source, flyout: start === -1 ? "" : source.slice(start, end) };
}

test("the sidebar More flyout previews on mouse hover and pins on a press", async () => {
  const { source, flyout } = await moreFlyout();

  assert.match(
    source,
    /const moreHover = useHoverFlyout\(NAV_FLYOUT_INTENT, moreContentRef\);/,
  );
  assert.match(source, /const moreOpen = moreHover\.open \|\| morePinnedOpen/);
  assert.match(flyout, /ref=\{moreTriggerRef\}\s*\{\.\.\.moreHover\.trigger\}/);
  assert.match(flyout, /\{\.\.\.moreHover\.content\}/);
  assert.doesNotMatch(flyout, /<SidebarMenuItem\s+onPointer/);
  // A press pins or unpins, and is not treated as an outside press.
  assert.match(
    flyout,
    /onPointerDown=\{\(event\) => \{[\s\S]*?event\.preventDefault\(\);[\s\S]*?setMoreOpen\(!morePinnedOpen\);/,
  );
  assert.match(
    flyout,
    /onPointerDownOutside=\{\(event\) => \{\s*if \(moreTriggerRef\.current\?\.contains\(event\.target as Node\)\) event\.preventDefault\(\);/,
  );
});

test("the More flyout keeps its tooltip hidden while open (#12216)", async () => {
  const { source, flyout } = await moreFlyout();

  assert.match(flyout, /open=\{moreOpen\}\s*\n\s*onOpenChange=\{setMoreOpen\}/);
  assert.match(flyout, /<DropdownMenuTrigger asChild>/);
  assert.match(flyout, /open=\{moreTooltipOpen && !moreOpen\}/);
  assert.match(flyout, /onOpenChange=\{handleMoreTooltipOpenChange\}/);
  assert.match(
    source,
    /if \(!\(next && moreFocusReturning\.current\)\) setMoreTooltipOpen\(next\)/,
  );
  assert.match(
    flyout,
    /moreFocusReturning\.current = true;\s*queueMicrotask\(/,
  );
  // Swallowing the press left the tooltip over the menu before #12216.
  assert.doesNotMatch(flyout, /stopPropagation|onPointerDownCapture/);
});

test("a hover preview leaves focus where it was", async () => {
  const { source, flyout } = await moreFlyout();

  // Only a chosen opening moves focus.
  assert.match(source, /if \(next\) moreChosen\.current = true;/);
  assert.match(
    source,
    /onOpenAutoFocus: \(event: Event\) => \{\s*if \(!moreChosen\.current\) event\.preventDefault\(\);/,
  );
  assert.match(flyout, /\{\.\.\.moreContentFocusProps\}/);
  // Pinning an open preview focuses the menu, as a click-open does.
  assert.match(flyout, /ref=\{moreContentRef\}/);
  assert.match(
    flyout,
    /if \(!morePinnedOpen && moreHover\.open\) \{\s*moreContentRef\.current\?\.focus\(\{ preventScroll: true \}\);/,
  );
  assert.match(
    flyout,
    /onCloseAutoFocus=\{\(event\) => \{\s*const chosen = moreChosen\.current;\s*moreChosen\.current = false;\s*if \(!chosen\) \{\s*event\.preventDefault\(\);/,
  );
});
