// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

test("the sidebar More flyout opens only on click and hides its tooltip while open", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");
  const start = source.indexOf("{/* Unpinned destinations, behind one row. */}");
  assert.notEqual(start, -1, "could not find the More flyout");
  const flyout = source.slice(start, source.indexOf("</SidebarMenuItem>", start));

  assert.match(source, /const \[moreOpen, setMoreOpen\] = useState\(false\)/);
  assert.match(flyout, /open=\{moreOpen\}\s*\n\s*onOpenChange=\{setMoreOpen\}/);
  assert.match(flyout, /<DropdownMenuTrigger asChild>/);
  assert.match(flyout, /open=\{moreTooltipOpen && !moreOpen\}/);
  assert.match(flyout, /onOpenChange=\{handleMoreTooltipOpenChange\}/);
  assert.match(source, /if \(!\(next && moreFocusReturning\.current\)\) setMoreTooltipOpen\(next\)/);
  assert.match(flyout, /onCloseAutoFocus=\{\(\) => \{\s*moreFocusReturning\.current = true;\s*queueMicrotask\(/);
  assert.doesNotMatch(flyout, /onPointerEnter|onPointerLeave|onPointerDownCapture/);
  assert.doesNotMatch(
    source,
    /openMorePreview|closeMorePreviewSoon|moreHoverOpen|morePinnedOpen/,
  );
});
