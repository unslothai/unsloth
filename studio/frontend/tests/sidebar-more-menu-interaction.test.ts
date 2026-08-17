// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("the sidebar More flyout previews on hover and pins on click", async () => {
  const source = await readFile(
    new URL("../src/components/app-sidebar.tsx", import.meta.url),
    "utf8",
  );

  assert.match(source, /const \[moreHoverOpen, setMoreHoverOpen\] = useState\(false\)/);
  assert.match(source, /const \[morePinnedOpen, setMorePinnedOpen\] = useState\(false\)/);
  assert.match(source, /const moreOpen = moreHoverOpen \|\| morePinnedOpen/);
  assert.match(source, /open=\{moreOpen\}\s*\n\s*onOpenChange=\{handleMoreOpenChange\}/);
  assert.match(source, /<DropdownMenuTrigger asChild>/);
  assert.match(source, /onPointerEnter=\{openMorePreview\}/);
  assert.match(source, /onPointerLeave=\{closeMorePreviewSoon\}/);
  assert.match(source, /onPointerDownCapture=/);
  assert.match(source, /event\.preventDefault\(\)/);
  assert.match(source, /event\.stopPropagation\(\)/);
  assert.match(source, /if \(morePinnedOpen\)/);
});
