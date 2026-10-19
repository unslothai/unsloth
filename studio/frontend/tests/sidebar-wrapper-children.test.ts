// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("pinning the desktop sidebar never adds or removes a child of the sidebar wrapper", () => {
  const trigger = readSrc("components/sidebar-edge-trigger.tsx");
  assert.match(trigger, /if \(!isTauri\) \{\s*return null;\s*\}/);
  assert.doesNotMatch(
    trigger,
    /sidebarShowing && !holding\)\) \{\s*return null/,
    "returning null while pinned inserted the strip between the sidebar and <main> on every unpin",
  );
  assert.match(
    trigger,
    /className="contents"[\s\S]*?\{\(!sidebarShowing \|\| holding\) && \(\s*<PanelResizeHandle/,
    "the container stays mounted and only its handle comes and goes",
  );
  assert.match(
    readSrc("components/ui/sidebar.tsx"),
    /md:peer-data-\[variant=inset\]:m-2/,
    "<main>'s peer rules flag the wrapper for sibling changes, and with KaTeX's MathML on the page Chromium restyles everything below a flagged parent when a child is inserted before another",
  );
});
