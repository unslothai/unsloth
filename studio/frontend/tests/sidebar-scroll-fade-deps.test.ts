// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

// onScroll never fires for a list that is not scrolling, and rows can change without rendering
// AppSidebar (the Images disclosure, project toggles, data landing at startup). The bottom fade
// follows the scroller's size instead of a list of row inputs, which went stale on any it missed.

test("the bottom fade re-measures off the list's size, not its row inputs", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");
  assert.equal(/Recompute bottom-fade on mount/.test(source), false);
  assert.equal(/window\.addEventListener\("resize"/.test(source), false);
  // The box and every section in it, including ones that mount later.
  assert.match(source, /observer\.observe\(el\);/);
  assert.match(source, /for \(const section of el\.children\) observer\.observe\(section\);/);
  assert.match(source, /sections\.observe\(el, \{ childList: true \}\);/);
  // A section that leaves is unobserved, so the observer does not hold its detached subtree,
  // and the fade re-measures since an unobserved node reports nothing.
  assert.match(source, /if \(node instanceof Element\) observer\.unobserve\(node\);/);
  assert.match(source, /if \(node instanceof Element\) observer\.observe\(node\);/);
  assert.match(source, /observer\.observe\(node\);\s*\}\s*\}\s*syncFade\(el\);/);
  // Measured on attach from both ends, since the footer mounts after the scroller.
  assert.match(source, /measureScrollRail\(el\);\s*syncFade\(el\);/);
  assert.match(source, /if \(node && scrollRef\.current\) syncFade\(scrollRef\.current\);/);
  // React never owns the attribute, so a render cannot reset it.
  assert.match(source, /ref=\{attachFade\}/);
  assert.match(source, /"opacity-0 data-\[visible=true\]:opacity-100"/);
  assert.equal(/data-visible=/.test(source), false);
});
