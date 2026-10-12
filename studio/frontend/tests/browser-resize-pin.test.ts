// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

type Box = {
  top: number;
  bottom: number;
  left: number;
  right: number;
  width: number;
};
const box = (
  left: number,
  top: number,
  right: number,
  bottom: number,
): Box => ({
  left,
  top,
  right,
  bottom,
  width: right - left,
});

// The panel body: toolbar 0-60, bookmarks bar 60-90, page 90-700, all in one positioned parent.
const parent = { getBoundingClientRect: () => box(500, 40, 1200, 700) };
const page = {
  parentElement: parent,
  style: { cssText: "" } as Record<string, string>,
  getBoundingClientRect: () => box(500, 130, 1200, 700),
};
Object.assign(globalThis, { document: { querySelector: () => page } });

const { pinBrowserPage } = await import(
  "../src/features/browser/resize-pin.ts"
);

test("a page pinned for a split drag stays below the toolbar", () => {
  const handle = { getBoundingClientRect: () => box(480, 0, 500, 900) };
  const unpin = pinBrowserPage(handle as unknown as Element);
  assert.equal(page.style.position, "absolute");
  // At its own offset in the parent, not over the toolbar at top 0.
  assert.equal(page.style.top, "90px");
  assert.equal(page.style.bottom, "0px");
  assert.equal(page.style.width, "700px");
  assert.equal(page.style.right, "0px");
  unpin();
  assert.equal(page.style.cssText, "");
});
