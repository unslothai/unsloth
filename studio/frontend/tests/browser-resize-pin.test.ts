// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

const box = (left: number, top: number, width: number, height: number) => ({
  left,
  top,
  width,
  height,
  right: left + width,
  bottom: top + height,
});

const style: Record<string, string> = { cssText: "" };
const page = {
  style,
  getBoundingClientRect: () => box(836, 134, 602, 1068),
  parentElement: { getBoundingClientRect: () => box(836, 83, 602, 1119) },
};
Object.assign(globalThis, {
  document: { querySelector: () => page },
});

const { pinBrowserPage } = await import("../src/features/browser/resize-pin.ts");

test("a page pinned for a split drag stays below the browser's toolbar", () => {
  const unpin = pinBrowserPage({ getBoundingClientRect: () => box(826, 0, 20, 1200) } as Element);
  assert.equal(style.position, "absolute");
  assert.equal(style.top, "51px", "pinned at top: 0, the page covered the address bar and bookmarks for the whole drag");
  assert.equal(style.bottom, "0");
  assert.equal(style.width, "602px");
  assert.equal(style.right, "0px", "anchored to the panel's outer edge, which stays put");
  unpin();
  assert.equal(style.cssText, "");
});
