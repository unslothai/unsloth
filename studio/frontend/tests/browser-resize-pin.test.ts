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

// The panel's page area (below the toolbar, with a 1px top border) and the page box inside it.
const style: Record<string, string> = { cssText: "" };
const page = {
  style,
  getBoundingClientRect: () => box(836, 135, 602, 1067),
};
const areaStyle: Record<string, string> = { cssText: "" };
const area = {
  style: areaStyle,
  clientTop: 1,
  getBoundingClientRect: () => box(836, 134, 602, 1068),
  querySelector: (selector: string) =>
    selector === ":scope > [data-browser-page]" ? page : null,
};
Object.assign(globalThis, {
  document: { querySelector: () => area },
});

const { pinBrowserPage } = await import("../src/features/browser/resize-pin.ts");

test("a page pinned for a split drag stays inside the page area, below the toolbar", () => {
  const unpin = pinBrowserPage({ getBoundingClientRect: () => box(826, 0, 20, 1200) } as Element);
  assert.equal(style.position, "absolute");
  assert.equal(style.top, "0px", "measured from inside the area's top border");
  assert.equal(style.bottom, "0");
  assert.equal(style.width, "602px");
  assert.equal(style.right, "0px", "anchored to the panel's outer edge, which stays put");
  unpin();
  assert.equal(style.cssText, "");
});

test("the page area itself keeps following the panel, so a widening drag shows the page background", () => {
  const unpin = pinBrowserPage({ getBoundingClientRect: () => box(826, 0, 20, 1200) } as Element);
  assert.deepEqual(areaStyle, { cssText: "" });
  unpin();
});
