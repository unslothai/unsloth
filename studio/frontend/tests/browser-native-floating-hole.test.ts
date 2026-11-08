// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Floating window paths, read from source (no DOM here).

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const NATIVE = readSrc("features/browser/native-view.ts");
const CSS = readSrc("index.css");

function body(name: string): string {
  const at = NATIVE.indexOf(`function ${name}(`);
  assert.notEqual(at, -1, `${name} not found`);
  return NATIVE.slice(at, NATIVE.indexOf("\n}\n", at));
}

test("backgrounds inside the floating window are cut in their own box, and the window is not masked", () => {
  const mark = body("markHole");
  assert.match(mark, /const floating = chain\.findIndex\(\(node\) => node\.hasAttribute\?\.\(FLOATING_ATTRIBUTE\)\)/);
  assert.match(mark, /if \(index <= floating\) local\.push\(node\);/);
  // A mask would hide UI drawn over the page too, e.g. the annotation comment form.
  assert.match(mark, /masked = layers;/);
  assert.match(CSS, /\[data-native-hole\]\[data-native-hole-local\] \{\s*background-attachment: scroll !important;\s*background-origin: border-box !important;/);
});

test("a box-local or mask cut is not trimmed to the box, and its lengths stay non-negative", () => {
  const place = body("placeHole");
  assert.match(place, /const x = rect\[0\] - box\.left;/);
  assert.match(place, /const \[width, height\] = \[rect\[2\], rect\[3\]\];/);
  const cut = body("cutAt");
  assert.match(cut, /100% \$\{Math\.max\(0, y\)\}px/);
  assert.match(cut, /\$\{Math\.max\(0, x\)\}px \$\{height\}px/);
});

test("every cut is also written as lengths, which WebKit repaints, and removed when unmarked", () => {
  const place = body("placeHole");
  assert.match(place, /cutAt\(node, "background", values\[0\], values\[1\], values\[2\], values\[3\]\);/);
  assert.match(place, /cutAt\(node, local\.includes\(node\) \? "background" : "mask", x, y, width, height\);/);
  const cut = body("cutAt");
  assert.match(cut, /const prefixes = kind === "background" \? \["background"\] : \["mask", "-webkit-mask"\];/);
  assert.match(cut, /setProperty\(`\$\{prefix\}-position`, position, "important"\)/);
  assert.match(body("unmarkHole"), /\.\.\.CUT_PROPERTIES\]\) node\.style\.removeProperty\(name\)/);
});

test("the panel-edge clip, for a docked split drag, is not applied in the floating window", () => {
  const clip = body("clipHole");
  assert.match(clip, /const docked = element && layered && !element\.closest\?\.\(`\[\$\{FLOATING_ATTRIBUTE\}\]`\);/);
  assert.match(clip, /const area = docked \? panelArea\(element\)\?\.getBoundingClientRect\(\) \?\? null : null;/);
});

test("switching between docked and floating shuts the cut until the page is moved and zoomed there", () => {
  const start = NATIVE.indexOf("  const sync = () => {");
  const sync = NATIVE.slice(start, NATIVE.indexOf("\n  };\n", start));
  assert.match(sync, /layered !== null && element && resized[^\n]*floats\(element\) !== floats\(resized\)\) \{\s*if \(layered\) shutHole\(\);/);
  assert.match(sync, /shutHole\(\);[\s\S]*?motionUntil = performance\.now\(\) \+ MOTION_MS;\s*\}/);
  const apply = body("applyView");
  const call = 'await call("browser_view_zoom", { tabId, zoom });';
  const show = apply.indexOf('await call("browser_view_show", { tabId, url:');
  const zoom = apply.indexOf(call, show);
  assert.ok(zoom > show, "zoomed after the move");
  assert.ok(zoom < apply.indexOf("if (layered) openHole(tabId, bounds);"), "and before the cut opens");
  const parked = apply.indexOf("if (rezoom && parkedView === tabId) {");
  assert.match(apply, /if \(rezoom && !layered && shownView === tabId && resized\) \{\s*await call\("browser_view_show", \{ tabId, bounds: shownBounds, parked: true \}\);/);
  assert.ok(parked >= 0 && apply.indexOf(call, parked) < show);
});

test("a moved or resized floating window parks its page under the picture on every platform", () => {
  assert.match(NATIVE, /function movingWindow\(\): boolean \{\s*return document\.querySelector\?\.\("\[data-floating-gesture\]"\) != null;/);
  assert.match(NATIVE, /const covered = movingWindow\(\) \|\| \(!layered && coveredNow\(rect\)\);/);
  assert.match(body("applyView"), /if \(desired && "covered" in desired && movingWindow\(\) && parkable\(desired\.tabId\)\) \{\s*await parkForDrag/);
  assert.match(NATIVE, /coverForDrag\(tabId\);\s*(\/\/[^\n]*\n\s*)*sent = "";\s*if \(!layered && parkable\(tabId\)\) void parkForDrag\(tabId\);/);
  assert.match(readSrc("features/browser/floating-preview.ts"), /event\.preventDefault\(\);\s*announceGrab\(\);/);
});

test("a drag picture taken at another size (before the page floated) is never shown", () => {
  assert.match(body("showDragShot"), /!sameSize\(dragShot\.size, shownBounds\)\) return;/);
  assert.match(body("captureForDrag"), /const size = shownBounds;/);
  assert.match(body("sameSize"), /Math\.abs\(a\.width - b\.width\) <= 2 && Math\.abs\(a\.height - b\.height\) <= 2/);
});

test("a resize crops the drag picture instead of stretching it", () => {
  const show = body("showDragShot");
  assert.match(show, /width: dragShot\.layoutWidth \? `\$\{dragShot\.layoutWidth\}px` : "100%",\s*height: "auto",/);
  assert.match(body("captureForDrag"), /const layoutWidth = placeholder\(tabId\)\?\.clientWidth \?\? 0;/);
  assert.match(show, /page\.style\.overflow = "hidden";/);
  assert.match(body("hideDragShot"), /removeProperty\("overflow"\)/);
});

test("a park started on a gesture's first frame finishes before the drop's restore", () => {
  assert.match(body("applyView"), /function applyView\(desired: Desired\): Promise<void> \{\s*if \(parking\) await parking\.done;/);
  assert.match(body("parkForDrag"), /if \(parking\?\.tabId === tabId\) return parking\.done;/);
});
