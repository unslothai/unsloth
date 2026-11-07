// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// macOS desktop: a floating window's page showed white because the fixed-background cuts inside the
// window missed it. Read from source: no DOM here.

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

test("a box-local cut is the page's rect from the node's box, kept inside it, so it is never voided", () => {
  const place = body("placeHole");
  assert.match(place, /if \(local\.includes\(node\)\) continue;/);
  assert.match(place, /for \(const node of \[\.\.\.local, \.\.\.masked\]\) \{\s*const box = node\.getBoundingClientRect\(\);/);
  assert.match(place, /const x = Math\.max\(0, rect\[0\] - box\.left\);/);
  assert.match(place, /const y = Math\.max\(0, rect\[1\] - box\.top\);/);
  assert.match(place, /Math\.min\(box\.width, rect\[0\] \+ rect\[2\] - box\.left\) - x/);
  assert.match(place, /Math\.min\(box\.height, rect\[1\] \+ rect\[3\] - box\.top\) - y/);
});
