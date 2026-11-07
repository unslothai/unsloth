// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// macOS desktop: a floating window's page showed white, and flashed white while dragged, because
// the fixed-background holes inside the window missed the page. Read from source: no DOM here.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const NATIVE = readSrc("features/browser/native-view.ts");

function body(name: string): string {
  const at = NATIVE.indexOf(`function ${name}(`);
  assert.notEqual(at, -1, `${name} not found`);
  return NATIVE.slice(at, NATIVE.indexOf("\n}\n", at));
}

test("nodes inside the floating window get no fixed-background hole; the window is masked", () => {
  const mark = body("markHole");
  assert.match(
    mark,
    /const floating = chain\.findIndex\(\(node\) => node\.hasAttribute\?\.\(FLOATING_ATTRIBUTE\)\)/,
  );
  assert.match(mark, /if \(index <= floating\) return;/);
  assert.match(
    mark,
    /masked = floating >= 0 \? \[\.\.\.layers, chain\[floating\]\] : layers;/,
  );
});

test("a mask's cut is the page's rect from the node's box, kept inside it, so a drag never voids it", () => {
  const place = body("placeHole");
  assert.match(
    place,
    /for \(const node of masked\) \{\s*const box = node\.getBoundingClientRect\(\);/,
  );
  assert.match(place, /const x = Math\.max\(0, rect\[0\] - box\.left\);/);
  assert.match(place, /const y = Math\.max\(0, rect\[1\] - box\.top\);/);
  assert.match(
    place,
    /Math\.min\(box\.width, rect\[0\] \+ rect\[2\] - box\.left\) - x/,
  );
  assert.match(
    place,
    /Math\.min\(box\.height, rect\[1\] \+ rect\[3\] - box\.top\) - y/,
  );
});
