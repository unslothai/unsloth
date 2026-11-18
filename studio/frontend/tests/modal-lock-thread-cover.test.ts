// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const CSS = readSrc("index.css");

function rule(selector: string): string {
  const at = CSS.indexOf(`${selector} {`);
  assert.ok(at >= 0, `missing rule ${selector}`);
  return CSS.slice(at, CSS.indexOf("}", at) + 1);
}

test("the thread scroller does not inherit a modal layer's body pointer-events lock", () => {
  // Radix writes pointer-events: none on <body> for dialogs and modal menus. Inherited, it
  // restyled every message on each open and close; an explicit value stops it at the scroller.
  // Zero specificity: any deliberate pointer-events rule on the scroller still wins.
  assert.match(rule(":where(.aui-thread-viewport)"), /pointer-events: auto;/);
});

test("while the lock is up a cover keeps the thread unclickable, below portaled layers", () => {
  // Keyed on attributes src/lib/modal-pointer-lock.ts mirrors from the body lock: on the thread
  // root itself, or on a page region holding a thread root it did not find.
  const cover = rule(
    '.aui-thread-root[data-modal-lock-thread]::after,\n\t[data-modal-lock-region] .aui-thread-root::after',
  );
  assert.match(cover, /content: "";/);
  assert.match(cover, /position: absolute;/);
  assert.match(cover, /inset: 0;/);
  // Its own value: it would otherwise inherit none from the locked body and let clicks through.
  assert.match(cover, /pointer-events: auto;/);
  // Below the z-50 dialogs, menus and popovers that Radix portals to <body>.
  const z = Number(/z-index: (\d+);/.exec(cover)?.[1]);
  assert.ok(z > 0 && z < 50, `cover z-index ${z} must sit under the z-50 portaled layers`);
  // The root positions the cover, and no other rule draws on its ::after.
  assert.match(
    readSrc("components/assistant-ui/thread.tsx"),
    /className="aui-root aui-thread-root relative /,
  );
  assert.equal(CSS.split(".aui-thread-root[data-modal-lock-thread]::after").length - 1, 1);
  assert.equal(CSS.split(" .aui-thread-root::after").length - 1, 1);
  // Plus the Firefox-only permanent cover below.
  assert.equal(CSS.split("\t.aui-thread-root::after {").length - 1, 1);
});

test("in Firefox the cover is permanent and inert, so a lock only flips its pointer-events", () => {
  // Adding the cover on each open made Firefox lay out the thread root's flex column again, every
  // message included (about 250 ms per modal open and close at 150 turns). Firefox only: in WebKit a
  // permanent cover slowed the Glass browser pane open.
  const at = CSS.indexOf("@supports (-moz-appearance: none) {\n\t\t.aui-thread-root::after {");
  assert.ok(at >= 0, "missing the Firefox-only permanent cover");
  const block = CSS.slice(at, CSS.indexOf("}", at) + 1);
  assert.match(block, /content: "";/);
  assert.match(block, /position: absolute;/);
  assert.match(block, /inset: 0;/);
  assert.match(block, /z-index: 49;/);
  assert.match(block, /pointer-events: none;/);
  assert.doesNotMatch(block, /background|backdrop-filter|box-shadow/);
  // Lower specificity than the lock rule, which sets pointer-events: auto.
  assert.ok(at > CSS.indexOf("[data-modal-lock-region] .aui-thread-root::after {"));
});

