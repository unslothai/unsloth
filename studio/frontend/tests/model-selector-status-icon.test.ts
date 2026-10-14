// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The loaded tick and the eject icon sit on the model name's capitals, not on the boxes' centres,
// which WKWebView draws the text about 1.5px away from.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const selector = readSrc("features/model-picker/components/model-selector.tsx");

test("the status icon's line uses the name's font, so its baseline is the name's", () => {
  assert.match(selector, /const TITLE_TEXT = "font-heading text-ui-16 font-medium leading-tight";/);
  assert.match(selector, /"flex max-w-full shrink-0 items-baseline whitespace-nowrap font-heading text-ui-16 font-medium leading-tight /);
  assert.match(selector, /const CAP_LINE = `block shrink-0 whitespace-nowrap text-center \$\{TITLE_TEXT\}`;/);
});

test("the icon centres in a capital-tall box resting on that baseline", () => {
  assert.match(selector, /const CAP_BOX = "relative inline-block h-\[0\.667em\] align-baseline";/);
  // A zero-width space gives the line a baseline before the box.
  assert.equal(selector.match(/\{"\\u200b"\}/g)?.length, 2);
  assert.doesNotMatch(selector, /"-m-1 flex size-5 shrink-0 cursor-pointer items-center/);
});

test("the eject hover circle is a square of its own, not the line's box", () => {
  assert.match(selector, /after:size-5 after:-translate-x-1\/2 after:-translate-y-1\/2 after:rounded-full/);
  assert.doesNotMatch(selector, /w-5 cursor-pointer rounded-full transition-colors hover:bg-/);
});

test("the header trigger's hover wash has less side padding", () => {
  assert.match(selector, /variant === "ghost" \? "h-9 pl-3 pr-2 text-sm" : "h-9 pl-4 pr-2 text-sm"/);
  assert.match(readSrc("features/chat/chat-page.tsx"), /className="max-w-\[62vw\] !pr-2\.5 /);
});
