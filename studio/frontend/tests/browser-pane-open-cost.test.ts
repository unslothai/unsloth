// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const PAGE = readSrc("features/chat/chat-page.tsx");
const CSS = readSrc("index.css");

test("the side-panel insets are found among the root's children, not by a :scope query over the thread", () => {
  const effect = PAGE.slice(PAGE.indexOf("// Width on header/notice only"), PAGE.indexOf("const resizeObserver = new ResizeObserver"));
  assert.match(effect, /const insets = \(\) =>\s*\[\.\.\.root\.children\]\.filter\(/);
  assert.doesNotMatch(effect, /querySelectorAll[^;]*:scope > \[data-side-panel-inset\]/,
    "the resize observer runs this each frame of the pane's open and close");
});

test("a long chat takes the pane's width in one step; a short one keeps the slide", () => {
  assert.match(CSS, /\.chat-artifact-split\[data-artifact-layout-animating="true"\]:not\(\[data-artifact-layout-instant\]\) > \[data-panel\] \{\s*transition:/);
  assert.match(PAGE, /import \{ rowReachPx \} from "@\/components\/assistant-ui\/distant-rows";/);
  assert.match(PAGE, /split\.toggleAttribute\(\s*ARTIFACT_LAYOUT_INSTANT_ATTRIBUTE,\s*viewport !== null &&\s*viewport\.clientHeight > 0 &&\s*viewport\.scrollHeight > viewport\.clientHeight \+ 2 \* rowReachPx\(window\.innerHeight\),\s*\);/);
  assert.match(PAGE, /const prepFrameId = window\.requestAnimationFrame\(\(\) => \{\s*markInstantLayout\(contextSurfaceRef\.current\);\s*resizeFrameId = window\.requestAnimationFrame\(\(\) => \{\s*panel\.resize\(/);
  assert.doesNotMatch(PAGE.slice(PAGE.indexOf("function markInstantLayout"), PAGE.indexOf("const SingleContent")), /:scope >/);
});
