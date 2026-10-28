// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  NAV_FLYOUT_INTENT,
  NAV_TOOLTIP_INTENT,
  isPointerHeadingInto,
} from "../src/lib/hover-intent.ts";
import { readSrc } from "./helpers/kit.ts";

const FLYOUT_RIGHT = { left: 300, top: 100, right: 492, bottom: 340 };
const FLYOUT_LEFT = { left: 400, top: 100, right: 690, bottom: 340 };

test("a pointer inside the flyout is always heading into it", () => {
  assert.equal(isPointerHeadingInto({ x: 290, y: 110 }, { x: 400, y: 300 }, FLYOUT_RIGHT), true);
});

test("leaving the row toward the flyout keeps it, straight or diagonal", () => {
  assert.equal(isPointerHeadingInto({ x: 290, y: 110 }, { x: 296, y: 110 }, FLYOUT_RIGHT), true);
  assert.equal(isPointerHeadingInto({ x: 290, y: 110 }, { x: 295, y: 125 }, FLYOUT_RIGHT), true);
  assert.equal(isPointerHeadingInto({ x: 150, y: 133 }, { x: 220, y: 170 }, FLYOUT_RIGHT), true);
});

test("moving on to the next row or back away is not heading into it", () => {
  assert.equal(isPointerHeadingInto({ x: 150, y: 133 }, { x: 150, y: 160 }, FLYOUT_RIGHT), false);
  assert.equal(isPointerHeadingInto({ x: 150, y: 133 }, { x: 140, y: 120 }, FLYOUT_RIGHT), false);
  assert.equal(isPointerHeadingInto({ x: 290, y: 110 }, { x: 300, y: 60 }, FLYOUT_RIGHT), false);
});

test("a flyout flipped to the other side is aimed at from that side", () => {
  assert.equal(isPointerHeadingInto({ x: 700, y: 110 }, { x: 695, y: 115 }, FLYOUT_LEFT), true);
  assert.equal(isPointerHeadingInto({ x: 700, y: 110 }, { x: 720, y: 115 }, FLYOUT_LEFT), false);
});

test("nav labels wait for intent, then chain without delay or hover grace", () => {
  assert.ok(NAV_TOOLTIP_INTENT.delayDuration > 0);
  assert.ok(NAV_TOOLTIP_INTENT.skipDelayDuration > 0);
  assert.equal(NAV_TOOLTIP_INTENT.disableHoverableContent, true);
});

test("the More flyout opens sooner than a label and outlasts the gap to it", () => {
  assert.ok(NAV_FLYOUT_INTENT.openDelay > 0);
  assert.ok(NAV_FLYOUT_INTENT.openDelay < NAV_TOOLTIP_INTENT.delayDuration);
  assert.ok(NAV_FLYOUT_INTENT.closeDelay > NAV_FLYOUT_INTENT.openDelay);
});

test("the hover flyout answers the mouse only", () => {
  const hook = readSrc("hooks/use-hover-flyout.ts");
  assert.equal(hook.match(/event\.pointerType !== "mouse"/g)?.length, 3);
  assert.equal(hook.match(/event\.pointerType === "mouse"/g)?.length, 1);
  assert.match(hook, /isPointerHeadingInto\(exit, pointer, target\)/);
});

test("a hover timer that fires under a modal opened meanwhile leaves the flyout shut (#9244)", () => {
  const hook = readSrc("hooks/use-hover-flyout.ts");
  assert.match(
    hook,
    /setTimeout\(\(\) => \{\s*if \(!isBlockedByActiveModal\(trigger\)\) setOpen\(true\);\s*\}, intent\.openDelay\)/,
  );
  assert.doesNotMatch(hook, /setTimeout\(\(\) => setOpen\(true\)/);
});

test("the sidebar scopes nav tooltip intent to itself", () => {
  const sidebar = readSrc("components/app-sidebar.tsx");
  assert.match(sidebar, /<TooltipProvider \{\.\.\.NAV_TOOLTIP_INTENT\}>\s*<Sidebar\b/);
  assert.match(sidebar, /<\/Sidebar>\s*<\/TooltipProvider>/);
});

test("only a tooltip that waited animates in, so chained ones glide", () => {
  const tooltip = readSrc("components/ui/tooltip.tsx");
  assert.match(tooltip, /data-\[state=delayed-open\]:animate-in/);
  assert.match(tooltip, /origin-\(--radix-tooltip-content-transform-origin\)/);
  assert.doesNotMatch(tooltip, /data-\[state=instant-open\]:animate-in/);
  assert.doesNotMatch(tooltip, /data-\[state=closed\]:animate-out/);
});
