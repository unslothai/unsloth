// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Both floating panels dragged through React state and through left/top, which
// is the layout path: the card carries a wide blurred shadow and the monitor is
// backdrop-blurred, so every frame re-laid-out and repainted them. A trackpad
// also reports moves faster than the display refreshes, so much of that work
// was never shown. These pin the cheap path, since the cost is invisible to a
// unit test and only shows up under the hand.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function source(path: string): string {
  return readFileSync(new URL(`../src/${path}`, import.meta.url), "utf8");
}

test("the card paints a drag frame through a transform", () => {
  const hook = source("features/loaded-models/use-drag-position.ts");
  assert.match(hook, /panel\.style\.transform = `translate3d\(/);
  assert.match(hook, /frameRef\.current = requestAnimationFrame\(paint\)/);
  // The old path: one setPosition per pointermove, so one render per event.
  assert.doesNotMatch(hook, /setPosition\(\s*clampToViewport\(/);
});

// This one is the sharpest: the effect was keyed on `position`, so every frame
// disconnected the observer and built a new one, and observing re-measures,
// which forces a synchronous layout.
test("the card's reclamp observer outlives a drag frame", () => {
  const hook = source("features/loaded-models/use-drag-position.ts");
  assert.match(hook, /\}, \[panelEl, reclamp\]\);/);
  assert.doesNotMatch(hook, /\}, \[position, panelEl, reclamp\]\);/);
});

test("the card hands the offset back to left/top on release", () => {
  const hook = source("features/loaded-models/use-drag-position.ts");
  assert.match(hook, /panel\.style\.left = `\$\{session\.lastLeft\}px`/);
  assert.match(hook, /panel\.style\.transform = ""/);
  assert.match(
    hook,
    /setPosition\(\{ left: session\.lastLeft, top: session\.lastTop \}\)/,
  );
});

test("the monitor paints a drag frame through a transform", () => {
  const panel = source("components/floating-monitor.tsx");
  assert.match(panel, /monitor\.style\.transform = `translate3d\(/);
  assert.match(
    panel,
    /dragFrameRef\.current = requestAnimationFrame\(paintDrag\)/,
  );
});

test("the monitor commits its position once, on release", () => {
  const panel = source("components/floating-monitor.tsx");
  const finish = panel.slice(panel.indexOf("function finishDrag"));
  assert.match(finish, /monitor\.style\.transform = ""/);
  assert.match(finish, /setLayout\(\(current\) =>/);
});

// The measured box carries the drag's transform, so committing it mid-drag
// would move the panel twice as far as the pointer went.
test("a resize mid-drag does not commit the transformed box", () => {
  const panel = source("components/floating-monitor.tsx");
  assert.match(panel, /const held = session && current \? current : null;/);
  assert.match(panel, /left: restLeft,/);
  assert.match(panel, /maxWidth: constraintsBox\.width - restLeft,/);
});

// Every frame published the monitor's box to a shared store, and each write
// re-rendered every overlay subscribed to it, the loaded models card included.
test("dragging the monitor does not republish its frame per frame", () => {
  const panel = source("components/floating-monitor.tsx");
  const update = panel.slice(
    panel.indexOf("function updateDrag"),
    panel.indexOf("function finishDrag"),
  );
  assert.doesNotMatch(update, /setFrame/);
  assert.doesNotMatch(update, /getBoundingClientRect/);
});
