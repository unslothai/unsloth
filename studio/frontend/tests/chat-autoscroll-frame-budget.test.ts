// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The chat viewport followed the bottom by chaining requestAnimationFrame off a 600ms window that
// every mutation re-armed, so a streaming message forced a layout read every frame throughout.
// Measured with tests/studio/playwright_chat_autoscroll.py on a frame pump, a message every 250ms
// (deep research synthesis cadence) held the loop at the pump's ceiling of 62 frames a second;
// after this change it settles to the 100ms settle check, about 12.
//
// The cost is invisible to a rendering test and the hook is a .tsx, which node's type stripping
// cannot import, so the shape is pinned here as drag-costs-no-render.test.ts pins the drag path:
// assert the cheap path, and assert the expensive one is gone.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const HOOK = readFileSync(
  new URL(
    "../src/components/assistant-ui/use-intent-aware-autoscroll.tsx",
    import.meta.url,
  ),
  "utf8",
);

/** The body of `tick`, which is the only place the chain re-arms. */
function tickBody(): string {
  const start = HOOK.indexOf("const tick = (): void => {");
  assert.notEqual(start, -1, "tick is gone; this test needs rewriting");
  const end = HOOK.indexOf("\n      };", start);
  return HOOK.slice(start, end);
}

test("a following frame re-arms only while layout is still moving", () => {
  const tick = tickBody();
  assert.match(tick, /if \(layoutChanged \|\| !pinned\) \{/);
  assert.match(tick, /layoutChanged = false;/);
  // The old path: one requestTick per frame while the window stayed open, and every mutation
  // extended it, so a streaming message never let it close.
  assert.doesNotMatch(
    tick,
    /setIsAtBottom\(true\);\s*requestTick\(\);\s*return;/,
    "tick chains unconditionally inside the follow window",
  );
});

test("a quiet pinned frame hands the rest of the window to the settle check", () => {
  const tick = tickBody();
  assert.match(tick, /scheduleSettleCheck\(\);/);
  // Growth that reaches neither observer (image decode, font-display: swap, a late KaTeX pass)
  // is why the window exists at all, so it must still be followed.
  assert.match(HOOK, /const SETTLE_CHECK_MS = 100;/);
  assert.match(
    HOOK,
    /settleTimer = window\.setTimeout\(/,
    "the settle check must be a timer, not another frame",
  );
  assert.match(HOOK, /Math\.min\(SETTLE_CHECK_MS, remaining\)/);
});

test("the settle check grants its frame one last follow pass", () => {
  const tick = tickBody();
  // Without settleCheckDue the timer's frame lands after the window closed and falls into the
  // settle branch, so the growth it was scheduled for goes unfollowed.
  assert.match(tick, /const settling = settleCheckDue;/);
  assert.match(tick, /settleCheckDue = false;/);
  assert.match(tick, /\(settling \|\| performance\.now\(\) < followUntilRef\.current\)/);
});

test("detaching cancels a queued settle check", () => {
  const detach = HOOK.slice(
    HOOK.indexOf("const detach = (): void => {"),
    HOOK.indexOf("const requestTick = (): void => {"),
  );
  // `following` checks userDetached before `settling`, so a queued check cannot re-pin a detached
  // viewport; cancelling it just avoids leaving a timer armed with nothing to do.
  assert.match(detach, /clearSettleCheck\(\);/);
  assert.match(HOOK, /const clearSettleCheck = \(\): void => \{/);
  assert.match(HOOK, /settleCheckDue = false;\s*\};/);
});

test("teardown clears the settle timer", () => {
  const cleanup = HOOK.slice(HOOK.indexOf("cancelAnimationFrame(rafId);"));
  assert.match(cleanup, /clearSettleCheck\(\);/);
  assert.match(cleanup, /resizeObserver\.disconnect\(\);/);
});

test("every layout signal marks layout as moving", () => {
  const onLayoutChange = HOOK.slice(
    HOOK.indexOf("const onLayoutChange = (): void => {"),
  ).slice(0, 400);
  // The single fan-in for ResizeObserver, MutationObserver and visualViewport.resize. Set the
  // flag anywhere narrower and a mutation can arrive on a frame that declines to chase it.
  assert.match(onLayoutChange, /layoutChanged = true;/);
  assert.match(onLayoutChange, /extendFollow\(\);/);
});
