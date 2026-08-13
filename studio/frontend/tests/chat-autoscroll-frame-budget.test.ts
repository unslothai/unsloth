// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The chat viewport followed the bottom by chaining a requestAnimationFrame off nothing but a
// 600ms window that every mutation re-armed, so a streaming message kept a forced layout read
// running every frame for its whole duration. Measured with tests/studio/playwright_chat_autoscroll.py
// on a frame pump, a message arriving every 250ms - the cadence deep research synthesis flushes
// at - held the loop at the pump's ceiling of 62 frames a second; after this change it settles
// to the 100ms settle check, about 12.
//
// The cost is invisible to a rendering test and the hook is a .tsx, which node's type stripping
// cannot import, so the shape is pinned here the way drag-costs-no-render.test.ts pins the drag
// path: assert the cheap path, and assert the expensive one is gone.

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
  // The old path: one requestTick per frame for as long as the window stayed open, which every
  // mutation extended, so a streaming message never let it close.
  assert.doesNotMatch(
    tick,
    /setIsAtBottom\(true\);\s*requestTick\(\);\s*return;/,
    "tick chains unconditionally inside the follow window",
  );
});

test("a quiet pinned frame hands the rest of the window to the settle check", () => {
  const tick = tickBody();
  assert.match(tick, /scheduleSettleCheck\(\);/);
  // Growth that reaches neither observer (an image decoding, a font-display: swap webfont, a
  // late KaTeX pass) is why the window exists at all, so it must still be followed.
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
  // Without settleCheckDue the timer's frame lands after the window has closed and falls
  // straight into the settle branch, so the growth it was scheduled for goes unfollowed.
  assert.match(tick, /const settling = settleCheckDue;/);
  assert.match(tick, /settleCheckDue = false;/);
  assert.match(tick, /\(settling \|\| performance\.now\(\) < followUntilRef\.current\)/);
});

test("detaching cancels a queued settle check", () => {
  const detach = HOOK.slice(
    HOOK.indexOf("const detach = (): void => {"),
    HOOK.indexOf("const requestTick = (): void => {"),
  );
  // `following` is gated on userDetached before `settling` is consulted, so a queued check
  // cannot re-pin a detached viewport. Cancelling it is about not leaving a timer armed for the
  // rest of the follow window with nothing left for it to do.
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
  // The single fan-in for ResizeObserver, MutationObserver and visualViewport.resize. If the
  // flag were set anywhere narrower, a mutation could arrive on a frame that then declined to
  // chase it.
  assert.match(onLayoutChange, /layoutChanged = true;/);
  assert.match(onLayoutChange, /extendFollow\(\);/);
});
