// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The hook is .tsx, which node's type stripping cannot import, so its shape is pinned from source.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const HOOK = readSrc("components/assistant-ui/use-intent-aware-autoscroll.tsx");

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
  assert.doesNotMatch(
    tick,
    /setIsAtBottom\(true\);\s*requestTick\(\);\s*return;/,
    "tick chains unconditionally inside the follow window",
  );
});

test("a quiet pinned frame hands the rest of the window to the settle check", () => {
  const tick = tickBody();
  assert.match(tick, /scheduleSettleCheck\(\);/);
  // Growth that reaches neither observer (image decode, font swap, late KaTeX) must still be followed.
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
  // Without settleCheckDue the timer's frame lands after the window closed and goes unfollowed.
  assert.match(tick, /const settling = settleCheckDue;/);
  assert.match(tick, /settleCheckDue = false;/);
  assert.match(tick, /\(settling \|\| performance\.now\(\) < followUntilRef\.current\)/);
});

test("detaching cancels a queued settle check", () => {
  const detach = HOOK.slice(
    HOOK.indexOf("const detach = (): void => {"),
    HOOK.indexOf("const requestTick = (): void => {"),
  );
  // `following` checks userDetached first, so a queued check cannot re-pin a detached viewport.
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
  // Single fan-in for all observers; setting the flag anywhere narrower drops mutations.
  assert.match(onLayoutChange, /layoutChanged = true;/);
  assert.match(onLayoutChange, /extendFollow\(\);/);
});
