// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The research activity scroller follows a run that mutates several times a second, so its
// per-frame work is what #8483 froze on. Cost and detach behaviour are invisible to a unit test
// (tests/studio/playwright_research_freeze.py measures them in a real browser), so these pin the
// shape the measurements were taken against, as drag-costs-no-render.test.ts does for panels.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

function followLoopSource(): string {
  const text = readFileSync(
    new URL(
      "../src/features/chat/components/research-activity-panel.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const start = text.indexOf("function useResearchActivityScroll");
  assert.ok(start >= 0, "useResearchActivityScroll is gone");
  const end = text.indexOf("\n}", text.indexOf("}, [runId];".replace(";", ");")));
  // Without this the slice widens to the rest of the file and every assertion passes for the
  // wrong reason.
  assert.ok(end > start, "could not find the end of useResearchActivityScroll");
  return text.slice(start, end);
}

test("a follow frame chains while unpinned, plus one after every layout signal", () => {
  const hook = followLoopSource();
  assert.match(
    hook,
    /if \(layoutChanged \|\| !pinned\) \{\s*layoutChanged = false;\s*requestTick\(\);/,
  );
  // The guaranteed frame is not redundant: a click's frame can land before the Collapsible height
  // animation it started has grown at all, so chaining on !pinned alone stops there and leaves the
  // log behind until the settle check. Neither observer sees a height animation to restart it.
  assert.match(hook, /layoutChanged = true;\s*followUntil = performance\.now\(\)/);
});

test("a quiet frame hands the rest of the window to one deferred check", () => {
  const hook = followLoopSource();
  assert.match(hook, /scheduleSettleCheck\(\);\s*return;\s*\}/);
  assert.match(hook, /settleTimer = window\.setTimeout\(/);
});

test("detach cancels every pending follow step", () => {
  const hook = followLoopSource();
  const detach = hook.slice(
    hook.indexOf("const detach = () => {"),
    hook.indexOf("const innerScrollWillConsumeUpward"),
  );
  // Without the frame cancel, a follow step queued before the detach still runs, reconciles
  // isAtBottom through the bottom threshold, and hides "Latest" after a short upward flick.
  assert.match(detach, /cancelAnimationFrame\(animationFrame\)/);
  assert.match(detach, /animationFrame = null/);
  assert.match(detach, /window\.clearTimeout\(settleTimer\)/);
  assert.match(detach, /settleCheckDue = false/);
});

test("the pinned threshold stays at 2px, not 1", () => {
  const text = readFileSync(
    new URL(
      "../src/features/chat/components/research-activity-panel.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // HiDPI subpixel rounding leaves a fractional gap; at 1px the loop never reads as pinned and
  // never exits, the configuration the freeze was reported on.
  assert.match(text, /const ACTIVITY_PINNED_THRESHOLD_PX = 2;/);
});
