// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Cost is only measurable in a browser (playwright_research_freeze.py), so the shape is pinned.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const RESEARCH_ACTIVITY_PANEL = readSrc("features/chat/components/research-activity-panel.tsx");

function followLoopSource(): string {
  const text = readSrc("features/chat/components/research-activity-panel.tsx");
  const start = text.indexOf("function useResearchActivityScroll");
  assert.ok(start >= 0, "useResearchActivityScroll is gone");
  const end = text.indexOf("\n}", text.indexOf("}, [runId];".replace(";", ");")));
  assert.ok(end > start, "could not find the end of useResearchActivityScroll");
  return text.slice(start, end);
}

test("a follow frame chains while unpinned, plus one after every layout signal", () => {
  const hook = followLoopSource();
  assert.match(
    hook,
    /if \(layoutChanged \|\| !pinned\) \{\s*layoutChanged = false;\s*requestTick\(\);/,
  );
  // A click's frame can land before the Collapsible animation grows, so the follow frame is needed.
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
  // Without the cancel, a queued follow step hides "Latest" after a short upward flick.
  assert.match(detach, /cancelAnimationFrame\(animationFrame\)/);
  assert.match(detach, /animationFrame = null/);
  assert.match(detach, /window\.clearTimeout\(settleTimer\)/);
  assert.match(detach, /settleCheckDue = false/);
});

test("the pinned threshold stays at 2px, not 1", () => {
  // HiDPI subpixel rounding leaves a fractional gap; at 1px the loop never exits.
  assert.match(RESEARCH_ACTIVITY_PANEL, /const ACTIVITY_PINNED_THRESHOLD_PX = 2;/);
});
