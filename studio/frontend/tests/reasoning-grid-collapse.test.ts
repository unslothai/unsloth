// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Source-pinned: node type stripping cannot compile JSX.
// The shared collapsible keeps its keyframes: app-sidebar.tsx keys its scroll fade off their names.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

const UNMEASURED = readText("../src/components/ui/unmeasured-collapsible.tsx");
const REASONING = readText("../src/components/assistant-ui/reasoning.tsx");
const FLAGS = readText("../src/components/assistant-ui/thread-feature-flags.ts");
const SHARED_COLLAPSIBLE = readText("../src/components/ui/collapsible.tsx");
const APP_SIDEBAR = readText("../src/components/app-sidebar.tsx");
const TOOL_GROUP = readText("../src/components/assistant-ui/tool-group.tsx");
const TOOL_FALLBACK = readText("../src/components/assistant-ui/tool-fallback.tsx");

// Comments here discuss measurement, so only code lines are asserted on.
function codeOf(source: string): string {
  return source
    .split("\n")
    .filter((line) => {
      const trimmed = line.trim();
      return (
        trimmed.length > 0 &&
        !trimmed.startsWith("//") &&
        !trimmed.startsWith("*") &&
        !trimmed.startsWith("/*")
      );
    })
    .join("\n");
}

test("the unmeasured collapsible reads no geometry at all", () => {
  const code = codeOf(UNMEASURED);
  for (const api of [
    "getBoundingClientRect",
    "getComputedStyle",
    "offsetHeight",
    "offsetWidth",
    "clientHeight",
    "clientWidth",
    "scrollHeight",
    "scrollWidth",
    "getClientRects",
    "ResizeObserver",
  ]) {
    assert.ok(
      !code.includes(api),
      `unmeasured-collapsible.tsx must not use ${api}; that is the whole point of it`,
    );
  }
});

test("the animating child carries min-height:0 and overflow:hidden", () => {
  assert.match(UNMEASURED, /className="min-h-0 overflow-hidden"/);
});

test("the collapse is a grid-template-rows transition between 0fr and 1fr", () => {
  const code = codeOf(UNMEASURED);
  assert.ok(code.includes("transition-[grid-template-rows]"));
  assert.ok(code.includes('"grid-rows-[1fr]"'));
  assert.ok(code.includes('"grid-rows-[0fr]"'));
  // UA `[hidden] { display: none }` loses to author-level `display: grid`.
  assert.ok(code.includes('present ? "grid" : "hidden"'));
});

test("the unmeasured trigger keeps the accessible collapsible contract", () => {
  const code = codeOf(UNMEASURED);
  assert.ok(code.includes("aria-expanded={context.open || false}"));
  assert.ok(code.includes("aria-controls={context.contentId}"));
  assert.ok(code.includes('type="button"'));
  // Content must carry the id the trigger points at, or aria-controls dangles.
  assert.ok(code.includes("id={context.contentId}"));
  // Every consumer's data-[state=...] classes key off data-state on root, trigger and content.
  assert.equal(code.match(/data-state=\{getState\(/g)?.length, 3);
});

test("children unmount while closed, exactly as Radix's presence does", () => {
  const code = codeOf(UNMEASURED);
  assert.ok(code.includes("{present && children}"));
  assert.ok(code.includes("setMounted(false)"));
});

test("the close path unmounts on transitionend for the right property, with a timeout backstop", () => {
  const code = codeOf(UNMEASURED);
  // transitionend bubbles and fires per property, so an unfiltered handler unmounts early.
  assert.ok(
    code.includes('event.target === node && event.propertyName === "grid-template-rows"'),
  );
  // The backstop starts before the transition, so it must exceed closeDurationMs.
  assert.ok(
    code.includes("window.setTimeout(finish, closeDurationMs + CLOSE_FALLBACK_MARGIN_MS)"),
  );
  assert.match(code, /const CLOSE_FALLBACK_MARGIN_MS = \d+;/);
});

test("nothing writes a ref during render", () => {
  const code = codeOf(UNMEASURED);
  // React does not roll back refs on abandoned renders, so the toggle closes over `open`.
  assert.ok(!code.includes("openRef"));
  assert.ok(code.includes("const next = !open;"));
});

test("the flag is on", () => {
  // Kept so the flag value stays a deliberate, reviewed choice.
  assert.match(FLAGS, /export const GRID_COLLAPSE_REASONING_ENABLED = true;/);
});

test("the reasoning pane picks its primitive from the flag on all three slots", () => {
  const code = codeOf(REASONING);
  assert.equal(code.match(/GRID_COLLAPSE_REASONING_ENABLED/g)?.length, 5);
  assert.ok(code.includes("<UnmeasuredCollapsible {...rootProps}>"));
  assert.ok(code.includes("<Collapsible {...rootProps}>"));
  assert.ok(code.includes("UnmeasuredCollapsibleTrigger"));
  assert.ok(code.includes("<UnmeasuredCollapsibleContent"));
});

test("the scroll lock outlasts the grid collapse, and the shared hook's other callers do not move", () => {
  const code = codeOf(REASONING);
  // The lock arms a commit before the transition starts, so an exact duration releases mid-collapse.
  assert.match(
    code,
    /useCollapseScrollLock\(\s*collapsibleRef,\s*GRID_COLLAPSE_REASONING_ENABLED/,
  );
  for (const other of [TOOL_GROUP, TOOL_FALLBACK]) {
    assert.ok(
      codeOf(other).includes("useCollapseScrollLock(collapsibleRef, ANIMATION_DURATION)"),
    );
  }
});

test("the flag-on reasoning content runs no height keyframes", () => {
  const gridBranch = REASONING.slice(
    REASONING.indexOf("if (GRID_COLLAPSE_REASONING_ENABLED) {"),
    REASONING.indexOf("</UnmeasuredCollapsibleContent>"),
  );
  assert.ok(gridBranch.length > 0);
  assert.ok(!gridBranch.includes("animate-collapsible-up"));
  assert.ok(!gridBranch.includes("animate-collapsible-down"));
  assert.ok(REASONING.includes('"data-[state=closed]:animate-collapsible-up"'));
  assert.ok(REASONING.includes('"data-[state=open]:animate-collapsible-down"'));
});

test("the height keyframes stay in use everywhere else, because the sidebar listens for them", () => {
  assert.ok(SHARED_COLLAPSIBLE.includes("animate-collapsible-down"));
  assert.ok(SHARED_COLLAPSIBLE.includes("animate-collapsible-up"));
  assert.ok(TOOL_GROUP.includes("animate-collapsible-down"));
  assert.ok(TOOL_FALLBACK.includes("animate-collapsible-down"));
  assert.ok(APP_SIDEBAR.includes('e.animationName === "collapsible-down"'));
  assert.ok(APP_SIDEBAR.includes('e.animationName === "collapsible-up"'));
});

test("reduced motion is reached by the transition, not bypassed by it", () => {
  const indexCss = readText("../src/index.css");
  // Both reduced-motion blankets force transition-duration too, covering the grid collapse.
  assert.ok(indexCss.includes("html.force-reduced-motion *"));
  assert.ok(indexCss.includes("@media (prefers-reduced-motion: reduce)"));
  assert.equal(
    indexCss.match(/transition-duration: 0\.01ms !important;/g)?.length,
    2,
    "both reduced-motion blankets must force transition-duration, not only animation-duration",
  );
});
