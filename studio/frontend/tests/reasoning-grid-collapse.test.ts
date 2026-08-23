// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Source-pinned, following the convention the other .tsx tests here use: node's type stripping
// cannot compile JSX, so a component in a .tsx file is pinned by reading it.
//
// Everything asserted below is something whose absence produces a pane that LOOKS fine in a
// screenshot and is wrong:
//
//   * a measurement anywhere in the unmeasured primitive puts the forced layout straight back,
//     and the change then costs a wrapper element and buys nothing;
//   * `min-height: 0` or `overflow: hidden` missing from the animating child means `0fr` never
//     reaches zero and the pane does not close;
//   * the height keyframes surviving on the flag-on path means both mechanisms run at once;
//   * the keyframes DISAPPEARING from the shared collapsible breaks `app-sidebar.tsx`, which keys
//     its scroll fade off `animationName === "collapsible-down" | "collapsible-up"` and would go
//     silently stale.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) =>
  readFileSync(new URL(path, import.meta.url), "utf8");

const UNMEASURED = read("../src/components/ui/unmeasured-collapsible.tsx");
const REASONING = read("../src/components/assistant-ui/reasoning.tsx");
const FLAGS = read("../src/components/assistant-ui/thread-feature-flags.ts");
const SHARED_COLLAPSIBLE = read("../src/components/ui/collapsible.tsx");
const APP_SIDEBAR = read("../src/components/app-sidebar.tsx");
const TOOL_GROUP = read("../src/components/assistant-ui/tool-group.tsx");
const TOOL_FALLBACK = read("../src/components/assistant-ui/tool-fallback.tsx");

// Comments in these files discuss measurement at length, so an assertion on the raw text would
// pass or fail on prose. Only code lines are considered.
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
  // Both on one element, and that element is the wrapper the component renders itself, so a
  // caller cannot omit them.
  assert.match(UNMEASURED, /className="min-h-0 overflow-hidden"/);
});

test("the collapse is a grid-template-rows transition between 0fr and 1fr", () => {
  const code = codeOf(UNMEASURED);
  assert.ok(code.includes("transition-[grid-template-rows]"));
  assert.ok(code.includes('"grid-rows-[1fr]"'));
  assert.ok(code.includes('"grid-rows-[0fr]"'));
  // The closed state must switch the display utility, not lean on the `hidden` attribute: the UA
  // sheet's `[hidden] { display: none }` loses to the author-level `display: grid`.
  assert.ok(code.includes('present ? "grid" : "hidden"'));
});

test("the unmeasured trigger keeps the accessible collapsible contract", () => {
  const code = codeOf(UNMEASURED);
  assert.ok(code.includes("aria-expanded={context.open || false}"));
  assert.ok(code.includes("aria-controls={context.contentId}"));
  assert.ok(code.includes('type="button"'));
  // Content must carry the id the trigger points at, or aria-controls dangles.
  assert.ok(code.includes("id={context.contentId}"));
  // data-state is what every consumer's `data-[state=...]` and `group-data-[state=...]` class
  // keys off, on the root, the trigger and the content alike.
  assert.equal(code.match(/data-state=\{getState\(/g)?.length, 3);
});

test("children unmount while closed, exactly as Radix's presence does", () => {
  const code = codeOf(UNMEASURED);
  assert.ok(code.includes("{present && children}"));
  // Keeping a closed pane's content mounted would grow the resting DOM of a long thread, which is
  // the opposite of what this change is for.
  assert.ok(code.includes("setMounted(false)"));
});

test("the close path unmounts on transitionend for the right property, with a timeout backstop", () => {
  const code = codeOf(UNMEASURED);
  // transitionend bubbles from descendants and fires once per property; an unfiltered handler
  // would unmount the content the first time anything inside it finished any transition.
  assert.ok(
    code.includes('event.target === node && event.propertyName === "grid-template-rows"'),
  );
  // The backstop must be armed BEYOND the transition duration. It starts counting in the same
  // passive-effect flush that queues `setExpanded(false)`, which is before the browser starts the
  // transition, so arming it at exactly `closeDurationMs` makes it always beat `transitionend` and
  // cut the close short.
  assert.ok(
    code.includes("window.setTimeout(finish, closeDurationMs + CLOSE_FALLBACK_MARGIN_MS)"),
  );
  assert.match(code, /const CLOSE_FALLBACK_MARGIN_MS = \d+;/);
});

test("nothing writes a ref during render", () => {
  const code = codeOf(UNMEASURED);
  // React does not roll a ref back when a render is abandoned or suspended, so a ref
  // assigned in the render body can hold a value that was never committed while the DOM
  // still shows the old one. The toggle closes over `open` instead. (`nodeRef` is written
  // inside the ref callback, which runs on commit, not during render.)
  assert.ok(!code.includes("openRef"));
  assert.ok(code.includes("const next = !open;"));
});

test("the flag is on", () => {
  // Was "off by default" while the A/B was outstanding. It has run: two independent waves at the
  // 100K rung, each with its own in-band null control, and `reasoning_toggle.open_ms` cleared all
  // three gates in both. The assertion is kept rather than deleted so that the flag's value stays
  // a deliberate, reviewed choice instead of something that can drift silently in either
  // direction.
  assert.match(FLAGS, /export const GRID_COLLAPSE_REASONING_ENABLED = true;/);
});

test("the reasoning pane picks its primitive from the flag on all three slots", () => {
  const code = codeOf(REASONING);
  // Three primitive slots, plus the streaming-height release and the scroll lock, which both
  // have to outlast the transition rather than the nominal duration.
  assert.equal(code.match(/GRID_COLLAPSE_REASONING_ENABLED/g)?.length, 6);
  assert.ok(code.includes("<UnmeasuredCollapsible {...rootProps}>"));
  assert.ok(code.includes("<Collapsible {...rootProps}>"));
  assert.ok(code.includes("UnmeasuredCollapsibleTrigger"));
  assert.ok(code.includes("<UnmeasuredCollapsibleContent"));
});

test("the streaming height cap outlives the grid collapse, but only on the grid path", () => {
  const code = codeOf(REASONING);
  // `1fr` resolves against live content every frame, so releasing `max-h-64` before the row has
  // finished shrinking grows it mid-collapse -- the jump the retention timer exists to prevent.
  // The height keyframes animate a height captured at toggle time, so the default path is immune
  // and must keep its exact ANIMATION_DURATION.
  assert.ok(
    code.includes("ANIMATION_DURATION + CLOSE_FALLBACK_MARGIN_MS"),
  );
  assert.ok(code.includes("const closeDelay = GRID_COLLAPSE_REASONING_ENABLED"));
});

test("the scroll lock outlasts the grid collapse, and the shared hook's other callers do not move", () => {
  const code = codeOf(REASONING);
  // The lock is armed in the click handler, a commit before the transition starts, so an exact
  // ANIMATION_DURATION releases the container mid-collapse.
  assert.match(
    code,
    /useCollapseScrollLock\(\s*collapsibleRef,\s*GRID_COLLAPSE_REASONING_ENABLED/,
  );
  // tool-group and tool-fallback share the hook but keep the height keyframes, so they must
  // still pass the plain duration.
  for (const other of [TOOL_GROUP, TOOL_FALLBACK]) {
    assert.ok(
      codeOf(other).includes("useCollapseScrollLock(collapsibleRef, ANIMATION_DURATION)"),
    );
  }
});

test("the flag-on reasoning content runs no height keyframes", () => {
  // Split at the flag branch: everything before the `return` of the flag-off path is the grid
  // path, and the height keyframes must appear only after it.
  const gridBranch = REASONING.slice(
    REASONING.indexOf("if (GRID_COLLAPSE_REASONING_ENABLED) {"),
    REASONING.indexOf("</UnmeasuredCollapsibleContent>"),
  );
  assert.ok(gridBranch.length > 0);
  assert.ok(!gridBranch.includes("animate-collapsible-up"));
  assert.ok(!gridBranch.includes("animate-collapsible-down"));
  // and the flag-off path is untouched, so flag-off is byte-identical behaviour.
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
  const indexCss = read("../src/index.css");
  // Two blankets, the OS media query and the in-app override class. Both force
  // transition-duration as well as animation-duration, which is what makes a transition-based
  // collapse honour reduced motion without any new rule.
  assert.ok(indexCss.includes("html.force-reduced-motion *"));
  assert.ok(indexCss.includes("@media (prefers-reduced-motion: reduce)"));
  assert.equal(
    indexCss.match(/transition-duration: 0\.01ms !important;/g)?.length,
    2,
    "both reduced-motion blankets must force transition-duration, not only animation-duration",
  );
});
