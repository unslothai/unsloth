// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The first commit mounts a probe tail and widens it before the first paint until it fills the
// viewport. The painted frame must show the same rows as the INITIAL_MESSAGES window would.

import assert from "node:assert/strict";
import test from "node:test";

import {
  FILL_RUNWAY_VIEWPORTS,
  type FillSample,
  INITIAL_MESSAGES,
  MAX_INITIAL_SPAN,
  MIN_PROGRESSIVE_MESSAGES,
  type MountWindow,
  PROBE_MESSAGES,
  fillWindow,
  initialWindow,
  widen,
} from "../src/components/assistant-ui/progressive-mount-controller.ts";
import { readText } from "./helpers/kit.ts";

const GLUE = readText("../src/components/assistant-ui/progressive-messages.tsx").replace(/\/\*[\s\S]*?\*\//g, "").replace(/^[ \t]*\/\/.*$/gm, "");

test("the probe is smaller than the window it replaces, and the cap is today's window", () => {
  assert.ok(PROBE_MESSAGES >= 2 && PROBE_MESSAGES < INITIAL_MESSAGES);
  assert.ok(PROBE_MESSAGES % 2 === 0, "a user turn and its reply arrive together");
  assert.ok(FILL_RUNWAY_VIEWPORTS >= 0);
  for (const count of [MIN_PROGRESSIVE_MESSAGES, 80, 200, 5000]) {
    const capped = initialWindow(count, false);
    const probed = initialWindow(count, false, undefined, PROBE_MESSAGES);
    assert.deepEqual(probed, { start: count - PROBE_MESSAGES, fillTo: capped?.start });
    const renders = () => true;
    assert.deepEqual(
      initialWindow(count, false, renders, PROBE_MESSAGES),
      { start: count - PROBE_MESSAGES, fillTo: initialWindow(count, false, renders)?.start },
    );
  }
});

test("the probe never changes which threads are windowed", () => {
  for (const count of [0, 1, MIN_PROGRESSIVE_MESSAGES - 1]) {
    assert.equal(initialWindow(count, false, undefined, PROBE_MESSAGES), null);
  }
  assert.equal(initialWindow(220, true, undefined, PROBE_MESSAGES), null);
  assert.equal(initialWindow(220, true, () => true, PROBE_MESSAGES), null);
  // Too few renderable rows inside the span: the cap, not a probe, exactly as without one.
  assert.equal(initialWindow(MAX_INITIAL_SPAN, false, () => false, PROBE_MESSAGES), null);
  assert.deepEqual(initialWindow(220, false, () => false, PROBE_MESSAGES), {
    start: 220 - MAX_INITIAL_SPAN,
  });
});

test("the probe counts rows that render, and stays inside MAX_INITIAL_SPAN", () => {
  const count = 220;
  // Every third message renders nothing.
  const renders = (index: number) => index % 3 !== 0;
  const w = initialWindow(count, false, renders, PROBE_MESSAGES);
  const cap = initialWindow(count, false, renders);
  assert.ok(w != null && cap != null && w.fillTo === cap.start);
  let visible = 0;
  for (let index = w.start; index < count; index += 1) if (renders(index)) visible += 1;
  assert.equal(visible, PROBE_MESSAGES);
  assert.ok(renders(w.start), "the probe starts at a row that renders");
  // Probe rows only reachable past the span: the probe is the cap itself.
  const sparse = (index: number) => index < count - MAX_INITIAL_SPAN;
  assert.deepEqual(initialWindow(count, false, sparse, PROBE_MESSAGES), {
    start: count - MAX_INITIAL_SPAN,
  });
});

test("a probe size at or over INITIAL_MESSAGES is today's window", () => {
  assert.deepEqual(initialWindow(220, false, undefined, INITIAL_MESSAGES), { start: 220 - INITIAL_MESSAGES });
  assert.deepEqual(initialWindow(220, false, undefined, 0), { start: 220 - INITIAL_MESSAGES });
});

const filledSample = (clientHeight: number): FillSample => ({
  runwayPx: clientHeight * FILL_RUNWAY_VIEWPORTS,
  clientHeight,
  rowsPx: clientHeight * (1 + FILL_RUNWAY_VIEWPORTS),
});

test("a probe that fills is returned as is, so the layout effect commits nothing", () => {
  const w = initialWindow(220, false, undefined, PROBE_MESSAGES);
  assert.equal(fillWindow(w, 220, filledSample(900)), w);
  assert.equal(fillWindow(w, 220, { ...filledSample(900), runwayPx: 5000 }), w);
});

test("a window without fillTo is never touched by fillWindow", () => {
  const plain: MountWindow = { start: 204 };
  assert.equal(fillWindow(plain, 220, null), plain);
  assert.equal(fillWindow(null, 220, null), null);
});

test("unknown geometry falls back to today's window instead of painting a short tail", () => {
  const w = initialWindow(220, false, undefined, PROBE_MESSAGES);
  const cap = { start: 220 - INITIAL_MESSAGES };
  for (const sample of [
    null,
    { runwayPx: -500, clientHeight: 0, rowsPx: 400 },
    { runwayPx: -500, clientHeight: 900, rowsPx: 0 },
    { runwayPx: Number.NaN, clientHeight: 900, rowsPx: 400 },
    { runwayPx: -500, clientHeight: 900, rowsPx: Number.POSITIVE_INFINITY },
  ]) {
    assert.deepEqual(fillWindow(w, 220, sample), cap);
  }
});

test("each fill step only widens, by an even step, and stops at fillTo without it", () => {
  const count = 220;
  let w = initialWindow(count, false, undefined, PROBE_MESSAGES);
  const fillTo = w?.fillTo;
  // 4 rows of 20 px against a 1890 px viewport: the estimate overshoots fillTo, so it lands there.
  w = fillWindow(w, count, { runwayPx: -1810, clientHeight: 1890, rowsPx: 80 });
  assert.deepEqual(w, { start: fillTo });
  // A small shortfall adds one pair.
  const probe = initialWindow(count, false, undefined, PROBE_MESSAGES);
  const step = fillWindow(probe, count, { runwayPx: 850, clientHeight: 900, rowsPx: 1750 });
  assert.ok(step != null && probe != null);
  assert.equal(probe.start - step.start, 2);
  assert.equal(step.fillTo, fillTo);
});

/** Bottom-pinned column of row heights; returns the rows a reader sees and their offsets. */
function paint(heights: number[], start: number, clientHeight: number, slack: number) {
  const mounted = heights.slice(start);
  const total = mounted.reduce((a, b) => a + b, 0) + slack;
  const scrollTop = Math.max(0, total - clientHeight);
  const visible: Array<[number, number]> = [];
  let top = 0;
  for (let i = 0; i < mounted.length; i += 1) {
    const offset = top - scrollTop;
    if (offset + mounted[i] > 0 && offset < clientHeight) visible.push([start + i, offset]);
    top += mounted[i];
  }
  return { visible, runwayPx: scrollTop - 0, rowsPx: total - slack };
}

/** The glue's loop: measure, commit fillWindow until it returns the same window. */
function settleBeforePaint(heights: number[], clientHeight: number, slack: number) {
  const count = heights.length;
  let w = initialWindow(count, false, undefined, PROBE_MESSAGES);
  let commits = 1;
  for (;;) {
    const start = w == null ? 0 : w.start;
    const { runwayPx, rowsPx } = paint(heights, start, clientHeight, slack);
    const next = fillWindow(w, count, { runwayPx, clientHeight, rowsPx });
    if (next === w) break;
    assert.ok(next != null && w != null && next.start < w.start, "fill must only widen");
    w = next;
    commits += 1;
    if (w?.fillTo == null) break;
  }
  return { w, commits };
}

test("the painted rows and offsets match the INITIAL_MESSAGES window for every row shape", () => {
  const shapes: Record<string, (i: number) => number> = {
    agent: () => 1400,
    pairs: (i) => (i % 2 === 0 ? 90 : 1200),
    oneWord: () => 56,
    shortestRows: () => 1890 / INITIAL_MESSAGES,
    hugeLast: (i) => (i === 199 ? 40_000 : 700),
    mixed: (i) => [56, 2400, 120, 56, 900, 3000][i % 6],
  };
  for (const clientHeight of [480, 900, 1890]) {
    for (const [name, height] of Object.entries(shapes)) {
      const heights = Array.from({ length: 200 }, (_, i) => height(i));
      const slack = 200;
      const today = paint(heights, 200 - INITIAL_MESSAGES, clientHeight, slack).visible;
      const { w, commits } = settleBeforePaint(heights, clientHeight, slack);
      const start = w == null ? 0 : w.start;
      assert.ok(start >= 200 - INITIAL_MESSAGES, `${name}: mounted more than today`);
      const painted = paint(heights, start, clientHeight, slack);
      // Rows today's window fills the viewport with are the rows painted now, at the same offsets.
      const todayFills = paint(heights, 200 - INITIAL_MESSAGES, clientHeight, slack).runwayPx > 0;
      if (todayFills) assert.deepEqual(painted.visible, today, `${name} at ${clientHeight}px`);
      assert.ok(commits <= 4, `${name} at ${clientHeight}px took ${commits} commits`);
      if (start > 200 - INITIAL_MESSAGES) {
        assert.ok(
          painted.runwayPx >= clientHeight * FILL_RUNWAY_VIEWPORTS,
          `${name} at ${clientHeight}px stopped short of the runway`,
        );
      }
    }
  }
});

test("background widening continues from a fill window and drops fillTo", () => {
  const w = initialWindow(220, false, undefined, PROBE_MESSAGES);
  const next = widen(w, 220);
  assert.deepEqual(next, { start: 220 - PROBE_MESSAGES - 32 });
  let current: MountWindow = next;
  let steps = 0;
  while (current != null) {
    const after = widen(current, 220);
    assert.ok(after == null || after.start < current.start);
    current = after;
    steps += 1;
    assert.ok(steps < 20);
  }
});

test("the glue widens the probe in the first commit's layout effect, before the bottom pin", () => {
  assert.equal(
    (GLUE.match(/initialWindow\(count, isRunningNow\(\), renderableAt\(\), PROBE_MESSAGES\)/g) ?? []).length,
    3,
    "every place that opens a window must open a probe",
  );
  const initial = GLUE.slice(GLUE.indexOf("if (!captured) {"), GLUE.indexOf("const [element, baseline] = captured"));
  assert.match(initial, /if \(mountWindow\?\.fillTo != null\) \{/);
  assert.match(initial, /const filled = fillWindow\(mountWindow, count, sampleFill\(viewport\)\);/);
  assert.match(initial, /if \(filled !== mountWindow\) \{\s*setMountWindow\(filled\);\s*return;/);
  assert.ok(
    initial.indexOf("fillWindow(") < initial.indexOf("adjustForContentInsertedAbove(0)"),
    "the pin runs on the commit that is painted",
  );
  // Inside a layout effect, so the widened commit is synchronous and the probe is never painted.
  const effect = GLUE.slice(GLUE.lastIndexOf("useLayoutEffect(() => {", GLUE.indexOf("if (!captured) {")), GLUE.indexOf("if (!captured) {"));
  assert.match(effect, /const captured = anchorRef\.current;/);
  assert.doesNotMatch(GLUE, /requestAnimationFrame\([^)]*fillWindow/);
});
