// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The window only grows and always reaches null; shrinking would unmount a message.
// The .tsx glue is pinned in progressive-mount-glue.test.ts.

import assert from "node:assert/strict";
import test from "node:test";

import {
  CHUNK_MESSAGES,
  INITIAL_MESSAGES,
  MAX_INITIAL_SPAN,
  MIN_PROGRESSIVE_MESSAGES,
  type MountWindow,
  admits,
  anchorCorrection,
  initialWindow,
  isCovered,
  stepsToCover,
  widen,
} from "../src/components/assistant-ui/progressive-mount-controller.ts";

test("a thread under the floor is never windowed", () => {
  for (const count of [0, 1, 20, MIN_PROGRESSIVE_MESSAGES - 1]) {
    assert.equal(
      initialWindow(count, false),
      null,
      `${count} messages should mount at once`,
    );
  }
});

test("a thread at the floor is windowed", () => {
  const w = initialWindow(MIN_PROGRESSIVE_MESSAGES, false);
  assert.notEqual(w, null);
  assert.equal(w?.start, MIN_PROGRESSIVE_MESSAGES - INITIAL_MESSAGES);
});

test("the first commit mounts the tail, not the head", () => {
  // The thread is bottom-anchored, so a head-anchored window paints an empty viewport.
  const w = initialWindow(220, false);
  assert.equal(w?.start, 220 - INITIAL_MESSAGES);
  assert.equal(admits(w, 219), true);
  assert.equal(admits(w, 220 - INITIAL_MESSAGES), true);
  assert.equal(admits(w, 220 - INITIAL_MESSAGES - 1), false);
  assert.equal(admits(w, 0), false);
});

test("a running thread opens unrestricted", () => {
  // Widening and streaming both write scrollTop, and a reply must never commit into a tree that
  // has not reached it.
  assert.equal(initialWindow(220, true), null);
});

test("a null window admits every index", () => {
  assert.equal(admits(null, 0), true);
  assert.equal(admits(null, 10_000), true);
  assert.equal(isCovered(null), true);
});

test("widening only ever moves start down", () => {
  let current = initialWindow(220, false);
  let previous = Number.POSITIVE_INFINITY;
  while (current != null) {
    assert.ok(
      current.start < previous,
      `start went ${previous} -> ${current.start}`,
    );
    previous = current.start;
    current = widen(current, 220);
  }
});

test("widening reaches null from every thread length", () => {
  for (const count of [40, 41, 80, 220, 501, 5000]) {
    assert.ok(stepsToCover(count) >= 1, `${count} never widened`);
  }
});

test("the number of widening frames is bounded and small", () => {
  // At 60Hz seven frames is under 120ms, the budget this change rests on.
  assert.equal(
    stepsToCover(220),
    Math.ceil((220 - INITIAL_MESSAGES) / CHUNK_MESSAGES),
  );
  assert.equal(stepsToCover(220), 7);
  assert.ok(
    stepsToCover(5000) < 160,
    "a 5000-message thread must still converge in seconds",
  );
});

test("widen returns null in the same commit that mounts the last chunk", () => {
  // {start: 0} renders the same as null, so leaving it wastes a full re-render frame.
  const nearlyDone: MountWindow = { start: CHUNK_MESSAGES };
  assert.equal(widen(nearlyDone, 220), null);
  const exactlyOneChunkLeft: MountWindow = { start: CHUNK_MESSAGES - 1 };
  assert.equal(widen(exactlyOneChunkLeft, 220), null);
});

test("widen never returns a start above the message count", () => {
  // A thread that shrank must not leave the window past the end.
  const stale: MountWindow = { start: 200 };
  const next = widen(stale, 10);
  assert.equal(
    next,
    null,
    "a window past the end of a shrunken thread must be dropped",
  );
});

test("widen on a null window is a no-op", () => {
  assert.equal(widen(null, 220), null);
});

test("isCovered is true exactly when nothing is being withheld", () => {
  assert.equal(isCovered({ start: 0 }), true);
  assert.equal(isCovered({ start: 1 }), false);
});

test("the constants stay inside the range they were measured over", () => {
  // Bounds, not equalities, so the first commit cannot quietly become one row or everything.
  assert.ok(
    INITIAL_MESSAGES >= 8 && INITIAL_MESSAGES <= 64,
    `INITIAL_MESSAGES ${INITIAL_MESSAGES} is outside the measured range`,
  );
  assert.ok(
    CHUNK_MESSAGES >= 8 && CHUNK_MESSAGES <= 128,
    `CHUNK_MESSAGES ${CHUNK_MESSAGES} is outside the measured range`,
  );
  assert.ok(
    MIN_PROGRESSIVE_MESSAGES >= INITIAL_MESSAGES * 2,
    "the floor must leave room for at least one widening, or the window is pointless",
  );
});

const sample = (viewportOffset: number, scrollTop: number, maxScrollTop = 1_000_000) => ({
  viewportOffset,
  scrollTop,
  maxScrollTop,
});

test("the correction is the height inserted above, in document space", () => {
  assert.equal(anchorCorrection(sample(-500, 20000), sample(11500, 20000)), 12000);
});

test("the reader's own scroll is subtracted, not skipped", () => {
  // Document space nets reader scroll against insertion above.
  assert.equal(anchorCorrection(sample(-500, 20000), sample(8500, 23000)), 12000);
  assert.equal(anchorCorrection(sample(-500, 20000), sample(-3500, 23000)), null);
});

test("a frame that inserted nothing is a no-op whatever the reader did", () => {
  for (const scrolled of [0, -900, 4000]) {
    assert.equal(
      anchorCorrection(sample(-500, 20000), sample(-500 - scrolled, 20000 + scrolled)),
      null,
      `no insertion should mean no correction (reader moved ${scrolled}px)`,
    );
  }
});

test("sub-pixel movement is never acted on", () => {
  assert.equal(anchorCorrection(sample(-500.0, 20000), sample(-499.4, 20000)), null);
});

test("a window past the end of a shrunken thread is dropped, never narrowed", () => {
  // Clamping then subtracting a chunk would unmount rows the caller already rendered.
  assert.equal(widen({ start: 204 }, 100), null);
  assert.equal(widen({ start: 204 }, 0), null);
  assert.equal(widen({ start: 100 }, 100), null);
  assert.deepEqual(widen({ start: 99 }, 100), { start: 99 - CHUNK_MESSAGES });
});

test("widening is monotone in start for every reachable window and count", () => {
  for (let count = 0; count <= 300; count += 7) {
    for (let start = 0; start <= 300; start += 5) {
      const next = widen({ start }, count);
      if (next == null) continue;
      assert.ok(
        next.start < start,
        `widen({start:${start}}, ${count}) returned {start:${next.start}}, which withholds more`,
      );
    }
  }
});

test("a shrink the browser already clamped is not corrected twice", () => {
  // With anchoring off the browser clamps scrollTop on shrink; apply only the unabsorbed part.
  assert.equal(
    anchorCorrection(sample(200, 9000, 9100), sample(100, 8600, 8600)),
    -100,
  );
  assert.equal(
    anchorCorrection(sample(200, 9000, 50_000), sample(-300, 9000, 49_500)),
    -500,
  );
});

// Roles with no component render null and have no height, so size the tail by renderable rows.

function tailIsUnrenderable(count: number, systemTail: number) {
  return (index: number) => index < count - systemTail;
}

function visibleRows(
  window: MountWindow,
  count: number,
  renders: (index: number) => boolean,
): number {
  let visible = 0;
  for (
    let index = window == null ? 0 : window.start;
    index < count;
    index += 1
  ) {
    if (renders(index)) visible += 1;
  }
  return visible;
}

test("the initial tail is sized on rows that render, not on message count", () => {
  const count = 220;
  const renders = tailIsUnrenderable(count, INITIAL_MESSAGES);
  const w = initialWindow(count, false, renders);
  assert.equal(visibleRows(w, count, renders), INITIAL_MESSAGES);
  assert.ok(w != null && w.start < count - INITIAL_MESSAGES);
});

test("a tail of unrenderable rows still only ever widens", () => {
  const count = 220;
  const renders = tailIsUnrenderable(count, 40);
  const w = initialWindow(count, false, renders);
  assert.ok(w != null && w.start >= 0 && w.start < count);
  assert.equal(visibleRows(w, count, renders), INITIAL_MESSAGES);
  let current: MountWindow = w;
  let steps = 0;
  while (current != null) {
    const next = widen(current, count);
    assert.ok(
      next == null || next.start < current.start,
      "start must only fall",
    );
    current = next;
    steps += 1;
    assert.ok(steps <= count + 2, "widen made no progress");
  }
});

test("a short thread with too few renderable rows is not windowed at all", () => {
  const count = MAX_INITIAL_SPAN;
  assert.equal(
    initialWindow(count, false, () => false),
    null,
  );
  assert.equal(
    initialWindow(
      count,
      false,
      (index) => index >= count - (INITIAL_MESSAGES - 1),
    ),
    null,
  );
});

test("a long thread with too few renderable rows is bounded, not mounted whole", () => {
  const count = 220;
  const shapes = [
    () => false,
    (index: number) => index >= count - (INITIAL_MESSAGES - 1),
  ];
  for (const renders of shapes) {
    const w = initialWindow(count, false, renders);
    assert.notEqual(w, null);
    assert.equal(w?.start, count - MAX_INITIAL_SPAN);
  }
});

test("omitting the predicate keeps the count-based window", () => {
  assert.deepEqual(initialWindow(220, false), {
    start: 220 - INITIAL_MESSAGES,
  });
  assert.deepEqual(
    initialWindow(220, false, () => true),
    {
      start: 220 - INITIAL_MESSAGES,
    },
  );
});

// The renderable-row walk must stay bounded, or a long non-rendering tail mounts everything.

test("the first commit's raw span is bounded even when the tail does not render", () => {
  const count = 220;
  const renders = (index: number) => index < INITIAL_MESSAGES;
  const w = initialWindow(count, false, renders);
  const first = w == null ? 0 : w.start;
  assert.ok(
    count - first <= MAX_INITIAL_SPAN,
    `first commit mounted ${count - first} providers, over the ${MAX_INITIAL_SPAN} cap`,
  );
});

test("the cap does not cost visible rows when they are reachable inside it", () => {
  const count = 220;
  const renders = (index: number) => index % 3 !== 0;
  const w = initialWindow(count, false, renders);
  assert.notEqual(w, null);
  const first = w == null ? 0 : w.start;
  assert.ok(count - first <= MAX_INITIAL_SPAN);
  assert.equal(visibleRows(w, count, renders), INITIAL_MESSAGES);
});

test("an all-renderable thread is unchanged by the cap", () => {
  assert.deepEqual(initialWindow(220, false, () => true), {
    start: 220 - INITIAL_MESSAGES,
  });
});

test("a capped window still only widens, and still converges", () => {
  const count = 220;
  const renders = (index: number) => index < INITIAL_MESSAGES;
  let current = initialWindow(count, false, renders);
  assert.ok(current != null && current.start >= 0 && current.start < count);
  let steps = 0;
  while (current != null) {
    const next = widen(current, count);
    assert.ok(next == null || next.start < current.start, "start must only fall");
    current = next;
    steps += 1;
    assert.ok(steps <= count + 2, "widen made no progress");
  }
});
