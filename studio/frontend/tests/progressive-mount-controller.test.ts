// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The widen-only mount window, tested where it is pure. The React glue is a .tsx and node's type
// stripping cannot import one, so its invariants are pinned by source assertion in
// progressive-mount-glue.test.ts instead.
//
// The property all of these protect: the window only ever grows, and it always reaches null. A
// window that shrank would unmount a message, the failure mode this change exists to avoid.

import assert from "node:assert/strict";
import test from "node:test";

import {
  CHUNK_MESSAGES,
  INITIAL_MESSAGES,
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
  // The thread is bottom-anchored, so a head-anchored window would mount 16 messages nobody can
  // see and still paint an empty viewport.
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
  // One frame per chunk after the first commit. 220 messages is #9016's largest fixture, and at
  // 60Hz seven frames is under 120ms, the budget this whole change rests on.
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
  // Not a frame later: {start: 0} renders the same as null, so leaving one spends an extra frame
  // re-rendering the whole thread into a byte-identical tree.
  const nearlyDone: MountWindow = { start: CHUNK_MESSAGES };
  assert.equal(widen(nearlyDone, 220), null);
  const exactlyOneChunkLeft: MountWindow = { start: CHUNK_MESSAGES - 1 };
  assert.equal(widen(exactlyOneChunkLeft, 220), null);
});

test("widen never returns a start above the message count", () => {
  // A thread that SHRANK under a live window (a delete or a branch switch) must not leave it
  // pointing past the end, which would admit nothing and paint an empty thread.
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
  // Bounds, not equalities, so a future edit cannot quietly turn the first commit into one row or
  // the floor into every thread and stay green.
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

// anchorCorrection, extracted. It used to live inside the .tsx where nothing could test it, and its
// one bug so far was found by a browser probe rather than a test.

const sample = (viewportOffset: number, scrollTop: number, maxScrollTop = 1_000_000) => ({
  viewportOffset,
  scrollTop,
  maxScrollTop,
});

test("the correction is the height inserted above, in document space", () => {
  // scrollTop did not move, so the anchor moved down by everything inserted above it.
  assert.equal(anchorCorrection(sample(-500, 20000), sample(11500, 20000)), 12000);
});

test("the reader's own scroll is subtracted, not skipped", () => {
  // 12000px inserted above in the frame the reader scrolled down 3000px; document space nets those
  // to the 12000px to apply. The version that skipped such frames left the reader 19,259px out.
  assert.equal(anchorCorrection(sample(-500, 20000), sample(8500, 23000)), 12000);
  // A frame holding only the reader's own scroll is a no-op, with no gesture bookkeeping needed.
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
  // The failure this rules out: {start: 204} against a thread that shrank to 100. Clamping to count
  // then subtracting a chunk gives {start: 68}, and the caller already rendered all 100 rows, so
  // rows 0 to 67 would be unmounted.
  assert.equal(widen({ start: 204 }, 100), null);
  assert.equal(widen({ start: 204 }, 0), null);
  assert.equal(widen({ start: 100 }, 100), null);
  // One below the end still widens normally.
  assert.deepEqual(widen({ start: 99 }, 100), { start: 99 - CHUNK_MESSAGES });
});

test("widening is monotone in start for every reachable window and count", () => {
  // The general property behind the case above: the next window never withholds more than this
  // one does.
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
  // With anchoring off the browser moves scrollTop for exactly one reason: content shrinks above a
  // reader near the bottom, their offset stops existing, and it clamps to the new maximum. The
  // document-space delta reports the whole shrink anyway, so applying it on top of the clamp would
  // move the viewport twice by the clamped part.
  //
  // 500px removed above a reader at scrollTop 9000, maximum dropping to 8600: the browser absorbed
  // 400, so 100 is left to apply.
  assert.equal(
    anchorCorrection(sample(200, 9000, 9100), sample(100, 8600, 8600)),
    -100,
  );
  // The same shrink with room to spare is corrected in full: nothing was clamped.
  assert.equal(
    anchorCorrection(sample(200, 9000, 50_000), sample(-300, 9000, 49_500)),
    -500,
  );
});
