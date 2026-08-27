// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The streaming reasoning pane flickered: roughly twice a second it jumped from the
// streamed tail to the first paragraph and back. A user screen recording of main put
// 40 of 287 frames in the wrong state, 16 episodes, median 467 ms apart.
//
// Cause: a `scroll` event cannot say who moved the scroller, and the pane treated ANY
// scrollTop decrease as the user scrolling up. When content shrank mid-stream the engine
// clamped scrollTop down, that read as user intent, the pin detached, and the next
// mutation re-attached within the threshold and snapped back to the bottom.
//
// These drive the reducer through the actual sequences rather than pinning source text,
// which is the point of extracting it: the sibling .tsx tests here can only assert on
// text, and no amount of text pinning would have caught a wrong comparison.

import assert from "node:assert/strict";
import test from "node:test";

import {
  AUTO_SCROLL_THRESHOLD_PX,
  createPinState,
  detachByUser,
  notePinnedTo,
  observeScroll,
  shouldAutoScroll,
} from "../src/components/assistant-ui/reasoning-autoscroll.ts";

const VIEWPORT = 256; // max-h-64
const maxFor = (scrollHeight: number) => Math.max(0, scrollHeight - VIEWPORT);

test("the real ordering: shrink, clamp, GROW BACK, then the scroll event arrives", () => {
  // This is the sequence that produces the flicker, and the one a naive fix misses.
  // `scroll` is a task; streaming mutations are microtasks. By the time the listener
  // runs, the shrink is over and the content is taller than before, so the listener
  // sees a low scrollTop against a LARGER maximum. Comparing maxima cannot help: the
  // smaller maximum never existed as far as this callback is concerned.
  let state = createPinState(maxFor(4000), maxFor(4000)); // pinned, max 3744
  state = observeScroll(state, 944, maxFor(4200));        // clamped to 944, max now 3944

  assert.equal(
    shouldAutoScroll(state),
    true,
    "no input event occurred, so nothing here is user intent and the pin must survive",
  );
});

test("the flicker sequence produces no detach across repeated shrink and grow", () => {
  let state = createPinState(maxFor(4000), maxFor(4000));
  // Each pair is (where the clamp left scrollTop, the height it grew back to).
  const episodes: Array<[number, number]> = [
    [944, 4200], [644, 4400], [44, 4600], [244, 4800], [0, 5000],
  ];

  let detachEpisodes = 0;
  for (const [clampedTo, grewTo] of episodes) {
    state = observeScroll(state, clampedTo, maxFor(grewTo));
    if (!shouldAutoScroll(state)) {
      detachEpisodes += 1;
    }
    state = notePinnedTo(state, maxFor(grewTo));
  }

  assert.equal(
    detachEpisodes,
    0,
    "each detach here is one visible flicker; the recording showed 16 in 9.57 s",
  );
});

test("the pane still follows the stream while content only grows", () => {
  let state = createPinState(maxFor(1000), maxFor(1000));
  for (const h of [1200, 1500, 2000, 3000]) {
    state = observeScroll(state, state.lastScrollTop, maxFor(h));
    assert.equal(shouldAutoScroll(state), true);
    state = notePinnedTo(state, maxFor(h));
  }
});

test("a user scrolling up DOES detach, and is not yanked back down", () => {
  let state = createPinState(maxFor(4000), maxFor(4000));

  // The input event is what says "the user". The scroll observation that follows it
  // only carries the geometry.
  state = detachByUser(state);
  state = observeScroll(state, maxFor(4000) - 900, maxFor(4000));
  assert.equal(shouldAutoScroll(state), false);

  // The stream keeps arriving and the document grows. The user stays where they are.
  for (const h of [4200, 4500, 5000]) {
    state = observeScroll(state, maxFor(4000) - 900, maxFor(h));
    assert.equal(
      shouldAutoScroll(state),
      false,
      "a reading user must not be yanked back to the bottom",
    );
  }
});

test("a wheel scroll up detaches even if the offset has not settled yet", () => {
  let state = createPinState(maxFor(4000), maxFor(4000));
  state = detachByUser(state);
  assert.equal(shouldAutoScroll(state), false);
});

test("returning to within the threshold of the bottom re-attaches", () => {
  let state = createPinState(maxFor(4000), maxFor(4000));
  state = detachByUser(state);
  state = observeScroll(state, maxFor(4000) - 900, maxFor(4000));
  assert.equal(shouldAutoScroll(state), false);

  state = observeScroll(state, maxFor(4000) - AUTO_SCROLL_THRESHOLD_PX, maxFor(4000));
  assert.equal(
    shouldAutoScroll(state),
    true,
    "scrolling back to the bottom resumes following the stream",
  );
});

test("a user parked mid-document stays parked, however the content moves", () => {
  // Geometry no longer detaches, so this has to come from the input event, and then
  // survive every subsequent shrink and grow until the user returns to the bottom.
  let state = createPinState(maxFor(4000), maxFor(4000));
  state = detachByUser(state);

  for (const [top, height] of [[900, 4200], [900, 1200], [900, 5000]] as const) {
    state = observeScroll(state, top, maxFor(height));
    assert.equal(
      shouldAutoScroll(state),
      false,
      "a reading user must not be yanked back to the bottom",
    );
  }
});

test("the pin targets scrollHeight - clientHeight, so a shrink cannot strand it", () => {
  // The old code wrote el.scrollTop = el.scrollHeight and relied on the engine to
  // clamp. notePinnedTo records where the write actually landed, so the scroll event
  // it provokes reads as no change rather than as a decrease.
  let state = createPinState(maxFor(4000), maxFor(4000));
  state = notePinnedTo(state, maxFor(1200));
  state = observeScroll(state, maxFor(1200), maxFor(1200));

  assert.equal(shouldAutoScroll(state), true);
  assert.equal(state.lastScrollTop, maxFor(1200));
});

test("a document shorter than the viewport pins at 0 without detaching", () => {
  // max-h-64 means content can be shorter than the pane. maxScrollTop clamps at 0,
  // and 0 must not read as the user having scrolled to the top.
  let state = createPinState(maxFor(4000), maxFor(4000));
  state = observeScroll(state, 0, maxFor(100));

  assert.equal(
    shouldAutoScroll(state),
    true,
    "content shorter than the viewport is a clamp to 0, not a user scroll to the top",
  );
});
