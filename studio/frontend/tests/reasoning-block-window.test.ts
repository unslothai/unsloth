// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The arithmetic behind the streaming reasoning pane's block window.
//
// The pane holds the whole reasoning body while a model thinks, so a long generation mounts tens
// of thousands of nodes into a 256px box. The window renders a contiguous SUFFIX of the block
// list and replaces the rest with one spacer. Everything here is about the two numbers that makes
// possible: where the window starts, and how tall the spacer is.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  PANE_BOTTOM_THRESHOLD_PX,
  RETAIN_ABOVE_PX,
  blockWindowFlippedRange,
  blockWindowSpacerHeight,
  chooseBlockWindowStart,
  createBlockWindowState,
  isBlockMounted,
  isBlockSuffix,
  mountedBlockContents,
  recordBlockContent,
  recordBlockOffset,
  resetBlockWindow,
  setBlockWindowStart,
} from "../src/components/assistant-ui/block-window.ts";

/** A document of `count` blocks, each `height` tall, already measured. */
function measured(count: number, height: number) {
  const state = createBlockWindowState();
  for (let index = 1; index < count; index += 1) {
    recordBlockOffset(state, index, index * height);
  }
  return state;
}

test("the window is a suffix and starts at the top until there is something to hide", () => {
  const state = measured(40, 100);

  // Nothing scrolled: RETAIN_ABOVE_PX of content has to sit above the viewport before any of it
  // can be dropped, so the whole document is mounted.
  assert.equal(chooseBlockWindowStart(state, 0), 0);
  assert.equal(chooseBlockWindowStart(state, RETAIN_ABOVE_PX), 0);
  assert.equal(blockWindowSpacerHeight(state, 0), 0);

  // One block past the retained band.
  assert.equal(chooseBlockWindowStart(state, RETAIN_ABOVE_PX + 100), 1);

  // Scrolled to the bottom of a 4000px document in a 256px pane.
  const start = chooseBlockWindowStart(state, 4000 - 256);
  assert.equal(start, 22);
  assert.equal(start * 100, 2200);
  assert.ok(4000 - 256 - start * 100 >= RETAIN_ABOVE_PX);

  assert.equal(isBlockMounted({ ...state, start }, 21), false);
  assert.equal(isBlockMounted({ ...state, start }, 22), true);
  assert.equal(isBlockMounted({ ...state, start }, 39), true);
});

test("the retained band really is what bounds the window, at any document length", () => {
  // 400 blocks, i.e. a 40,000px document: what is mounted must not grow with it.
  const state = measured(400, 100);
  const start = chooseBlockWindowStart(state, 40_000 - 256);
  const mountedPx = 40_000 - start * 100;
  assert.ok(
    mountedPx <= RETAIN_ABOVE_PX + 256 + 100,
    `mounted ${mountedPx}px above the bound`,
  );
  // And the same fixture ten times longer picks the same bound, not ten times more.
  const longer = measured(4000, 100);
  const longerStart = chooseBlockWindowStart(longer, 400_000 - 256);
  assert.equal(400_000 - longerStart * 100, mountedPx);
});

test("only indices with a measured offset can become the window start", () => {
  const state = createBlockWindowState();
  recordBlockOffset(state, 3, 300);
  recordBlockOffset(state, 9, 900);
  // 5 and 7 were never measured (empty blocks render no element), so a viewport that has passed
  // them still stops at the last index whose height is known.
  assert.equal(chooseBlockWindowStart(state, 900 + RETAIN_ABOVE_PX - 1), 3);
  assert.equal(chooseBlockWindowStart(state, 900 + RETAIN_ABOVE_PX), 9);
  assert.equal(blockWindowSpacerHeight(state, 3), 300);
  assert.equal(blockWindowSpacerHeight(state, 9), 900);
});

test("the spacer is the height of the range it replaces, from the frame before the drop", () => {
  const state = measured(40, 100);

  // The identity the design rests on: moving the window from s to s' grows the spacer by exactly
  // offset(s') - offset(s), which is the height of the removed blocks INCLUDING their collapsed
  // margins, because both numbers were read in the same pre-mutation frame.
  for (const [from, to] of [
    [0, 5],
    [5, 12],
    [12, 30],
  ] as const) {
    const grew =
      blockWindowSpacerHeight(state, to) - blockWindowSpacerHeight(state, from);
    const removed =
      (state.offsets.get(to) ?? 0) - (from === 0 ? 0 : (state.offsets.get(from) ?? 0));
    assert.equal(grew, removed);
  }

  setBlockWindowStart(state, 12);
  assert.equal(state.spacerHeight, 1200);
  setBlockWindowStart(state, 30);
  assert.equal(state.spacerHeight, 3000);
  // Back up again: the earlier height is restored EXACTLY. An incremental spacer would have
  // accumulated whatever the two subtractions rounded off.
  setBlockWindowStart(state, 12);
  assert.equal(state.spacerHeight, 1200);
  setBlockWindowStart(state, 0);
  assert.equal(state.spacerHeight, 0);
});

test("moving the window reports a change only when something changed", () => {
  const state = measured(10, 100);
  assert.equal(setBlockWindowStart(state, 4), true);
  assert.equal(setBlockWindowStart(state, 4), false);
  assert.equal(setBlockWindowStart(state, 0), true);
});

test("only the blocks whose mounted state flips are in the flipped range", () => {
  assert.deepEqual(blockWindowFlippedRange(0, 22), { from: 0, to: 22 });
  assert.deepEqual(blockWindowFlippedRange(22, 14), { from: 14, to: 22 });
  assert.deepEqual(blockWindowFlippedRange(7, 7), { from: 7, to: 7 });

  // The point of the range: for a window move from 22 to 14, blocks 14..21 flip and NOTHING else
  // does. 22 and up were mounted before and after; 13 and down were hidden before and after.
  const { from, to } = blockWindowFlippedRange(22, 14);
  for (const index of [0, 13, 22, 39]) {
    assert.ok(
      index < from || index >= to,
      `${index} must not be in the flipped range`,
    );
  }
  for (const index of [14, 18, 21]) {
    assert.ok(index >= from && index < to);
  }
});

test("a width change is the thing that invalidates every frozen height at once", () => {
  const state = measured(40, 100);
  setBlockWindowStart(state, 22);
  recordBlockContent(state, 22, "body");
  assert.equal(state.offsets.size, 39);

  resetBlockWindow(state);

  assert.equal(state.start, 0);
  assert.equal(state.spacerHeight, 0);
  assert.equal(state.offsets.size, 0);
  assert.equal(state.contents.size, 0);
  assert.equal(state.highestIndex, -1);
  // Which is the whole argument that a stale height cannot outlive a reflow: after a reset the
  // window can only be 0, so every block remounts and re-measures.
  assert.equal(chooseBlockWindowStart(state, 100_000), 0);
});

test("a re-parse behind the live edge is reported, and the live edge itself is not", () => {
  const state = createBlockWindowState();
  for (let index = 0; index <= 30; index += 1) {
    assert.equal(
      recordBlockContent(state, index, `block ${index}`),
      false,
      "first sight of a block is never a re-parse",
    );
  }

  // The newest block grows with every token. That is not a re-parse.
  assert.equal(recordBlockContent(state, 30, "block 30 and more"), false);
  assert.equal(recordBlockContent(state, 29, "block 29 and more"), false);
  assert.equal(recordBlockContent(state, 23, "block 23 rewritten"), false);

  // Far behind it, a block that changes means the renderer re-segmented the document, which is
  // exactly what a late GFM footnote definition does, and the frozen heights no longer describe
  // it. The source string is never touched, so this can never corrupt the TEXT; it can only make
  // the spacer the wrong height, and the answer is to remeasure everything.
  assert.equal(recordBlockContent(state, 10, "block 10 rewritten"), true);
  // Re-reporting the same content is not a change.
  assert.equal(recordBlockContent(state, 10, "block 10 rewritten"), false);
});

test("what is mounted is always a contiguous suffix of the renderer's own block list", () => {
  const blocks = ["a", "\n\n", "b", "\n\n", "c", "\n\n", "d"];
  for (let start = 0; start <= blocks.length; start += 1) {
    const mounted = mountedBlockContents(blocks, start);
    assert.equal(mounted.length, blocks.length - start);
    assert.ok(
      isBlockSuffix(blocks, mounted),
      `start ${start} did not produce a suffix`,
    );
    assert.equal(mounted.join(""), blocks.slice(start).join(""));
  }

  // The negative, so the check above is not vacuous.
  assert.equal(isBlockSuffix(blocks, ["a", "\n\n", "b"]), false);
  assert.equal(isBlockSuffix(blocks, ["c", "d"]), false);
  assert.equal(isBlockSuffix(blocks, [...blocks, "e"]), false);
});

test("the pinned threshold is the autoscroll's own, so the two cannot drift apart", () => {
  const reasoning = new URL(
    "../src/components/assistant-ui/reasoning.tsx",
    import.meta.url,
  );
  const source = readFileSync(reasoning, "utf8");
  assert.match(
    source,
    /distanceFromBottom <= PANE_BOTTOM_THRESHOLD_PX/,
    "the reasoning pane's autoscroll must use the shared threshold",
  );
  assert.doesNotMatch(
    source,
    /AUTO_SCROLL_THRESHOLD_PX\s*=/,
    "a second copy of the threshold would be free to drift from the window's",
  );
  assert.equal(PANE_BOTTOM_THRESHOLD_PX, 24);
});
