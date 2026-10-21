// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { updateFenceChunks } from "../src/components/assistant-ui/code-fence-chunks.ts";
import {
  advanceHighlightProgress,
  EMPTY_HIGHLIGHT_COUNTS,
  EMPTY_HIGHLIGHT_PROGRESS,
  HIGHLIGHT_BATCH_LINES,
  HIGHLIGHT_BATCH_TOKENS,
  type HighlightProgress,
  highlightLineCount,
  initialHighlightProgress,
  needsHighlightBatches,
  retainHighlightProgress,
} from "../src/components/assistant-ui/fence-highlight-progress.ts";

const linesOf = (...lengths: number[]) => lengths.map((length) => ({ length }));
const coverage = (lines: readonly { length: number }[], progress: HighlightProgress | null): number[] =>
  lines.map((line, index) => progress === null ? line.length : highlightLineCount(progress, index));

function assertBoundedStep(before: readonly number[], after: readonly number[]): number {
  let tokens = 0;
  let touched = 0;
  for (let index = 0; index < after.length; index += 1) {
    const added = after[index] - before[index];
    assert.ok(added >= 0, `line ${index} retains its highlighted tokens`);
    if (added > 0) touched += 1;
    tokens += added;
  }
  assert.ok(tokens <= HIGHLIGHT_BATCH_TOKENS);
  assert.ok(touched <= HIGHLIGHT_BATCH_LINES);
  return tokens;
}

test("batching activates above either the line cap or the token cap", () => {
  assert.equal(needsHighlightBatches([], 240), false);
  assert.equal(needsHighlightBatches(linesOf(4096), 240), false);
  assert.equal(needsHighlightBatches(linesOf(4097), 240), true);
  assert.equal(needsHighlightBatches(linesOf(2048, 2048), 240), false);
  assert.equal(needsHighlightBatches(linesOf(2048, 2049), 240), true);
  assert.equal(needsHighlightBatches(Array.from({ length: 240 }, () => ({ length: 1 })), 240), false);
  assert.equal(needsHighlightBatches(Array.from({ length: 241 }, () => ({ length: 0 })), 240), true);
});

test("the first batch is bounded and leaves visible prioritization available", () => {
  const lines = linesOf(10_000, 10);
  const initial = initialHighlightProgress(lines);
  assert.deepEqual(initial, { counts: [[512, 0]], cursor: 0, located: false });
  assert.equal(initialHighlightProgress([]), null);
  assert.equal(initialHighlightProgress(linesOf(0, 0)), null);
  assert.equal(initialHighlightProgress(linesOf(512)), null);
});

test("a 10000-token line is partitioned into strict 512-token batches without gaps", () => {
  const lines = linesOf(10_000, 3);
  let progress: HighlightProgress | null = EMPTY_HIGHLIGHT_PROGRESS;
  let before = coverage(lines, progress);
  let batches = 0;
  let visited = 0;
  while (progress !== null) {
    progress = advanceHighlightProgress(lines, progress);
    const after = coverage(lines, progress);
    const added = assertBoundedStep(before, after);
    assert.ok(added > 0);
    visited += added;
    before = after;
    assert.ok(++batches < 30);
  }
  assert.deepEqual(before, [10_000, 3]);
  assert.equal(visited, 10_003);
  assert.equal(batches, 20);
});

test("ordinary lines remain whole and each batch touches at most 64 incomplete lines", () => {
  assert.deepEqual(advanceHighlightProgress(linesOf(300, 300), EMPTY_HIGHLIGHT_PROGRESS), {
    counts: [[300, 0]], cursor: 1, located: true,
  });
  const lines = Array.from({ length: 200 }, () => ({ length: 1 }));
  const result = advanceHighlightProgress(lines, EMPTY_HIGHLIGHT_PROGRESS);
  assert.ok(result);
  assert.deepEqual(coverage(lines, result), [...Array<number>(64).fill(1), ...Array<number>(136).fill(0)]);
  assert.equal(result.cursor, 64);
});

test("completed lines do not consume the batch line budget", () => {
  const lines = Array.from({ length: 200 }, () => ({ length: 1 }));
  const counts = Array.from({ length: 200 }, (_, index) => index % 2);
  const result = advanceHighlightProgress(lines, { counts: updateFenceChunks(null, counts), cursor: 0, located: true });
  assert.ok(result);
  assert.equal(assertBoundedStep(counts, coverage(lines, result)), 64);
  assert.equal(result.cursor, 128);
});

test("mixed sparse and dense lines complete monotonically within both budgets", () => {
  const lines = Array.from({ length: 2000 }, (_, index) => ({
    length: index % 97 === 0 ? 10_000 : (index * 17) % 41,
  }));
  let progress: HighlightProgress | null = EMPTY_HIGHLIGHT_PROGRESS;
  let before = coverage(lines, progress);
  let visited = 0;
  let batches = 0;
  while (progress !== null) {
    progress = advanceHighlightProgress(lines, progress);
    const after = coverage(lines, progress);
    const added = assertBoundedStep(before, after);
    assert.ok(added > 0);
    visited += added;
    before = after;
    assert.ok(++batches < 2000);
  }
  assert.deepEqual(before, lines.map((line) => line.length));
  assert.equal(visited, lines.reduce((sum, line) => sum + line.length, 0));
});

test("visible prioritization happens once and wraps to finish earlier lines without decoloring", () => {
  const lines = Array.from({ length: 400 }, (_, index) => ({ length: index === 320 ? 10_000 : 12 }));
  let progress = initialHighlightProgress(lines);
  assert.ok(progress);
  let before = coverage(lines, progress);
  const initial = before.slice();
  let steps = 0;
  while (progress !== null) {
    const next = advanceHighlightProgress(lines, progress, steps === 0 ? 300 : 200);
    const after = coverage(lines, next);
    assert.ok(assertBoundedStep(before, after) > 0);
    if (steps === 0) {
      assert.deepEqual(after.slice(0, 300), initial.slice(0, 300));
      assert.equal(after[300], lines[300].length);
      assert.equal(after[320], 272);
    }
    if (steps === 1) {
      assert.equal(after[200], 0);
      assert.equal(after[320], 784);
    }
    before = after;
    progress = next;
    assert.ok(++steps < 100);
  }
  assert.deepEqual(before, lines.map((line) => line.length));
});

test("batch advancement never mutates previously rendered progress", () => {
  const counts = Object.freeze([Object.freeze([10, 0, 0])]);
  const previous = Object.freeze({ counts, cursor: 1, located: true });
  const result = advanceHighlightProgress(linesOf(10, 10_000, 10), previous);
  assert.ok(result);
  assert.notEqual(result.counts, counts);
  assert.deepEqual(previous.counts, [[10, 0, 0]]);
  assert.deepEqual(result.counts, [[10, 512, 0]]);
});

test("replacement of an early line preserves every already colored unchanged suffix line", () => {
  const previous = [[1, 2], [3, 4], [5, 6]];
  const next = [Array.from({ length: 10_000 }, (_, index) => index), previous[1], previous[2]];
  const retained = retainHighlightProgress(previous, next, null);
  assert.deepEqual(retained, { counts: [[2, 2, 2]], cursor: 0, located: false });
  assert.equal(next[2], previous[2]);
  assert.equal(highlightLineCount(retained!, 2), next[2].length);
  const advanced = advanceHighlightProgress(next, retained!);
  assert.deepEqual(advanced?.counts, [[514, 2, 2]]);
});

test("several changed middle lines preserve disjoint colored islands and partial dense lines", () => {
  const previous = Array.from({ length: 6 }, () => Array<number>(1000).fill(0));
  const next = previous.slice();
  next[1] = Array<number>(1100).fill(1);
  next[3] = Array<number>(1200).fill(2);
  const progress = { counts: [[1000, 1000, 512, 1000, 0, 1000]], cursor: 2, located: true };
  const retained = retainHighlightProgress(previous, next, progress);
  assert.deepEqual(retained, progress);
  assert.notEqual(retained?.counts, progress.counts);
  assert.equal(highlightLineCount(retained!, 2), 512);
  assert.equal(highlightLineCount(retained!, 5), 1000);
  assert.equal(retained?.counts[0], progress.counts[0]);
});

test("null tokens and small coverage upgraded to dense tokens regain bounded progress", () => {
  const dense = [Array.from({ length: 10_000 }, (_, index) => index)];
  assert.deepEqual(retainHighlightProgress(null, dense, null), { counts: [EMPTY_HIGHLIGHT_COUNTS], cursor: 0, located: false });
  const retained = retainHighlightProgress([[0]], dense, null);
  assert.deepEqual(retained, { counts: [[1]], cursor: 0, located: false });
  assert.deepEqual(advanceHighlightProgress(dense, retained!), {
    counts: [[513]], cursor: 0, located: true,
  });
});

test("append starts only new lines at zero and leaves existing completed lines intact", () => {
  const previous = [[1], [2], [3]];
  const next = [...previous, [4]];
  const progress = retainHighlightProgress(previous, next, null);
  assert.deepEqual(progress, { counts: [[1, 1, 1, 0]], cursor: 0, located: false });
  assert.equal(advanceHighlightProgress(next, progress!), null);
  assert.equal(retainHighlightProgress(previous, previous.slice(), null), null);
});

test("truncation clamps counts and cursor and recognizes entirely complete retained content", () => {
  const previous = linesOf(1000, 1000, 1000);
  const progress = { counts: [[1000, 512, 0]], cursor: 2, located: true };
  assert.deepEqual(retainHighlightProgress(previous, linesOf(1000, 800), progress), {
    counts: [[1000, 512]], cursor: 1, located: true,
  });
  assert.equal(retainHighlightProgress(previous, linesOf(10, 20), progress), null);
  assert.equal(retainHighlightProgress(previous, [], progress), null);
});

test("a batch copies only its changed groups and shares untouched empty groups", () => {
  const lines = Array.from({ length: 3200 }, () => ({ length: 12 }));
  const previous = initialHighlightProgress(lines)!;
  for (const group of previous.counts) Object.freeze(group);
  Object.freeze(previous.counts);
  Object.freeze(previous);
  const next = advanceHighlightProgress(lines, previous, 1024)!;
  assert.equal(next.counts.length, 100);
  const changed = next.counts.flatMap((group, index) => group !== previous.counts[index] ? [index] : []);
  assert.deepEqual(changed, [32, 33]);
  assert.equal(next.counts[2], EMPTY_HIGHLIGHT_COUNTS);
  assert.equal(next.counts[99], EMPTY_HIGHLIGHT_COUNTS);
  assert.equal(highlightLineCount(previous, 1024), 0);
  assert.equal(highlightLineCount(next, 1024), 12);
  assert.equal(assertBoundedStep(coverage(lines, previous), coverage(lines, next)), 504);
});

test("retaining replacement tokens reuses disjoint unchanged count groups", () => {
  const previous = Array.from({ length: 96 }, () => ({ length: 1000 }));
  const flat = previous.map((line, index) => index === 33 ? 512 : index === 95 ? 0 : line.length);
  const groups = updateFenceChunks(null, flat);
  for (const group of groups) Object.freeze(group);
  Object.freeze(groups);
  const progress = { counts: groups, cursor: 33, located: true };
  const next = previous.slice();
  next[2] = { length: 800 };
  next[34] = { length: 1100 };
  const retained = retainHighlightProgress(previous, next, progress)!;
  assert.notEqual(retained.counts[0], groups[0]);
  assert.equal(retained.counts[1], groups[1]);
  assert.equal(retained.counts[2], groups[2]);
  assert.equal(highlightLineCount(retained, 2), 800);
  assert.equal(highlightLineCount(retained, 33), 512);
  assert.equal(highlightLineCount(retained, 34), 1000);
  assert.equal(highlightLineCount(progress, 2), 1000);
});

test("appending and truncating preserve every unaffected count group", () => {
  const previous = Array.from({ length: 96 }, () => ({ length: 1000 }));
  const progress = {
    counts: updateFenceChunks(null, previous.map((line, index) => index === 64 ? 512 : line.length)),
    cursor: 64,
    located: true,
  };
  const appended = retainHighlightProgress(previous, [...previous, { length: 1 }], progress)!;
  assert.deepEqual(appended.counts.slice(0, 3), progress.counts);
  for (let index = 0; index < 3; index += 1) assert.equal(appended.counts[index], progress.counts[index]);
  assert.equal(appended.counts[3], EMPTY_HIGHLIGHT_COUNTS);
  assert.equal(highlightLineCount(appended, 96), 0);
  const truncated = retainHighlightProgress(previous, previous.slice(0, 65), progress)!;
  assert.equal(truncated.counts[0], progress.counts[0]);
  assert.equal(truncated.counts[1], progress.counts[1]);
  assert.deepEqual(truncated.counts[2], [512]);
  assert.notEqual(truncated.counts[2], progress.counts[2]);
});
