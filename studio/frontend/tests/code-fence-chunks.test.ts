// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  FENCE_CHUNK_LINES,
  type FenceBodyMemoProps,
  sameFenceBodyProps,
  updateFenceChunks,
} from "../src/components/assistant-ui/code-fence-chunks.ts";

const makeLines = (count: number) =>
  Array.from({ length: count }, (_, index) => [{ content: `line ${index}` }]);

test("groups preserve token line order and use bounded immutable slices", () => {
  const lines = makeLines(FENCE_CHUNK_LINES * 2 + 3);
  const chunks = updateFenceChunks(null, lines);
  assert.deepEqual(chunks.map((chunk) => chunk.length), [FENCE_CHUNK_LINES, FENCE_CHUNK_LINES, 3]);
  assert.deepEqual(chunks.flat(), lines);
  assert.notEqual(chunks[0], lines);
  for (const [index, line] of chunks.flat().entries()) {
    assert.equal(line, lines[index]);
  }
});

test("identical line references reuse every chunk and the chunk list", () => {
  const lines = makeLines(FENCE_CHUNK_LINES * 2 + 3);
  const previous = updateFenceChunks(null, lines);
  assert.equal(updateFenceChunks(previous, lines), previous);
  assert.equal(updateFenceChunks(previous, [...lines]), previous);
});

test("a growing live tail only replaces its own chunk", () => {
  const lines = makeLines(FENCE_CHUNK_LINES * 2 + 3);
  const previous = updateFenceChunks(null, lines);
  Object.freeze(previous);
  for (const chunk of previous) Object.freeze(chunk);
  const nextLines = [...lines];
  nextLines[nextLines.length - 1] = [{ content: "live tail grows" }];
  const next = updateFenceChunks(previous, nextLines);
  assert.notEqual(next, previous);
  assert.equal(next[0], previous[0]);
  assert.equal(next[1], previous[1]);
  assert.notEqual(next[2], previous[2]);
  assert.equal(previous[2][2], lines[lines.length - 1]);
  assert.deepEqual(next.flat(), nextLines);
});

test("middle line changes are detected when both edge references are unchanged", () => {
  const lines = makeLines(FENCE_CHUNK_LINES * 3);
  const previous = updateFenceChunks(null, lines);
  const nextLines = [...lines];
  nextLines[FENCE_CHUNK_LINES + 11] = [{ content: "retokenized middle" }];
  const next = updateFenceChunks(previous, nextLines);
  assert.equal(next[0], previous[0]);
  assert.notEqual(next[1], previous[1]);
  assert.equal(next[2], previous[2]);
  assert.deepEqual(next.flat(), nextLines);
});

test("appending across a boundary reuses completed groups", () => {
  const lines = makeLines(FENCE_CHUNK_LINES * 2 - 1);
  const previous = updateFenceChunks(null, lines);
  const nextLines = [...lines, ...makeLines(3)];
  const next = updateFenceChunks(previous, nextLines);
  assert.deepEqual(next.map((chunk) => chunk.length), [FENCE_CHUNK_LINES, FENCE_CHUNK_LINES, 2]);
  assert.equal(next[0], previous[0]);
  assert.notEqual(next[1], previous[1]);
  assert.equal(previous[1].length, FENCE_CHUNK_LINES - 1);
  assert.deepEqual(next.flat(), nextLines);
  const appended = updateFenceChunks(next, [...nextLines, ...makeLines(1)]);
  assert.equal(appended[0], next[0]);
  assert.equal(appended[1], next[1]);
  assert.notEqual(appended[2], next[2]);
});

test("a full previous tail remains stable when a new group starts", () => {
  const lines = makeLines(FENCE_CHUNK_LINES * 2);
  const previous = updateFenceChunks(null, lines);
  const next = updateFenceChunks(previous, [...lines, ...makeLines(1)]);
  assert.equal(next[0], previous[0]);
  assert.equal(next[1], previous[1]);
  assert.equal(next[2].length, 1);
});

test("truncation drops later groups and replaces only a shortened group", () => {
  const lines = makeLines(FENCE_CHUNK_LINES * 3);
  const previous = updateFenceChunks(null, lines);
  const nextLines = lines.slice(0, FENCE_CHUNK_LINES + 7);
  const next = updateFenceChunks(previous, nextLines);
  assert.equal(next.length, 2);
  assert.equal(next[0], previous[0]);
  assert.notEqual(next[1], previous[1]);
  assert.equal(next[1].length, 7);
  assert.deepEqual(next.flat(), nextLines);
  const boundary = updateFenceChunks(next, lines.slice(0, FENCE_CHUNK_LINES));
  assert.equal(boundary.length, 1);
  assert.equal(boundary[0], previous[0]);
});

test("resetting the source and empty transitions retain no stale lines", () => {
  const lines = makeLines(FENCE_CHUNK_LINES * 2);
  const previous = updateFenceChunks(null, lines);
  const replacement = makeLines(lines.length);
  const next = updateFenceChunks(previous, replacement);
  assert.notEqual(next[0], previous[0]);
  assert.notEqual(next[1], previous[1]);
  assert.deepEqual(next.flat(), replacement);
  const empty = updateFenceChunks(next, []);
  assert.deepEqual(empty, []);
  assert.equal(updateFenceChunks(empty, []), empty);
  assert.deepEqual(updateFenceChunks(null, []), []);
  assert.deepEqual(updateFenceChunks(empty, lines).flat(), lines);
});

type TestResult = { tokens: ReturnType<typeof makeLines>; fg?: string };
const bodyProps = (
  result: TestResult | null = { tokens: makeLines(1) },
): FenceBodyMemoProps<TestResult> => ({
  result,
  source: "line 0",
  language: "python",
  isIncomplete: true,
  windowing: false,
});

test("source growth skips a body render only with the same nonempty tokens result", () => {
  const previous = bodyProps();
  assert.equal(sameFenceBodyProps(previous, { ...previous }), true);
  assert.equal(
    sameFenceBodyProps(previous, { ...previous, source: "line 0 grows" }),
    true,
  );
  const nextResult = { tokens: previous.result!.tokens };
  assert.equal(
    sameFenceBodyProps(previous, { ...previous, result: nextResult }),
    false,
  );
  assert.equal(
    sameFenceBodyProps(previous, {
      ...previous,
      result: { ...nextResult, fg: "#eee" },
    }),
    false,
  );
});

test("null and empty results still render source changes in the plain fallback", () => {
  for (const result of [null, { tokens: [] }]) {
    const previous = bodyProps(result);
    assert.equal(sameFenceBodyProps(previous, { ...previous }), true);
    assert.equal(
      sameFenceBodyProps(previous, { ...previous, source: "replacement" }),
      false,
    );
  }
  const previous = bodyProps();
  assert.equal(
    sameFenceBodyProps(previous, { ...previous, result: null }),
    false,
  );
});

test("source equality does not inspect tokens", () => {
  const previous = bodyProps({
    get tokens(): ReturnType<typeof makeLines> {
      throw new Error("tokens were unnecessarily inspected");
    },
  });
  assert.equal(sameFenceBodyProps(previous, { ...previous }), true);
});

test("completion, language and window changes always update the body", () => {
  const previous = bodyProps();
  for (const update of [
    { isIncomplete: false },
    { isIncomplete: undefined },
    { language: "javascript" },
    { language: null },
    { windowing: true },
  ]) {
    assert.equal(sameFenceBodyProps(previous, { ...previous, ...update }), false);
  }
  const completed = { ...previous, isIncomplete: false };
  assert.equal(sameFenceBodyProps(completed, previous), false);
  assert.equal(
    sameFenceBodyProps(completed, { ...completed, source: "unchanged tokens" }),
    true,
  );
});
