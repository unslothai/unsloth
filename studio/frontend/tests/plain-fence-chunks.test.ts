// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  PLAIN_FENCE_CHUNK_LINES,
  type PlainFenceChunks,
  updatePlainFenceChunks,
} from "../src/components/assistant-ui/plain-fence-chunks.ts";

function expectedNodes(source: string): string[] {
  const nodes: string[] = [];
  const body = source.replace(/\n+$/, "") || "\n";
  for (const [index, line] of body.split("\n").entries()) {
    if (index > 0) nodes.push("\n");
    if (line !== "") nodes.push(line);
  }
  return nodes;
}

function freezeState(state: PlainFenceChunks): PlainFenceChunks {
  state.chunks.forEach(Object.freeze);
  Object.freeze(state.chunks);
  return Object.freeze(state);
}

test("plain chunks preserve every text node and trim only trailing LF", () => {
  const sources = [
    "",
    "\n",
    "\n\n",
    "a",
    "a\n",
    "\n\na\n\n",
    "a\n\nb",
    "\r\n",
    "a\r\nb\r\n",
    "a\r",
    "\t \n\t",
    "🐢\n日本語\n𝄞",
    Array.from({ length: 131 }, (_, index) =>
      index % 3 ? `line ${index}` : "",
    ).join("\n"),
  ];
  for (const source of sources) {
    const state = updatePlainFenceChunks(null, source);
    assert.deepEqual(
      state.chunks.flat(),
      expectedNodes(source),
      JSON.stringify(source),
    );
    assert.equal(state.source, source);
  }
});

test("streaming appends reuse completed chunks and replace only the growing tail", () => {
  const source = Array.from(
    { length: 140 },
    (_, index) => `line ${index}`,
  ).join("\n");
  const first = freezeState(updatePlainFenceChunks(null, source));
  const second = updatePlainFenceChunks(first, `${source} more`);
  assert.equal(second.chunks.length, 3);
  assert.equal(second.chunks[0], first.chunks[0]);
  assert.equal(second.chunks[1], first.chunks[1]);
  assert.notEqual(second.chunks[2], first.chunks[2]);
  assert.deepEqual(second.chunks.flat(), expectedNodes(second.source));
  assert.deepEqual(first.chunks.flat(), expectedNodes(source));
});

test("filling a chunk preserves its position and reuses it across the next boundary", () => {
  let source = Array.from(
    { length: PLAIN_FENCE_CHUNK_LINES - 1 },
    (_, index) => `line ${index}`,
  ).join("\n");
  const first = freezeState(updatePlainFenceChunks(null, source));
  source += "\nlast";
  const full = freezeState(updatePlainFenceChunks(first, source));
  assert.equal(full.chunks.length, 1);
  assert.notEqual(full.chunks[0], first.chunks[0]);
  const next = updatePlainFenceChunks(full, `${source}\nnext`);
  assert.equal(next.chunks.length, 2);
  assert.equal(next.chunks[0], full.chunks[0]);
  assert.deepEqual(next.chunks[1], ["\n", "next"]);
});

test("trailing newlines reuse all chunks until subsequent content makes them visible", () => {
  for (const source of [
    "",
    "a",
    Array.from({ length: 64 }, () => "x").join("\n"),
  ]) {
    const initial = freezeState(updatePlainFenceChunks(null, source));
    const trailing = freezeState(
      updatePlainFenceChunks(initial, `${source}\n\n`),
    );
    assert.equal(trailing.chunks, initial.chunks);
    const continued = updatePlainFenceChunks(trailing, `${source}\n\nend`);
    assert.deepEqual(continued.chunks.flat(), expectedNodes(continued.source));
  }
});

test("identical source returns the existing state", () => {
  const state = freezeState(updatePlainFenceChunks(null, "a\nb\n"));
  assert.equal(updatePlainFenceChunks(state, state.source), state);
});

test("replacement and truncation cannot retain stale lines", () => {
  const source = Array.from(
    { length: 140 },
    (_, index) => `line ${index}`,
  ).join("\n");
  const initial = freezeState(updatePlainFenceChunks(null, source));
  for (const replacement of [
    source.slice(0, -20),
    source.slice(0, 30),
    "replacement\ntext",
    "",
    "\n",
    source.replace("line 3", "edited"),
  ]) {
    const state = updatePlainFenceChunks(initial, replacement);
    assert.deepEqual(state.chunks.flat(), expectedNodes(replacement));
    assert.deepEqual(initial.chunks.flat(), expectedNodes(source));
  }
});

test("mixed appends, edits and truncations preserve Unicode and blank-line segmentation", () => {
  const pieces = ["a", "\n", "\r", "🐢", "", "\n\n", "\t", "日本語", " ", "𝄞"];
  let seed = 5381;
  let state: PlainFenceChunks | undefined;
  let source = "";
  for (let index = 0; index < 700; index += 1) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    if (index % 113 === 0)
      source = source.slice(0, Math.floor(source.length / 2));
    else if (index % 97 === 0) source = `edited\n${source}`;
    else source += pieces[seed % pieces.length];
    state = freezeState(updatePlainFenceChunks(state, source));
    assert.deepEqual(
      state.chunks.flat(),
      expectedNodes(source),
      `update ${index}`,
    );
  }
});
