// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";

import { parseMarkdownIntoBlocks as streamdownSplit } from "streamdown";

import {
  parseMarkdownBlockDetails,
  parseMarkdownIntoBlocks,
} from "../src/lib/parse-markdown-blocks.ts";

// Pieces that move the `$$` parity, open or close HTML, or end a block at a code fence.
const PIECES = [
  "$",
  "$$",
  "$$$",
  "$$$$",
  "\\$$",
  "x",
  "word ",
  "a $b$ c",
  "\n",
  "\n\n",
  "\r\n",
  "\r\n\r\n",
  "\n\n$$\n",
  "\n$$\n\n",
  "\n\n```\nprint('$$')\n```\n\n",
  "\n\n```sh\necho $$\n",
  "\n\n    indented $$ code\n\n",
  "\n\n<div>\n\n",
  "\n\n</div>\n\n",
  "\n\n<details><summary>s $$</summary>\n\n",
  "\n\n</details>\n\n",
  "\n\n<br/>\n",
  "\n\n- item $$\n- item\n\n",
  "\n\n> quote $\n\n",
  "\n\n# Heading $$\n\n",
  "\n\n| a | $$ |\n|---|---|\n| 1 | 2 |\n\n",
  "\n\n---\n\n",
];

function randomDocument(next: () => number): string {
  const length = 1 + Math.floor(next() * 40);
  let out = "";
  for (let i = 0; i < length; i += 1) {
    out += PIECES[Math.floor(next() * PIECES.length)];
  }
  return out;
}

function seeded(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1_103_515_245) + 12_345) >>> 0;
    return state / 2 ** 32;
  };
}

test("block split matches streamdown on random dollar, HTML and code layouts", () => {
  const next = seeded(7);
  for (let i = 0; i < 4000; i += 1) {
    const markdown = randomDocument(next);
    assert.deepEqual(
      parseMarkdownIntoBlocks(markdown),
      streamdownSplit(markdown),
      JSON.stringify(markdown),
    );
  }
});

test("block details split matches the plain split", () => {
  const next = seeded(11);
  for (let i = 0; i < 500; i += 1) {
    const markdown = randomDocument(next);
    assert.deepEqual(
      parseMarkdownBlockDetails(markdown).blocks,
      parseMarkdownIntoBlocks(markdown),
    );
  }
});

const paragraphs = (count: number): string =>
  Array.from(
    { length: count },
    (_, i) =>
      `Paragraph ${i} with inline math $x_${i}$ and enough words to fill a line.\n\n`,
  ).join("");

function fastest(markdown: string): number {
  let best = Number.POSITIVE_INFINITY;
  for (let i = 0; i < 3; i += 1) {
    const start = performance.now();
    parseMarkdownIntoBlocks(markdown);
    best = Math.min(best, performance.now() - start);
  }
  return best;
}

// The quadratic rescan costs ~500x the unmerged split; the bound leaves room for loaded runners.
test("blocks merged after a stray $$ cost about as much as unmerged ones", () => {
  const body = paragraphs(3000);
  const stray = `Bash keeps the shell PID in $$ here.\n\n${body}`;
  assert.equal(parseMarkdownIntoBlocks(stray).length, 1);
  assert.equal(parseMarkdownIntoBlocks(stray)[0], stray);
  const merged = fastest(stray);
  const split = fastest(body);
  assert.ok(
    merged < split * 10 + 50,
    `merged split ${merged.toFixed(1)}ms vs ${split.toFixed(1)}ms unmerged`,
  );
});
