// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

// Regression for unslothai/unsloth#11376: long single-line backslash runs must not
// invoke marked's inline tokenizer during block splitting.

import assert from "node:assert/strict";
import test from "node:test";

import { parseMarkdownIntoBlocks as streamdownSplit } from "streamdown";

import { parseMarkdownIntoBlocks } from "../src/lib/parse-markdown-blocks.ts";

const BACKSLASH_RUN = "\\".repeat(1500);

function longBackslashLine(chars: number): string {
  const unit = BACKSLASH_RUN;
  const repeats = Math.ceil(chars / unit.length);
  return unit.repeat(repeats).slice(0, chars);
}

test("block split matches streamdown on a long backslash line", () => {
  const markdown = longBackslashLine(200_000);
  assert.deepEqual(parseMarkdownIntoBlocks(markdown), streamdownSplit(markdown));
});

test("block split stays fast on a long backslash line", () => {
  const markdown = longBackslashLine(200_000);
  const start = performance.now();
  parseMarkdownIntoBlocks(markdown);
  const elapsed = performance.now() - start;
  assert.ok(
    elapsed < 500,
    `expected block split under 500ms, got ${elapsed.toFixed(1)}ms`,
  );
});

test("newlines keep backslash lines cheap", () => {
  const markdown = `${longBackslashLine(1504)}\n`.repeat(Math.ceil(200_000 / 1505));
  const start = performance.now();
  parseMarkdownIntoBlocks(markdown);
  const elapsed = performance.now() - start;
  assert.ok(elapsed < 500, `expected chunked line under 500ms, got ${elapsed.toFixed(1)}ms`);
});
