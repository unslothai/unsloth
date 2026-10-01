// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import assert from "node:assert/strict";
import test from "node:test";

import { parseMarkdownIntoBlocks as streamdownSplit } from "streamdown";

import { parseMarkdownIntoBlocks } from "../src/lib/parse-markdown-blocks.ts";

const longBackslashLine = (chars: number): string =>
  `[${"\\".repeat(1500)}]: `.repeat(Math.ceil(chars / 1504)).slice(0, chars);

const SAMPLES = [
  "# Title\n\npara with *em* and `code`\n\n```py\nprint('$$')\n```\n",
  "$$\na=b\n\n$$\n\nafter\n",
  "<div>\n\nhi\n\n</div>\n\nafter\n",
  "<details><summary>s</summary>\n\nbody\n\n</details>\n",
  "<br/>\n<div>\n\ninside\n\n</div>",
  "<my-tag>\n\nx\n\n</my-tag>\n",
  "| a | b |\n|---|---|\n| 1 | 2 |\n\n- x\n- y\n",
  "line\r\nwith crlf\r\n\r\nnext\r\n",
  "text[^1]\n\n[^1]: note\n",
  "C:\\Users\\a\\b \\* \\_ \\(x\\) \\[y\\]\n\n$$ unclosed\n",
  longBackslashLine(5_000),
];

test("block split matches streamdown", () => {
  for (const markdown of SAMPLES) {
    assert.deepEqual(
      parseMarkdownIntoBlocks(markdown),
      streamdownSplit(markdown),
    );
  }
});

test("block split stays fast on a long backslash line (#11376)", () => {
  const markdown = longBackslashLine(200_000);
  const start = performance.now();
  parseMarkdownIntoBlocks(markdown);
  const elapsed = performance.now() - start;
  assert.ok(elapsed < 500, `block split took ${elapsed.toFixed(1)}ms`);
});
