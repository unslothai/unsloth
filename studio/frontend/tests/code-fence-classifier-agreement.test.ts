// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getCodeFence } from "../src/features/chat/artifacts/html-fences.ts";
import { markdownBlockFallback } from "../src/components/assistant-ui/markdown-block-fallback.ts";

const BODIES = [
  "x = 1",
  "a\nb",
  "",
  "foo `",
  "foo ``",
  "const a = `x`;",
  "  indented",
  "}\n",
];
const LANGS = ["", "ts", "typescript", "python startLine=10"];
const CLOSE_RUNS = [3, 4, 5, 6];

const grammarOf = (info: string | null): string | null =>
  info?.trim().split(/\s+/)[0] || null;

test("a fence's body is the same text while it streams and once it closes", () => {
  let checked = 0;
  for (const lang of LANGS) {
    for (const body of BODIES) {
      for (const closeRun of CLOSE_RUNS) {
        const opening = "```" + lang + "\n";
        const streaming = opening + body;
        const closed = `${opening}${body}\n${"`".repeat(closeRun)}`;

        const before = markdownBlockFallback(streaming);
        const after = getCodeFence(closed);
        assert.ok(
          before.fenced,
          `${JSON.stringify(streaming)} must be recognised as a fence while it streams, or the ` +
            "block falls back to the renderer this change exists to keep it away from",
        );
        if (after === null) continue;
        checked++;
        assert.equal(
          before.text,
          after.source,
          "the body a reader is watching must not change when the closing delimiter lands " +
            `(${JSON.stringify(closed)})`,
        );
      }
    }
  }
  assert.ok(checked > 50, `expected the table to exercise the handover, got ${checked}`);
});

test("the grammar does not change when the closing delimiter lands", () => {
  // getCodeFence returns the whole info string, the fallback only the first word; compare the
  // grammar each highlights with.
  for (const info of ["", "ts", "python startLine=10", "typescript  extra  bits"]) {
    const streaming = "```" + info + "\nx = 1";
    const closed = "```" + info + "\nx = 1\n```";
    const before = markdownBlockFallback(streaming);
    const after = getCodeFence(closed);
    assert.ok(after, "closed fence must parse");
    assert.equal(
      grammarOf(before.language),
      grammarOf(after.language),
      `the highlighter's grammar must survive the handover (info ${JSON.stringify(info)})`,
    );
  }
});

test("a closing run longer than the opener agrees across the handover", () => {
  for (const closeRun of [4, 5, 6]) {
    const closed = "```python\nx = 1\n" + "`".repeat(closeRun);
    const after = getCodeFence(closed);
    assert.equal(after?.source, "x = 1", `a ${closeRun}-backtick close must not leak delimiters`);
    assert.equal(
      markdownBlockFallback("```python\nx = 1").text,
      after?.source,
      "and the streaming scanner must already have said the same thing",
    );
  }
});

test("a block holding more than one fence is claimed by neither classifier", () => {
  /* CommonMark closes a fence at the first bare run on its own line; expectations from micromark. */
  for (const block of [
    "```md\nA:\n```\nB\n```",
    "```js\ncode\n```\n```",
    "```md\nUse this:\n```js\nx\n```\nDone.\n```",
    "```js\ncode\n```\n\nprose\n\n```py\nmore\n```",
  ]) {
    assert.equal(
      getCodeFence(block),
      null,
      `a multi-fence block is not one fence: ${JSON.stringify(block)}`,
    );
    assert.equal(
      markdownBlockFallback(block).fenced,
      false,
      `and the streaming scanner must say the same: ${JSON.stringify(block)}`,
    );
  }
});

test("a delimiter line that carries an info string is still code", () => {
  assert.equal(
    getCodeFence("```js\nconst s = `x`;\n```")?.source,
    "const s = `x`;",
  );
  assert.equal(getCodeFence("```\nfoo `\n```")?.source, "foo `");
  assert.equal(getCodeFence("```\nfoo ``\n```")?.source, "foo ``");
  assert.equal(getCodeFence("```python\nx = 1\n`````")?.source, "x = 1");
});

test("a run too short to close leaves the block streaming on both routes", () => {
  for (const short of ["`", "``"]) {
    const text = `\`\`\`python\nx = 1\n${short}`;
    assert.equal(
      getCodeFence(text),
      null,
      `${JSON.stringify(short)} is code, not a closing delimiter`,
    );
    assert.equal(
      markdownBlockFallback(text).fenced,
      true,
      "and the block is still an open fence, so it keeps streaming",
    );
  }
});

test("an indented fence loses its indentation by COLUMNS, so a tab is not one character", () => {
  /* CommonMark strips opener indentation by column (4-column tab stops); per micromark. */
  const rows: [string, string][] = [
    [" ```js\n\tx\n ```", "   x"],
    ["  ```js\n\tx\n  ```", "  x"],
    ["   ```js\n\tx\n   ```", " x"],
    ["```js\n\tx\n```", "\tx"],
    ["   ```js\n \tx\n   ```", " x"],
    ["  ```js\n    x\n  ```", "  x"],
    ["  ```js\n x\n  ```", "x"],
  ];
  for (const [block, expected] of rows) {
    assert.equal(
      markdownBlockFallback(block).text,
      expected,
      `indentation must match CommonMark for ${JSON.stringify(block)}`,
    );
  }
});
