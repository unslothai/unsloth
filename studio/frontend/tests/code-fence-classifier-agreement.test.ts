// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getCodeFence } from "../src/features/chat/artifacts/html-fences.ts";
import { markdownBlockFallback } from "../src/components/assistant-ui/markdown-block-fallback.ts";

/**
 * The two classifiers have to agree about the same reply.
 *
 * A block is owned by `markdownBlockFallback` while it arrives and by `getCodeFence` the instant it
 * closes. Since `FenceBody` renders the body text each reports, rather than re-parsing the block as
 * streamdown's `Block` did, a disagreement is a block whose contents change under the reader at
 * that moment. `code-fence-close-run.test.ts` pins each against CommonMark; this pins them against
 * EACH OTHER, which neither file catches alone.
 *
 * Fences `getCodeFence` declines by design are skipped: there is no second opinion to compare.
 */

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

/** The first word, which is what both routes hand to the highlighter as the grammar. */
const grammarOf = (info: string | null): string | null =>
  info?.trim().split(/\s+/)[0] || null;

test("a fence's body is the same text while it streams and once it closes", () => {
  let checked = 0;
  for (const lang of LANGS) {
    for (const body of BODIES) {
      for (const closeRun of CLOSE_RUNS) {
        const opening = "```" + lang + "\n";
        // Exactly the two strings one growing block passes through.
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
  // `getCodeFence` hands back the WHOLE info string and `markdownBlockFallback` hands back the
  // first word, so these two are not equal and are not supposed to be. What must match is what
  // each route actually highlights with, which is the first word of either.
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
  // The specific shape that was broken, kept as its own row so a regression names itself rather
  // than arriving as one failure among a hundred table entries.
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
  /*
   * CommonMark closes a fence at the FIRST bare run on its own line; a match running past one has
   * swallowed the close, the prose after it and the next opener, and `FenceBody` renders all of
   * that as the reader's code. Expectations read off `micromark`'s HTML, not off this file: it
   * makes "```md\nA:\n```\nB\n```" a fence holding `A:`, a paragraph `B`, and an empty fence.
   */
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
  // The other side of the rule: an info-carrying run is not a close, so these stay single fences.
  assert.equal(
    getCodeFence("```js\nconst s = `x`;\n```")?.source,
    "const s = `x`;",
  );
  assert.equal(getCodeFence("```\nfoo `\n```")?.source, "foo `");
  assert.equal(getCodeFence("```\nfoo ``\n```")?.source, "foo ``");
  // A closing run longer than the opener still closes, and the body keeps its own backticks.
  assert.equal(getCodeFence("```python\nx = 1\n`````")?.source, "x = 1");
});

test("a run too short to close leaves the block streaming on both routes", () => {
  // A stream that has written only its first delimiter backtick must not read as closed.
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
