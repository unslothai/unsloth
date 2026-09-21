// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getCodeFence } from "../src/features/chat/artifacts/html-fences.ts";
import { markdownBlockFallback } from "../src/components/assistant-ui/markdown-block-fallback.ts";

/**
 * THE TWO CLASSIFIERS HAVE TO AGREE ABOUT THE SAME REPLY.
 *
 * One block is owned by `markdownBlockFallback` while it is still arriving and by `getCodeFence`
 * the instant its closing delimiter lands, because only the second can see a close. Since
 * `FenceBody` renders the body TEXT each one reports, rather than re-parsing the block the way
 * streamdown's `Block` did, a disagreement between them is not academic: it is a block whose
 * contents change under a reader at the moment the model finishes writing it.
 *
 * That is not hypothetical. `CODE_FENCE_RE` used to require exactly three closing backticks and
 * matched a longer run by its LAST three, so a fence closed with four handed the surplus backtick
 * to `source` while the streaming scanner, which already implemented CommonMark's "at least as
 * many" rule, did not. `code-fence-close-run.test.ts` pins each classifier against CommonMark
 * separately. This file pins them against EACH OTHER, which is the invariant the rendering
 * actually depends on and the one neither file would catch alone.
 *
 * Fences that `getCodeFence` declines by design are skipped rather than asserted: it claims only
 * unindented three-backtick fences, and where it declines, the settled block falls to streamdown's
 * own parser and there is no second opinion to compare.
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

test("a run too short to close leaves the block streaming on both routes", () => {
  // The other direction: a stream that has written only its first delimiter backtick must not be
  // read as a completed fence by either classifier, or the block settles early and then re-opens.
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
