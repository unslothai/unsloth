// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getCodeFence } from "../src/features/chat/artifacts/html-fences.ts";

/**
 * WHAT A CLOSING FENCE IS ALLOWED TO LOOK LIKE.
 *
 * CommonMark 0.31.2: "The closing code fence ... may be followed only by spaces or tabs" and its
 * run must be "at least as many" of the fence's character as the opening one. A model therefore
 * closes a fence whose body contains a three-backtick run with FOUR backticks -- the shape the
 * `getCodeFence` regex is the only remaining consumer of, since the streaming route reads the
 * grammar-complete scanner in `markdown-block-fallback.ts`.
 *
 * The regex matched such a line by its LAST three backticks and handed the surplus delimiter to
 * `source`: a fence opened with three and closed with four became the code ``x = 1\n` ``, one line
 * longer and one character wider than what CommonMark says it holds. Only a COMPLETED block comes
 * through here, so a reader watching a reply stream saw the block they were reading grow a stray
 * backtick the moment its closing delimiter landed.
 *
 * The other direction matters just as much: a bare lazy run would accept ONE or TWO backticks as a
 * close, so a stream that had written only its first delimiter backtick would already parse as a
 * completed fence with that backtick deleted, and an aborted reply ending in a short backtick run
 * would be permanently misread. A run shorter than the opener is code, so the close is `{3,}` --
 * "at least as many" as the opener, which is three.
 *
 * These rows are RUN rather than described, and each one names the parser's answer so the
 * expectation is CommonMark's and not this file's.
 */

/** The code text `micromark` produces for a whole-block fence, with its trailing break removed. */
const commonmarkSource = (markdown: string): string => {
  // Hard-coded rather than imported: the frontend test runner has no HTML parser, and a second
  // parser in the runner would be a second thing to disagree. Each value below was read out of
  // `micromark(markdown)` directly.
  const expected: Record<string, string> = {
    "```python\nx = 1\n```\n": "x = 1",
    "```python\nx = 1\n````\n": "x = 1",
    "```python\nx = 1\n`````\n": "x = 1",
    "```python\nx = 1\n``````\n": "x = 1",
    "```python\nx = 1\n```": "x = 1",
    "```python\nx = 1\n````": "x = 1",
    "```python\na = 1\nb = 2\n````\n": "a = 1\nb = 2",
    "```python\n\n\n````\n": "\n",
    "```python\n````\n": "",
    "```python\n```\n": "",
    "```python startLine=10\nx = 1\n````\n": "x = 1",
    "```\nfoo `\n```\n": "foo `",
    "```\nfoo ``\n```\n": "foo ``",
    "```python\r\nx = 1\r\n````\r\n": "x = 1",
  };
  return expected[markdown];
};

test("every closing run a model writes yields the body CommonMark gives it", () => {
  for (const [markdown, expected] of Object.entries({
    "```python\nx = 1\n```\n": "x = 1",
    "```python\nx = 1\n````\n": "x = 1",
    "```python\nx = 1\n`````\n": "x = 1",
    "```python\nx = 1\n``````\n": "x = 1",
    "```python\nx = 1\n```": "x = 1",
    "```python\nx = 1\n````": "x = 1",
    "```python\na = 1\nb = 2\n````\n": "a = 1\nb = 2",
    "```python\n\n\n````\n": "\n",
    "```python\n````\n": "",
    "```python\n```\n": "",
    "```python startLine=10\nx = 1\n````\n": "x = 1",
    "```\nfoo `\n```\n": "foo `",
    "```\nfoo ``\n```\n": "foo ``",
    "```python\r\nx = 1\r\n````\r\n": "x = 1",
  })) {
    const fence = getCodeFence(markdown);
    assert.ok(fence, `${JSON.stringify(markdown)} must still parse as a fence`);
    assert.equal(
      fence.source,
      expected,
      `the surplus delimiter of a longer closing run must not reach source (${JSON.stringify(markdown)})`,
    );
    assert.equal(
      fence.source,
      commonmarkSource(markdown),
      `source must be what CommonMark says it is (${JSON.stringify(markdown)})`,
    );
  }
});

test("a body's OWN trailing backticks survive", () => {
  // The close is lazy, so a run the body owns is not mistaken for the delimiter: three backticks
  // inside a three-backtick fence are not a close, and the line above a four-backtick close may
  // end in backticks of its own.
  assert.equal(getCodeFence("```\nfoo `\n```\n")?.source, "foo `");
  assert.equal(getCodeFence("```\nfoo ``\n```\n")?.source, "foo ``");
  // A body ending in a run SHORTER than the close keeps all of it: the close is the last three and
  // the two before them are the code's.
  assert.equal(
    getCodeFence("```\nfoo `````\n`````\n")?.source,
    "foo `````",
    "a body line of backticks is code when the closing run is longer",
  );
});

test("the language tag and the block-level shapes are unchanged", () => {
  assert.equal(getCodeFence("```python\nx = 1\n```\n")?.language, "python");
  assert.equal(
    getCodeFence("```python startLine=10\nx = 1\n````\n")?.language,
    "python startLine=10",
    "the whole info string is still handed back; the caller takes its first word",
  );
  assert.equal(getCodeFence("~~~python\nx = 1\n~~~\n"), null, "tildes are not this regex's form");
  assert.equal(getCodeFence("   ```python\n   x = 1\n   ```\n"), null, "nor is an indented fence");
  assert.equal(
    getCodeFence("```python\nx = 1"),
    null,
    "an open fence has no close, which is what routes it to the streaming branch",
  );
  assert.equal(getCodeFence("plain prose"), null);
});

test("the close is at least three, not exactly three", () => {
  // The regression in one row: `{3}` matched the last three of a four-backtick close.
  const four = getCodeFence("```python\nx = 1\n````\n");
  assert.equal(four?.source, "x = 1");
  assert.ok(
    !/```$/m.test(four?.source ?? ""),
    "no delimiter may be left inside the body",
  );
  const five = getCodeFence("```python\nx = 1\n`````\n");
  assert.equal(five?.source, "x = 1");
});

test("a run shorter than three does not close the fence", () => {
  // A one- or two-backtick run is CODE, so the block is still open and the collapse must decline
  // it -- exactly what `micromark` says, and what the streaming route already did.
  for (const short of ["`", "``"]) {
    const markdown = `\`\`\`python\nx = 1\n${short}`;
    assert.equal(
      getCodeFence(markdown),
      null,
      `${JSON.stringify(markdown)} is not a closed fence; the short run is code, not a delimiter`,
    );
  }
  assert.equal(
    getCodeFence("```python\nx = 1\n`\n`"),
    null,
    "two separate one-backtick lines are still not a close",
  );
  // The same run one line higher is a body line, and must survive both a short and a long close.
  assert.equal(getCodeFence("```\nfoo `\n```\n")?.source, "foo `");
  assert.equal(getCodeFence("```\nfoo ``\n```\n")?.source, "foo ``");
});
