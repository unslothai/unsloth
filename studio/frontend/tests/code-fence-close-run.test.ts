// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getCodeFence } from "../src/features/chat/artifacts/html-fences.ts";

/**
 * WHAT A CLOSING FENCE IS ALLOWED TO LOOK LIKE.
 *
 * CommonMark 0.31.2: the close uses the same character and "at least as many" of it as the opener,
 * so a model closing a fence whose body holds three backticks writes four. `{3}` matched that by
 * its LAST three and handed the surplus to `source`; a bare lazy `+?` is the other half of the rule
 * and accepts a one- or two-backtick run as a close, deleting real code. The opener is fixed at
 * three, so the close is `{3,}`.
 *
 * Each row names the answer `micromark` gives, so the expectation is CommonMark's and not this
 * file's.
 */

/** The code text `micromark` produces for a whole-block fence, with its trailing break removed. */
const commonmarkSource = (markdown: string): string => {
  // Read out of `micromark(markdown)` directly: the runner has no HTML parser, and a second parser
  // would be a second thing to disagree.
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
  // A run the BODY owns is code, not a delimiter, at both fence lengths.
  assert.equal(getCodeFence("```\nfoo `\n```\n")?.source, "foo `");
  assert.equal(getCodeFence("```\nfoo ``\n```\n")?.source, "foo ``");
  // Only the close's own run is eaten; anything shorter before it is the body's.
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
  // A short run is code, so the block is still open and the collapse must decline it.
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
