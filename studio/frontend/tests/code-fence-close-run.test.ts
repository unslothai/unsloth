// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { getCodeFence } from "../src/features/chat/artifacts/html-fences.ts";

/** CommonMark close: same character, at least as many as the opener, so `{3,}`. */

const commonmarkSource = (markdown: string): string => {
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
  assert.equal(getCodeFence("```\nfoo `\n```\n")?.source, "foo `");
  assert.equal(getCodeFence("```\nfoo ``\n```\n")?.source, "foo ``");
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
  const four = getCodeFence("```python\nx = 1\n````\n");
  assert.equal(four?.source, "x = 1");
  assert.ok(
    !/```$/m.test(four?.source ?? ""),
    "no delimiter may be left inside the body",
  );
  const five = getCodeFence("```python\nx = 1\n`````\n");
  assert.equal(five?.source, "x = 1");
});

test("a run with no line break before it does not close the fence", () => {
  assert.equal(getCodeFence("```\nfoo````"), null);
  assert.equal(getCodeFence("```\nfoo```"), null);
  assert.equal(getCodeFence("```python\n````")?.source, "", "an empty fence still closes");
});

test("a run shorter than three does not close the fence", () => {
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
  assert.equal(getCodeFence("```\nfoo `\n```\n")?.source, "foo `");
  assert.equal(getCodeFence("```\nfoo ``\n```\n")?.source, "foo ``");
});
