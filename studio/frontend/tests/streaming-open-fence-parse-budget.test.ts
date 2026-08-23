// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  IncrementalMarkdownCache,
  createCompletedCodeFenceCache,
  getCompletedCodeFences,
  hasFootnoteConstruct,
} from "../src/components/assistant-ui/streaming-render-schedule.ts";
import { getTerminalStreamingCodeFence } from "../src/components/assistant-ui/streaming-code-policy.ts";

import { renderBlockContents } from "./streaming-render-plan.ts";

/**
 * A fence that has opened and not closed is the whole live tail, and the tail is
 * re-lexed on every token. #9517 removed one whole-tail pass from the repair
 * path; two more were charging a whole-document CommonMark parse per chunk, one
 * to list the block's completed fences and one to locate the terminal fence.
 * Both are recomputing a constant while the fence stays open.
 *
 * These are complexity and equivalence tests, not timing tests. The shortcut has
 * to answer exactly what the parse answered, and it has to stop re-deriving the
 * settled part of the block.
 */

const codeLine = (index: number) =>
  `    result_${index} = transform(payload[${index}], mode="strict")`;

const fenceBody = (lines: number): string =>
  Array.from({ length: lines }, (_, index) => codeLine(index)).join("\n");

// A settled fence in front of the open one, so there is a result that must
// survive untouched while the open one grows.
const SETTLED = "```js\nconst a = 1;\n```\n\nNotes.\n\n```python\n";

test("a settled fence is carried forward rather than re-parsed on every chunk", () => {
  // `getCompletedCodeFences` allocates a fresh object per fence per call, so the
  // settled fence surviving by reference is proof the block was not re-parsed.
  const completedCodeFences = createCompletedCodeFenceCache();
  const settled = completedCodeFences(`${SETTLED}${fenceBody(1)}`, "block")[0];

  let previousOpen = null;
  for (let lines = 2; lines <= 400; lines += 1) {
    const fences = completedCodeFences(`${SETTLED}${fenceBody(lines)}`, "block");
    assert.equal(fences.length, 2);
    assert.equal(
      fences[0],
      settled,
      "the settled fence was re-derived, so the whole block was parsed again",
    );
    assert.notEqual(
      fences[1],
      previousOpen,
      "the open fence must still grow with the reply",
    );
    previousOpen = fences[1];
  }
});

test("the carried-forward result is what the parse would have produced", () => {
  const corpus = [
    SETTLED,
    "```python\n",
    "~~~\n",
    "````md\nnested ``` inside\n",
    "> quoted\n\n```py\n",
    "    indented code\n\n```py\n",
    "<div>\n```py\n</div>\n\n```py\n",
    "# Heading\n\n```py startLine=10\n",
    "```\n",
  ];
  const suffixes = [
    "x = 1",
    "x = 1\n",
    "x = 1\n\n",
    "x = 1\n```",
    "x = 1\n```\n",
    "x = 1\n```\n\nafter\n\n```js\ny",
    "x = 1\n~~~\n",
    "x = 1\n````\n",
  ];

  for (const head of corpus) {
    for (const suffix of suffixes) {
      // Stream it a character at a time, which is the only way the shortcut is
      // ever entered, and check every intermediate state against the parse.
      const completedCodeFences = createCompletedCodeFenceCache();
      const whole = `${head}${suffix}`;
      for (let end = 1; end <= whole.length; end += 1) {
        const content = whole.slice(0, end);
        assert.deepEqual(
          completedCodeFences(content, "block"),
          getCompletedCodeFences(content, "block"),
          `diverged at ${JSON.stringify(content)}`,
        );
      }
    }
  }
});

test("a rewritten or shortened block falls back to the parse", () => {
  const completedCodeFences = createCompletedCodeFenceCache();
  completedCodeFences(`${SETTLED}${fenceBody(20)}`, "block");
  for (const content of [
    `${SETTLED}${fenceBody(3)}`,
    `Rewritten.\n\n${SETTLED}${fenceBody(20)}`,
    SETTLED,
  ]) {
    assert.deepEqual(
      completedCodeFences(content, "block"),
      getCompletedCodeFences(content, "block"),
    );
  }
  // A different block never reads another block's memo.
  assert.deepEqual(
    completedCodeFences(`${SETTLED}${fenceBody(21)}`, "other"),
    getCompletedCodeFences(`${SETTLED}${fenceBody(21)}`, "other"),
  );
});

test("the terminal fence offset shortcut is actually taken", () => {
  // Inside an HTML block CommonMark calls this fence HTML content, so the
  // whole-tail parse answers null while the line scan the shortcut uses answers
  // the fence. The cache never hints such an offset -- it only ever passes one a
  // parse produced -- so this input exists purely to make a disabled shortcut
  // visible instead of silently costing a parse per token again.
  const inHtmlBlock = "<div>\n```py\nx = 1";
  assert.equal(getTerminalStreamingCodeFence(inHtmlBlock), null);
  assert.equal(
    getTerminalStreamingCodeFence(inHtmlBlock, 6)?.openingOffset,
    6,
    "the offset hint is being ignored, so every chunk pays for the parse",
  );
});

test("the terminal fence offset shortcut agrees with the parse", () => {
  const documents = [
    "```python\nx = 1",
    "```python\nx = 1\n```",
    "```python\nx = 1\n```\ntrailing prose",
    "Intro.\n\n```python\nx = 1",
    "Intro.\n\n```python\nx = 1\n```\n",
    "<div>\n```py\n</div>\n\n```py\nx",
    "    ```py\n    x = 1",
    "- item\n\n  ```py\n  x = 1",
  ];
  // The hint the cache actually passes: the offset the previous chunk's parse
  // found, while that fence was open. Stream each document and check every step.
  for (const document of documents) {
    let hint: number | undefined;
    for (let end = 1; end <= document.length; end += 1) {
      const content = document.slice(0, end);
      const parsed = getTerminalStreamingCodeFence(content);
      assert.deepEqual(
        getTerminalStreamingCodeFence(content, hint),
        parsed,
        `hint ${hint} changed the answer for ${JSON.stringify(content)}`,
      );
      hint = parsed && !parsed.isClosed ? parsed.openingOffset : undefined;
    }
  }
});

test("streaming an unterminated fence does not re-derive the settled prefix", () => {
  // The whole-cache view of the same property. The terminal tail's prefix blocks
  // are already memoised; its identity and the committed chunks must hold too,
  // and the plan must still describe the exact reply.
  const text = `Intro.\n\n\`\`\`python\ndef process(payload):\n${fenceBody(400)}`;
  const cache = new IncrementalMarkdownCache();
  let render = cache.update(text.slice(0, 64));
  const identity = () => render.terminalCodeTail?.id;
  let firstIdentity: string | undefined;
  for (let end = 128; end < text.length; end += 128) {
    render = cache.update(text.slice(0, end));
    firstIdentity ??= identity();
    assert.equal(
      identity(),
      firstIdentity,
      "the open fence must keep one identity for its whole life",
    );
    assert.equal(render.terminalCodeTail?.isClosed, false);
  }
  render = cache.update(text);
  assert.equal(
    render.terminalCodeTail?.source,
    text.slice(text.indexOf("\n", text.indexOf("```")) + 1),
  );
});

/**
 * A character class in ordinary code is shaped exactly like a footnote
 * reference. Reading one as a footnote switches the cache to full-document
 * mode, which is sticky, so every later token re-repairs and re-lexes the whole
 * reply: the same quadratic, restored by one line of the reply's own code.
 */
test("a character class inside a fence is code, not a footnote", () => {
  for (const line of [
    "  slug = re.sub(/[^a-z]/g, '-', name);",
    "  if (/[^0-9]/.test(value)) return null;",
    "  grep -v '[^abc]' file.txt",
  ]) {
    assert.equal(
      hasFootnoteConstruct(`\`\`\`js\n${line}\n`),
      false,
      `read as a footnote inside an open fence: ${line}`,
    );
    assert.equal(
      hasFootnoteConstruct(`\`\`\`js\n${line}\n\`\`\`\n`),
      false,
      `read as a footnote inside a closed fence: ${line}`,
    );
    // The same characters outside a fence really are a footnote reference.
    assert.equal(hasFootnoteConstruct(line), true, line);
  }
});

test("real footnotes are still found outside fenced code", () => {
  for (const markdown of [
    "See the note[^1] for detail.",
    "[^1]: the definition\n",
    "```js\nconst x = 1;\n```\n\nAfter the fence[^note].",
    "~~~\ncode\n~~~\n\n[^a]: after a tilde fence\n",
    "````\n```js\nnested\n```\n````\n\nOutside[^x].",
  ]) {
    assert.equal(hasFootnoteConstruct(markdown), true, markdown);
  }
  for (const markdown of ["plain text", "```\n[^1]: inside\n```\n", ""]) {
    assert.equal(hasFootnoteConstruct(markdown), false, markdown);
  }
});

test("a fence holding a character class keeps the incremental plan", () => {
  // Entering full-document mode resets the incremental state, which bumps the
  // epoch. So a flat epoch across the stream is exactly the property at issue.
  const body = Array.from({ length: 400 }, (_, index) =>
    index === 3
      ? "  slug = re.sub(/[^a-z]/g, '-', name);"
      : `  ${codeLine(index)}`,
  ).join("\n");
  const text = `Intro.\n\n\`\`\`js\nfunction process() {\n${body}`;

  const cache = new IncrementalMarkdownCache();
  let render = cache.update(text.slice(0, 64));
  for (let end = 128; end < text.length; end += 128) {
    render = cache.update(text.slice(0, end));
    assert.equal(
      render.epoch,
      0,
      "the reply's own regex literal dropped the retained prefix",
    );
  }
  render = cache.update(text);
  assert.equal(render.epoch, 0);
  assert.equal(renderBlockContents(render).join(""), render.markdown);
});
