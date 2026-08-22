// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import ts from "typescript";

import {
  markdownBlockFallback,
} from "../src/components/assistant-ui/markdown-block-fallback.ts";

/**
 * Streamdown loads the syntax highlighted code body and the Mermaid renderer
 * through `React.lazy`, and it fetches them the first time a reply contains
 * that construct. A rejected import rethrows during render. Before the boundary
 * these tests protect, the nearest catcher was TanStack Router's, so ONE chunk
 * that would not load replaced all of Studio with "Something went wrong!",
 * unmounted the assistant-ui runtime with it, and left the reply's stream with
 * nothing consuming it.
 *
 * Measured on an unmodified tree by aborting exactly that one request: the
 * document went from 122 elements to 21, the pane's readable text went from
 * 11,968 characters to 0, and the stream stopped at 720 of 12,000 characters
 * and never resumed.
 *
 * So what these tests hold is not "the app does not crash". It is that the
 * READER STILL HAS THE CONTENT. A degraded fence has to be the same characters
 * in the same order, because that is the answer they asked for, and an error
 * card or an empty box in its place is a worse outcome than missing colour.
 */

const FENCE_BODY = [
  "def score(rows, cap):",
  "    total = 0.0",
  "",
  "    for row in rows:",
  "        total += min(cap, row.weight)",
  "    return total",
].join("\n");

test("a fenced block degrades to the code itself, without the fence scaffolding", () => {
  const fallback = markdownBlockFallback("```python\n" + FENCE_BODY + "\n```");

  assert.equal(
    fallback.text,
    FENCE_BODY,
    "the degraded fence is not the code it was carrying, so the reader lost the answer",
  );
  assert.equal(fallback.language, "python");
  assert.equal(
    fallback.fenced,
    true,
    "a fence that is not reported as fenced renders as prose, so its indentation and line breaks collapse",
  );
});

test("the degraded fence keeps every character and every blank line", () => {
  const fallback = markdownBlockFallback("```python\n" + FENCE_BODY + "\n```");

  assert.equal(
    fallback.text.split("\n").length,
    FENCE_BODY.split("\n").length,
    "the degraded fence lost a line, and a blank line inside a function is not decoration",
  );
  assert.ok(
    fallback.text.includes("    total = 0.0"),
    "the degraded fence lost its leading whitespace, which in python is the program",
  );
});

test("a fence that is still arriving degrades too", () => {
  // The failure happens MID-fence: the chunk is fetched when the fence first
  // renders, which is long before it closes. A fallback that only handled
  // closed fences would show backticks at exactly the moment it is needed.
  const fallback = markdownBlockFallback("```python\n" + FENCE_BODY);

  assert.equal(
    fallback.text,
    FENCE_BODY,
    "an unclosed fence was not recognised, so a reader sees the opening backticks and the language tag as text",
  );
  assert.equal(fallback.fenced, true);
});

test("a fence with no language tag still degrades to its body", () => {
  const fallback = markdownBlockFallback("```\n" + FENCE_BODY + "\n```");

  assert.equal(fallback.text, FENCE_BODY);
  assert.equal(
    fallback.language,
    null,
    "an absent language tag has to be absent, not the empty string, or the header renders blank",
  );
});

test("a tilde fence degrades the same way", () => {
  const fallback = markdownBlockFallback("~~~ts\nconst a = 1;\n~~~");

  assert.equal(fallback.text, "const a = 1;");
  assert.equal(fallback.language, "ts");
});

test("prose is handed back unchanged and is not treated as a fence", () => {
  const prose = "The scorer skips empty rows, then applies the cap.";
  const fallback = markdownBlockFallback(prose);

  assert.equal(
    fallback.text,
    prose,
    "a paragraph was rewritten on its way to the fallback",
  );
  assert.equal(
    fallback.fenced,
    false,
    "prose rendered as a code block is a visible change to a reply that had nothing wrong with it",
  );
});

test("a closing fence longer than the opening one still closes the block", () => {
  // CommonMark 0.31.2 requires the close to carry AT LEAST as many characters
  // as the open, so a four-backtick close is legitimate and is how a model
  // closes a fence whose body contains a three-backtick one. Demanding the same
  // run left the close on screen as though it were the last line of the code.
  const fallback = markdownBlockFallback("```python\n" + FENCE_BODY + "\n````");

  assert.equal(
    fallback.text,
    FENCE_BODY,
    "a longer closing fence was not recognised, so the reader sees stray backticks below their code",
  );
  assert.equal(fallback.language, "python");
  assert.equal(fallback.fenced, true);
});

test("a fence closed by a longer run keeps an inner fence in the body", () => {
  // The reason the rule exists: the four-backtick fence is carrying a
  // three-backtick one, which has to survive as content.
  const fallback = markdownBlockFallback("````md\n```py\nx = 1\n```\n````");

  assert.equal(fallback.text, "```py\nx = 1\n```");
  assert.equal(fallback.language, "md");
});

test("an empty fence degrades to nothing, not to its own closing backticks", () => {
  const fallback = markdownBlockFallback("```\n```");

  assert.equal(
    fallback.text,
    "",
    "the closing fence was returned as the body, so an empty code block renders ``` as if the model had written it",
  );
  assert.equal(fallback.fenced, true);
});

test("a block with content never degrades to nothing", () => {
  for (const content of [
    "```python\nx = 1\n```",
    "plain",
    "| a | b |\n|---|---|\n| 1 | 2 |",
  ]) {
    const fallback = markdownBlockFallback(content);
    assert.ok(
      fallback.text.length > 0,
      `a non-empty block degraded to an empty string: ${JSON.stringify(content)}`,
    );
  }
});

const MARKDOWN_TEXT_PATH = new URL(
  "../src/components/assistant-ui/markdown-text.tsx",
  import.meta.url,
);
const source = ts.createSourceFile(
  MARKDOWN_TEXT_PATH.pathname,
  readFileSync(MARKDOWN_TEXT_PATH, "utf8"),
  ts.ScriptTarget.ESNext,
  true,
  ts.ScriptKind.TSX,
);

/** The JSX element names that wrap `StreamdownBlockContent` where it is rendered. */
function wrappersAroundBlockContent(): string[] {
  const wrappers: string[] = [];
  const visit = (node: ts.Node, open: string[]): void => {
    if (
      ts.isJsxSelfClosingElement(node) &&
      node.tagName.getText(source) === "StreamdownBlockContent"
    ) {
      wrappers.push(...open);
    }
    const next =
      ts.isJsxElement(node)
        ? [...open, node.openingElement.tagName.getText(source)]
        : open;
    node.forEachChild((child) => visit(child, next));
  };
  source.forEachChild((node) => visit(node, []));
  return wrappers;
}

test("every markdown block is rendered inside the boundary", () => {
  // A source check because no output test can tell them apart: an unwrapped tree
  // renders identically until a lazy chunk fails, and then it takes the whole
  // application with it. This is the only thing standing between that and a
  // quiet revert.
  assert.ok(
    wrappersAroundBlockContent().includes("MarkdownBlockBoundary"),
    "the block component is rendered outside MarkdownBlockBoundary, so a fence whose highlighter fails to load unmounts all of Studio through the router's error boundary again",
  );
});

test("the boundary does not retry the import it caught", () => {
  const boundary = readFileSync(
    new URL(
      "../src/components/assistant-ui/markdown-block-boundary.tsx",
      import.meta.url,
    ),
    "utf8",
  );

  // React and the browser's module map both cache a rejected dynamic import
  // (whatwg/html#6768), so a boundary that resets on new props rethrows on every
  // frame of a streaming reply and issues no new request for its trouble.
  assert.ok(
    !boundary.includes("getDerivedStateFromProps"),
    "the boundary resets itself from props, which on a streaming reply means throwing and catching on every chunk for an import that can never succeed again",
  );
});
