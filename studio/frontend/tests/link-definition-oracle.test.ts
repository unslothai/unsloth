// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `markdownRenderScope` decides whether a reply is lexed as one document or per block, and it
// answers with a cheap line scan rather than a second markdown parse -- the render already
// pays one inside `parseMarkdownIntoBlocks`, and doubling that on every streaming chunk is the
// cost this path is written to avoid.
//
// A scan is an approximation of the block grammar, so what is worth pinning is not that it
// agrees everywhere, but that it never errs in the direction that loses content. The oracle
// below is not a second opinion about markdown; it is the rendered outcome, built from the
// two pieces production actually uses -- streamdown's `parseMarkdownIntoBlocks` for the split
// and the remark pipeline for the render, the same shape as math-block-marker-pipeline.test.ts.
// Deliberately no `marked` import: the frontend does not depend on marked directly, and a bare
// specifier resolves to the copy hoisted for mermaid rather than the one streamdown splits
// with, so an oracle built on it would be measuring a different lexer from the renderer.
//
// The two errors are not symmetric:
//   blocks when the reply needed one document -> the reference/definition pair is split apart
//     and the reference survives as literal `[label][ref]` text. Content is lost.
//   one document when blocks would have done -> that reply loses its per-block Copy code and
//     Download file controls. Nothing is lost from the content, and it is what main did for
//     EVERY reply containing a `]:` substring, which is what this path set out to narrow.
// So this file guards the expensive half, exhaustively.

import assert from "node:assert/strict";
import test from "node:test";

import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { parseMarkdownIntoBlocks } from "streamdown";
import { unified } from "unified";

import { markdownRenderScope } from "../src/components/assistant-ui/streaming-render-schedule.ts";

const pipeline = unified().use(remarkParse).use(remarkGfm).use(remarkRehype);

function anchorCount(markdown: string): number {
  const tree = pipeline.runSync(pipeline.parse(markdown)) as unknown;
  let found = 0;
  const walk = (node: unknown): void => {
    const element = node as {
      tagName?: string;
      properties?: { href?: unknown };
      children?: unknown[];
    };
    if (element.tagName === "a" && element.properties?.href !== undefined) {
      found += 1;
    }
    for (const child of element.children ?? []) {
      walk(child);
    }
  };
  walk(tree);
  return found;
}

// The whole point of the document path: how many references resolve when the reply is lexed
// in one piece, versus when streamdown splits it first.
const asOneDocument = (markdown: string): number => anchorCount(markdown);
const asBlocks = (markdown: string): number =>
  parseMarkdownIntoBlocks(markdown).reduce(
    (total, block) => total + anchorCount(block),
    0,
  );

const DEFINITION_CONTEXTS = [
  "[g]: /guide",
  "  [g]: /guide",
  "   [g]: /guide",
  "> [g]: /guide",
  ">> [g]: /guide",
  "> > [g]: /guide",
  "> > > > > > > [g]: /guide",
  "- [g]: /guide",
  "* [g]: /guide",
  "+ [g]: /guide",
  "1. [g]: /guide",
  "1) [g]: /guide",
  "- > [g]: /guide",
  "- - [g]: /guide",
  "> - [g]: /guide",
  "- > - > [g]: /guide",
];

const NEUTRAL_BLOCKS = [
  "",
  "Some ordinary prose in between.",
  "```ts\ninterface G {\n  [key: string]: number[][];\n}\n```",
  "```css\na[href]:hover { color: red; }\n```",
  "```md\n[g]: /inside-a-fence\n```",
  "````\n```python\n````",
  "```text\n~~~\n```",
  "```ts\nconst x = 1;\n``` \n```",
  "<pre>\n```\n</pre>",
  "<div>\n```\n</div>",
  "<center>\n```\n",
  "<summary>\n```\n",
  "<div>\n \n```\n",
  "<!--\n```\n-->",
  "<?php\n```\n?>",
  "<![CDATA[\n```\n]]>",
  "<!DOCTYPE html>",
  "    [g]: /indented-code-block",
  "| a | b |\n| - | - |\n| 1 | 2 |",
];

test("a reply whose reference only resolves in one piece is never split into blocks", () => {
  const failures: string[] = [];
  for (const definition of DEFINITION_CONTEXTS) {
    for (const neutral of NEUTRAL_BLOCKS) {
      for (const reply of [
        `See [guide][g].\n\n${neutral}\n\n${definition}\n`,
        `See [guide][g].\n\n${definition}\n\n${neutral}\n`,
      ]) {
        // Only replies that actually lose an anchor when split are in scope here.
        if (asOneDocument(reply) <= asBlocks(reply)) {
          continue;
        }
        if (markdownRenderScope(reply) !== "document") {
          failures.push(JSON.stringify(reply));
        }
      }
    }
  }
  assert.deepEqual(
    failures,
    [],
    "these replies resolve their reference only when rendered in one piece, but the scan " +
      `split them into blocks, so the reference renders as literal text:\n${failures.join("\n")}`,
  );
});

test("ordinary code is still rendered per block", () => {
  // The regression this path exists for: nothing here is a definition, so each of these must
  // keep its per-block Copy code / Download file controls.
  for (const reply of [
    "Shape.\n\n```ts\ninterface G {\n  [key: string]: number[][];\n}\n\nconst c = grid[row][col];\n```\n",
    "Compare [one][two].\n\n```css\na[href]:hover { color: red; }\n```\n",
    "Compare [one][two].\n\n    [two]: not-a-definition\n",
    "How:\n\n```md\n[two]: https://example.com/two\n```\n\nText [one][two].\n",
  ]) {
    assert.equal(markdownRenderScope(reply), "blocks", reply);
    assert.equal(
      asOneDocument(reply) > asBlocks(reply),
      false,
      `this reply does lose an anchor when split, so it belongs in the guard above:\n${reply}`,
    );
  }
});
