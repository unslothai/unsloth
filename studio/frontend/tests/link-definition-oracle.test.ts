// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `markdownRenderScope` decides whether a reply is lexed as one document or per block, and it
// answers with a cheap line scan rather than a second markdown parse -- the render already
// pays one in `parseMarkdownIntoBlocks`, and doubling that on every streaming chunk is the
// cost this path is written to avoid.
//
// A scan is an approximation of marked's block grammar, so the thing worth pinning is not that
// it agrees everywhere, but that it never errs in the direction that loses content. marked
// itself is the oracle here: `lexer.tokens.links` IS the document-wide definition map, so if
// marked registered a definition and the reply uses a reference, rendering per block splits
// the pair apart and the reference survives as literal `[label][ref]` text.
//
// Erring the other way -- one document where blocks would have done -- only costs that reply
// its per-block Copy/Download controls, which is what main did for EVERY reply containing a
// `]:` substring. So the asymmetry is deliberate and this file guards the expensive half.

import { marked } from "marked";
import assert from "node:assert/strict";
import test from "node:test";
import { markdownRenderScope } from "../src/components/assistant-ui/streaming-render-schedule.ts";

const REFERENCE_RE = /!?\[(?:\\.|[^\]\n\\]){1,200}\]\[(?:\\.|[^\]\n\\]){0,200}\]/;

function markedRegistersDefinition(markdown: string): boolean {
  const lexer = new marked.Lexer();
  lexer.lex(markdown);
  const links = (lexer as unknown as { tokens: { links?: object } }).tokens.links;
  return Object.keys(links ?? {}).length > 0;
}

function mustRenderAsDocument(markdown: string): boolean {
  return markedRegistersDefinition(markdown) && REFERENCE_RE.test(markdown);
}

const DEFINITION_CONTEXTS = [
  "[g]: /guide",
  "> [g]: /guide",
  ">> [g]: /guide",
  "> > [g]: /guide",
  "- [g]: /guide",
  "* [g]: /guide",
  "+ [g]: /guide",
  "1. [g]: /guide",
  "1) [g]: /guide",
  "- > [g]: /guide",
  "- - [g]: /guide",
  "> - [g]: /guide",
  "  [g]: /guide",
   "   [g]: /guide",
];

const NEUTRAL_BLOCKS = [
  "",
  "Some ordinary prose in between.",
  "```ts\ninterface G {\n  [key: string]: number[][];\n}\n```",
  "```css\na[href]:hover { color: red; }\n```",
  "```md\n[g]: /inside-a-fence\n```",
  "````\n```python\n````",
  "```text\n~~~\n```",
  "<pre>\n```\n</pre>",
  "<div>\n```\n</div>",
  "<!--\n```\n-->",
  "<?php\n```\n?>",
  "<![CDATA[\n```\n]]>",
  "    [g]: /indented-code-block",
  "| a | b |\n| - | - |\n| 1 | 2 |",
];

test("a reply marked lexes with a definition is never split into blocks", () => {
  const failures: string[] = [];
  for (const definition of DEFINITION_CONTEXTS) {
    for (const neutral of NEUTRAL_BLOCKS) {
      for (const order of [
        `See [guide][g].\n\n${neutral}\n\n${definition}\n`,
        `See [guide][g].\n\n${definition}\n\n${neutral}\n`,
      ]) {
        if (mustRenderAsDocument(order) && markdownRenderScope(order) !== "document") {
          failures.push(JSON.stringify(order));
        }
      }
    }
  }
  assert.deepEqual(
    failures,
    [],
    `these replies carry a definition marked registered document-wide, but the scan split ` +
      `them into blocks, so the reference renders as literal text:\n${failures.join("\n")}`,
  );
});

test("ordinary code is still rendered per block", () => {
  // The regression this whole path exists for: nothing here is a definition, so every one of
  // these must keep its per-block Copy code / Download file controls.
  for (const reply of [
    "Shape.\n\n```ts\ninterface G {\n  [key: string]: number[][];\n}\n\nconst c = grid[row][col];\n```\n",
    "Compare [one][two].\n\n```css\na[href]:hover { color: red; }\n```\n",
    "Compare [one][two].\n\n    [two]: not-a-definition\n",
    "How:\n\n```md\n[two]: https://example.com/two\n```\n\nText [one][two].\n",
  ]) {
    assert.equal(markdownRenderScope(reply), "blocks", reply);
    assert.equal(mustRenderAsDocument(reply), false, `oracle disagrees for:\n${reply}`);
  }
});
