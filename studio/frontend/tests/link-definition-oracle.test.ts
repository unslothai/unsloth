// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// `markdownRenderScope` uses a cheap line scan instead of a second markdown parse. It may err
// toward one document (losing per-block controls) but must never split a reference from its
// definition (losing content). The oracle is the rendered outcome from streamdown's
// parseMarkdownIntoBlocks plus remark; no `marked` import, since a bare specifier resolves
// to the copy hoisted for mermaid, not streamdown's.

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
  // Four columns absolute, but flush with the content column `10. ` opened.
  ">  [g]: /guide",
  ">   [g]: /guide",
  "10. item\n\n    [g]: /guide",
  "1. item\n\n   [g]: /guide",
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
  "\t[g]: /tab-indented-code-block",
  "-[g]: /no-space-is-not-a-list",
  "1.[g]: /no-space-is-not-a-list",
  "| a | b |\n| - | - |\n| 1 | 2 |",
  // A bare custom tag opens a type-7 HTML block, and raw HTML nests inside a list item.
  "<x>\n```\n",
  "<my-widget>\n```\n",
  "- <pre>\n  ```\n  </pre>",
  "> ```ts\n> [g]: number\n> ```",
  // A backtick opener carrying a backtick in its info string is not a fence at all.
  "```bad` still prose",
  "````bad` still prose",
  // A tilde opener has no such rule, so this one really is a fence.
  "~~~bad` really a fence\n~~~",
];

test("a reply whose reference only resolves in one piece is never split into blocks", () => {
  const failures: string[] = [];
  for (const definition of DEFINITION_CONTEXTS) {
    for (const neutral of NEUTRAL_BLOCKS) {
      for (const reply of [
        `See [guide][g].\n\n${neutral}\n\n${definition}\n`,
        `See [guide][g].\n\n${definition}\n\n${neutral}\n`,
      ]) {
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

test("a reference is never split into blocks because its label is long", () => {
  // Every case above uses a one-character label; this varies length. Past 999 drops out of scope.
  const failures: string[] = [];
  let inScope = 0;
  for (const length of [1, 2, 199, 200, 201, 400, 998, 999, 1000, 1001, 2000]) {
    const label = "L".repeat(length);
    for (const reply of [
      `See [guide][${label}].\n\nplain prose between them\n\n[${label}]: /guide\n`,
      `[${label}]: /guide\n\nplain prose between them\n\nSee [guide][${label}].\n`,
    ]) {
      if (asOneDocument(reply) <= asBlocks(reply)) {
        continue;
      }
      inScope += 1;
      if (markdownRenderScope(reply) !== "document") {
        failures.push(`label length ${length}`);
      }
    }
  }
  assert.ok(inScope > 0, "no length lost an anchor when split, so this test proved nothing");
  assert.deepEqual(
    failures,
    [],
    "these labels resolve their reference only when the reply is rendered in one piece, but the " +
      `scan split them into blocks, so the reference renders as literal text: ${failures.join(", ")}`,
  );
});

test("line endings other than LF do not hide the definition", () => {
  for (const reply of [
    "~~~ts\rconst x = 1;\r~~~\r\rSee [guide][g].\r\r[g]: /guide\r",
    "```ts\r\nconst x = 1;\r\n```\r\n\r\nSee [guide][g].\r\n\r\n[g]: /guide\r\n",
    "See [guide][g].\r\r```ts\rconst x = 1;\r```\r\r[g]: /guide\r",
  ]) {
    if (asOneDocument(reply) <= asBlocks(reply)) {
      continue;
    }
    assert.equal(markdownRenderScope(reply), "document", JSON.stringify(reply));
  }
});

test("a backtick opener carrying a backtick is prose, not a fence", () => {
  for (const opener of ["```bad`", "````bad`", "```js`x`"]) {
    const reply = `${opener} See [guide][g].\n\n[g]: /guide\n`;
    assert.ok(
      asOneDocument(reply) > asBlocks(reply),
      `${opener} should lose its anchor when split`,
    );
    assert.equal(markdownRenderScope(reply), "document", opener);
  }

  const tilde = "~~~bad` See [guide][g].\n\n[g]: /guide\n";
  assert.equal(asOneDocument(tilde), asBlocks(tilde));
  assert.equal(markdownRenderScope(tilde), "blocks");
});

test("a definition lookalike that no parser registers keeps block rendering", () => {
  // Wrong answers here only cost the reply its Copy/Download controls, not content.
  for (const lookalike of [
    "\t[two]: /tab-indented-code-block",
    "    [two]: /indented-code-block",
    "-[two]: /no-space-is-not-a-list",
    "1.[two]: /no-space-is-not-a-list",
    // Its first content character counts as content in CommonMark but as whitespace to JS `\S`.
    "     first\n    [two]: /still-code",
    "    \u00a0\n    [two]: /still-code",
  ]) {
    const reply = `Compare [one][two].\n\n${lookalike}\n\n\`\`\`ts\nconst x = 1;\n\`\`\`\n`;
    assert.equal(asOneDocument(reply), asBlocks(reply), lookalike);
    assert.equal(markdownRenderScope(reply), "blocks", lookalike);
  }
});

test("a live reference pair inside a list or quote still resolves", () => {
  for (const definition of [
    "- [g]: /guide",
    "> [g]: /guide",
    "- - [g]: /guide",
    "- > [g]: /guide",
    "> - [g]: /guide",
  ]) {
    const reply = `See [guide][g].\n\n${definition}\n`;
    assert.ok(asOneDocument(reply) > asBlocks(reply), definition);
    assert.equal(markdownRenderScope(reply), "document", definition);
  }
});

test("ordinary code is still rendered per block", () => {
  for (const reply of [
    "Shape.\n\n```ts\ninterface G {\n  [key: string]: number[][];\n}\n\nconst c = grid[row][col];\n```\n",
    "Compare [one][two].\n\n```css\na[href]:hover { color: red; }\n```\n",
    "Compare [one][two].\n\n    [two]: not-a-definition\n",
    "How:\n\n```md\n[two]: https://example.com/two\n```\n\nText [one][two].\n",
    "See [a][ref].\n\n1. item\n   ```python\n   def f() -> list[str]:\n       return []\n   ```\n",
    "See [a][ref].\n\n- item\n  ```python\n  def f() -> list[str]:\n      return []\n  ```\n",
    "See [a][ref].\n\n```python\nlist[\n str\n]:\n```\n",
    'See [a][ref].\n\nd[\n "key"\n]: int\n',
    'See [a][ref].\n\nd["key"]: int\n',
  ]) {
    assert.equal(markdownRenderScope(reply), "blocks", reply);
    assert.equal(
      asOneDocument(reply) > asBlocks(reply),
      false,
      `this reply does lose an anchor when split, so it belongs in the guard above:\n${reply}`,
    );
  }
});
