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
  // Indented quote content, and a definition on a list item's continuation line: four columns
  // absolute, but flush with the content column `10. ` opened.
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

test("a reference is never split into blocks because its label is long", () => {
  // Every case above uses a one-character label. Length is the one dimension the probes bound and
  // the only one this file never varied, which is why #9540 and #9645's half-fix were invisible.
  // In scope only when the two paths really differ, so past 999 drops out and this cannot pass
  // vacuously.
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
  // CommonMark forbids a backtick in a backtick fence's info string, so the line never opens a
  // fence and the block is ordinary prose -- which is where a reference can still be waiting.
  for (const opener of ["```bad`", "````bad`", "```js`x`"]) {
    const reply = `${opener} See [guide][g].\n\n[g]: /guide\n`;
    assert.ok(
      asOneDocument(reply) > asBlocks(reply),
      `${opener} should lose its anchor when split`,
    );
    assert.equal(markdownRenderScope(reply), "document", opener);
  }

  // A tilde opener has no such rule: backticks in its info string are fine and it is a fence,
  // so the definition inside it is code and the reply keeps block rendering.
  const tilde = "~~~bad` See [guide][g].\n\n[g]: /guide\n";
  assert.equal(asOneDocument(tilde), asBlocks(tilde));
  assert.equal(markdownRenderScope(tilde), "blocks");
});

test("a definition lookalike that no parser registers keeps block rendering", () => {
  // These cost nothing if we get them wrong -- the reply keeps its content either way -- but
  // each one that reaches `document` is a reply that loses its Copy/Download controls for no
  // reason, which is the residue this path exists to shrink.
  for (const lookalike of [
    "\t[two]: /tab-indented-code-block",
    "    [two]: /indented-code-block",
    "-[two]: /no-space-is-not-a-list",
    "1.[two]: /no-space-is-not-a-list",
    // An indented code block whose first content character is one CommonMark counts as
    // ordinary content but JavaScript's `\S` calls whitespace.
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
  // The regression this path exists for: nothing here is a definition, so each of these must
  // keep its per-block Copy code / Download file controls.
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

test("a shortcut or collapsed reference is never split from its definition", () => {
  const failures: string[] = [];
  for (const definition of DEFINITION_CONTEXTS) {
    for (const neutral of NEUTRAL_BLOCKS) {
      for (const reference of ["[g]", "[g][]", "![g]", "[G]", "[ g ]"]) {
        for (const reply of [
          `See ${reference}.\n\n${neutral}\n\n${definition}\n`,
          `${definition}\n\n${neutral}\n\nSee ${reference}.\n`,
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
  }
  assert.deepEqual(failures, [], failures.join("\n"));
});

test("a shortcut reference matches its definition the way CommonMark matches labels", () => {
  for (const [reference, definition] of [
    ["Paris is the capital [1].", "[1]: https://en.wikipedia.org/wiki/Paris"],
    ["Read [Unsloth docs] first.", "[unsloth  DOCS]: https://docs.unsloth.ai"],
    ["Read [Unsloth\ndocs] first.", "[unsloth docs]: https://docs.unsloth.ai"],
    ["Read [unsloth docs] first.", "[Unsloth\tDocs]: https://docs.unsloth.ai"],
    ["Read [a\\]b] first.", "[a\\]b]: https://x.test/ab"],
    ["Read [`code`] first.", "[`code`]: https://x.test/code"],
    ["Read [1](not a link) first.", "[1]: https://x.test/one"],
    ["Read [1](foo(and(bar)) first.", "[1]: https://x.test/one"],
    ["Read [1](<foo\nbar>) first.", "[1]: https://x.test/one"],
    ["Read [1](<https://x.test>\"title\") first.", "[1]: https://x.test/one"],
    [
      "Read [site](https://x.test/`tag) [1] ` first.",
      "[1]: https://x.test/one",
    ],
    [
      "Read <https://x.test/`> [1] ` first.",
      "[1]: https://x.test/one",
    ],
    [
      "Read <foo`bar@example.com> [1] ` first.",
      "[1]: https://x.test/one",
    ],
    [
      "Read [a [b]](https://x.test/`tag) [1] ` first.",
      "[1]: https://x.test/one",
    ],
    [
      "Read [a `[`](https://x.test/`tag) [1] ` first.",
      "[1]: https://x.test/one",
    ],
    [
      "Read [site `]`](https://x.test/`tag) [1] ` first.",
      "[1]: https://x.test/one",
    ],
    [
      "Read [site](foo`bar\\ ) [1] ` first.",
      "[1]: https://x.test/one",
    ],
    [
      "[site](foo`bar\\ ) [1] `",
      "[1]: https://x.test/one",
    ],
    [
      '[site](foo`bar "title\\\ncontinued") [1] `',
      "[1]: https://x.test/one",
    ],
    [
      "`[x](foo`[site](url`tag)) [1] `",
      "[1]: https://x.test/one",
    ],
    ["[x]: <broken [1]", "[1]: /one"],
    [
      'Read <span title="`"> [1] ` first.',
      "[1]: https://x.test/one",
    ],
    [
      'Read <span hidden title="`"> [1] ` first.',
      "[1]: https://x.test/one",
    ],
    [
      'Read <span title=">`"> [1] ` first.',
      "[1]: https://x.test/one",
    ],
    ["Read <!-- ` --> [1] ` first.", "[1]: https://x.test/one"],
    ["> `open\n>\n> [1]\n> `", "[1]: https://x.test/one"],
    ["> `open\n> # heading\n> [1]\n> `", "[1]: https://x.test/one"],
    [
      "> [1](/inline \"title\n> # heading\n> continuation\")",
      "[1]: https://x.test/one",
    ],
    [
      `Read [1](${"(".repeat(33)}x${")".repeat(33)}) first.`,
      "[1]: https://x.test/one",
    ],
    ["Read [SS] first.", "[\u1E9E]: https://x.test/ss"],
    ["Read [Stra\u00DFe] first.", "[STRASSE]: https://x.test/strasse"],
  ]) {
    const reply = `${reference}\n\n${definition}\n`;
    assert.ok(asOneDocument(reply) > asBlocks(reply), JSON.stringify(reply));
    assert.equal(markdownRenderScope(reply), "document", JSON.stringify(reply));
  }
});

test("inline math does not lend its backticks to a later code span", () => {
  assert.equal(
    markdownRenderScope("Math $a ` b$ [1] `\n\n[1]: /one\n"),
    "document",
  );
  assert.equal(
    markdownRenderScope(
      "Cost $\n\nMath $a ` b$ [1] `\n\n[1]: /one\n",
    ),
    "document",
  );
});

test("table cells do not share code span delimiters", () => {
  const reply =
    "| left | right |\n| --- | --- |\n| `open | [1] |\n| x | close` |\n\n[1]: /one\n";
  assert.ok(asOneDocument(reply) > asBlocks(reply));
  assert.equal(markdownRenderScope(reply), "document");
});

test("a bracketed label that is not a shortcut reference keeps block rendering", () => {
  for (const reply of [
    "Use `[1]` here.\n\n[1]: https://x.test\n",
    "Use ``a [1] b`` here.\n\n[1]: https://x.test\n",
    "Use `[site](https://x.test) [1]` here.\n\n[1]: https://x.test/unused\n",
    "Use ``<https://x.test/`> [1]`` here.\n\n[1]: https://x.test/unused\n",
    "Use ``[a [b]](https://x.test/`tag) [1]`` here.\n\n[1]: https://x.test/unused\n",
    'Use ``<span title="`"> [1]`` here.\n\n[1]: https://x.test/unused\n',
    "Use <span title=`bad> [1] ` here.\n\n[1]: https://x.test/unused\n",
    "Use ``$a ` b$ [1]`` here.\n\n[1]: https://x.test/unused\n",
    "Use [1](https://x.test/inline).\n\n[1]: https://x.test/unused\n",
    "Use [1](https://x.test/a_(b)).\n\n[1]: https://x.test/unused\n",
    "Use [1](foo(and(bar))).\n\n[1]: https://x.test/unused\n",
    `Use [1](${"(".repeat(32)}x${")".repeat(32)}).\n\n[1]: https://x.test/unused\n`,
    "Use [1](\\(foo\\)).\n\n[1]: https://x.test/unused\n",
    "Use [1](<>).\n\n[1]: https://x.test/unused\n",
    "Use [1](<https://x.test/a b> \"title\").\n\n[1]: https://x.test/unused\n",
    "Use [1](/url 'title').\n\n[1]: https://x.test/unused\n",
    "Use [1](/url (title)).\n\n[1]: https://x.test/unused\n",
    "Use [1](   /url\n  \"title\"  ).\n\n[1]: https://x.test/unused\n",
    "Use ![1](https://x.test/image.png).\n\n[1]: https://x.test/unused\n",
    "Not a link \\[1] here.\n\n[1]: https://x.test\n",
    "Note [^1].\n\n[^1]: a footnote\n",
    "Cites [2].\n\n[1]: https://x.test/1\n",
    "No uses.\n\n[1]: https://x.test/a\n[1]: https://x.test/b\n",
    "No uses.\n\n[1]: https://x.test/a[1]\n",
    "```py\nx = a[1]\n```\n\n[1]: https://x.test\n",
  ]) {
    assert.equal(asOneDocument(reply), asBlocks(reply), JSON.stringify(reply));
    assert.equal(markdownRenderScope(reply), "blocks", JSON.stringify(reply));
  }
});
