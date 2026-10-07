// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createMathPlugin } from "@streamdown/math";
import rehypeRaw from "rehype-raw";
import rehypeSanitize, { defaultSchema } from "rehype-sanitize";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";

import { withMathBlockMarker } from "../src/components/assistant-ui/math-block-marker.ts";

/* Runs Streamdown's real plugin order: the sanitizer strips a className put on <p> before it. */

const MARKDOWN = [
  "Energy is $E=mc^2$ in prose and $p=mv$ too.",
  "",
  "$$",
  "\\frac{a^3}{b}",
  "$$",
  "",
  // Loose list: a tight list is unwrapped and skips the paragraph hoist.
  "- loose item with $x_i$ inline",
  "",
  "- a second item, which is what makes the list loose",
  "",
  // In a blockquote so a wrong hoist past the cell has somewhere to land.
  "> | h |",
  "> |---|",
  "> | $y$ |",
  "",
  "> $$",
  "> \\frac{c}{d}",
  "> $$",
  "",
  "# heading with $z$",
  "",
  "> quoted $q$ here",
  "",
].join("\n");

type Node = {
  type: string;
  tagName?: string;
  properties?: { className?: unknown };
  children?: Node[];
};

const render = async (markdown: string) => {
  const baseMath = createMathPlugin({ singleDollarTextMath: true });
  const [remarkMath, remarkMathOptions] = baseMath.remarkPlugin as [
    // biome-ignore lint/suspicious/noExplicitAny: a unified attacher, typed loosely on purpose
    any,
    unknown,
  ];
  const processor = unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath, remarkMathOptions)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeRaw)
    .use(rehypeSanitize, defaultSchema)
    // unified's Plugin generics cannot describe a runtime-composed attacher.
    .use(withMathBlockMarker(baseMath.rehypePlugin) as never);
  return (await processor.run(processor.parse(markdown))) as unknown as Node;
};

const collect = (tree: Node) => {
  const marked: string[] = [];
  let displayRoots = 0;
  let mathRoots = 0;
  const walk = (node: Node) => {
    if (node.type === "element") {
      const classes = Array.isArray(node.properties?.className)
        ? node.properties.className.map(String)
        : [];
      if (classes.includes("aui-math-block")) marked.push(node.tagName ?? "?");
      if (classes.includes("katex-display")) displayRoots += 1;
      if (classes.includes("katex")) mathRoots += 1;
    }
    for (const child of node.children ?? []) walk(child);
  };
  walk(tree);
  return { marked, displayRoots, mathRoots };
};

test("the class survives the sanitizer and lands on the right blocks", async () => {
  const tree = await render(MARKDOWN);
  const { marked, displayRoots, mathRoots } = collect(tree);

  assert.ok(
    mathRoots >= 6,
    `PRECONDITION: KaTeX rendered maths, saw ${mathRoots} roots`,
  );
  assert.equal(
    displayRoots,
    2,
    "PRECONDITION: both display formulae rendered as such",
  );

  assert.deepEqual(
    marked,
    ["p", "h1", "p"],
    "the prose paragraph, the heading and the blockquote's paragraph",
  );
  assert.equal(
    marked.includes("li"),
    false,
    "the list item is NOT among them: containing it would cost the item its number, see " +
      "UNCONTAINABLE_TAGS in math-block-marker.ts",
  );
});

test("the list item in the fixture really does carry maths, so its absence means something", async () => {
  const tree = await render(MARKDOWN);
  const items: Node[] = [];
  const walk = (node: Node): void => {
    if (node.tagName === "li") items.push(node);
    for (const child of node.children ?? []) walk(child);
  };
  walk(tree);
  assert.ok(items.length >= 1, "PRECONDITION: the fixture has a list item");
  const text = JSON.stringify(items);
  assert.ok(
    text.includes("katex"),
    "PRECONDITION: that list item rendered maths, so declining to mark it is a choice",
  );
});

test("two inline formulae in one paragraph mark it once", async () => {
  const tree = await render(MARKDOWN);
  const paragraphs = collect(tree).marked.filter((tag) => tag === "p");
  assert.equal(
    paragraphs.length,
    2,
    "one for the prose paragraph, one for the blockquote's",
  );
});

/* Blockquote wrapping gives wrong behaviour a containable ancestor to land on. */

const containableAncestors = (tree: Node): string[] => {
  const found: string[] = [];
  const walk = (node: Node) => {
    if (node.type === "element" && node.tagName === "blockquote") {
      found.push(node.tagName);
    }
    for (const child of node.children ?? []) walk(child);
  };
  walk(tree);
  return found;
};

test("display maths is not marked, because `.katex-display` already is a block", async () => {
  const tree = await render("> $$\n> \\frac{a}{b}\n> $$\n");
  const { marked, displayRoots } = collect(tree);
  assert.equal(
    displayRoots,
    1,
    "PRECONDITION: the fixture rendered display maths",
  );
  assert.deepEqual(
    containableAncestors(tree),
    ["blockquote"],
    "PRECONDITION: there is a containable ancestor a wrong answer could land on",
  );
  assert.deepEqual(marked, [], "nothing else needed marking");
});

test("maths in a table cell is left alone", async () => {
  const tree = await render("> | h |\n> |---|\n> | $y$ |\n");
  const { marked, mathRoots } = collect(tree);
  assert.ok(mathRoots >= 1, "PRECONDITION: the cell's maths rendered");
  assert.deepEqual(
    containableAncestors(tree),
    ["blockquote"],
    "PRECONDITION: there is a containable ancestor a wrong hoist could land on",
  );
  assert.deepEqual(
    marked,
    [],
    "size containment does not apply to internal table elements, so nothing is marked",
  );
});

test("a document with no maths gets no class", async () => {
  const tree = await render(
    "Just prose, and `code`, and a [link](https://example.com).\n",
  );
  assert.deepEqual(collect(tree).marked, []);
});
