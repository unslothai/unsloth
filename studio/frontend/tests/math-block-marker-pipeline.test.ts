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

/*
 * THE MARKER, RUN THROUGH THE PIPELINE IT ACTUALLY RUNS IN, rather than over a tree this file
 * built to suit itself.
 *
 * `tests/math-block-marker.test.ts` feeds the transform hast trees it constructs. That cannot
 * catch the failure this change was one refactor away from shipping: the class landing somewhere
 * the SANITIZER strips it. `rehype-sanitize`'s `defaultSchema` permits `className` on `a`, `code`,
 * `h2`, `li`, `ol`, `section` and `ul` and nowhere else, so a class put on a `<p>` before the
 * sanitize pass disappears without a word, and the whole change measures as doing nothing.
 *
 * So this reproduces Streamdown's real plugin order -- parse, gfm, remark-math, remark-rehype,
 * raw, sanitize, and only then the maths plugin's rehype pass with the marker composed in front of
 * it -- and asserts on the tree that comes out.
 */

const MARKDOWN = [
  "Energy is $E=mc^2$ in prose and $p=mv$ too.",
  "",
  "$$",
  "\\frac{a^3}{b}",
  "$$",
  "",
  // A LOOSE list, with the blank line that makes it one. A TIGHT list is unwrapped by
  // `mdast-util-to-hast`, so its item holds the maths directly and the hoist past an inline
  // paragraph never runs. A fixture built on a tight list leaves that hoist untested while
  // looking as though it covers it.
  "- loose item with $x_i$ inline",
  "",
  "- a second item, which is what makes the list loose",
  "",
  // Inside a blockquote, so that hoisting PAST the cell would have somewhere to land. A
  // root-level table has no containable ancestor at all, so it cannot tell the abandon apart
  // from the walk simply running out of tree.
  "> | h |",
  "> |---|",
  "> | $y$ |",
  "",
  // Same reasoning for display maths: inside a blockquote, so treating it as inline would put a
  // class somewhere visible to this test.
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
    // unified's `Plugin` generics cannot describe an attacher composed at runtime. This is the
    // same value Streamdown takes as `MathPlugin["rehypePlugin"]`, where it typechecks.
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

  // PRECONDITION: the pipeline really rendered maths. If KaTeX silently did nothing, every
  // assertion below would be about an empty document and would still be capable of passing.
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
  // Without this the assertion above passes just as well on a fixture whose list item never had a
  // formula in it, which is the shape of a test that stops testing when the fixture drifts.
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
  // The fixture's first paragraph deliberately carries two.
  const tree = await render(MARKDOWN);
  const paragraphs = collect(tree).marked.filter((tag) => tag === "p");
  assert.equal(
    paragraphs.length,
    2,
    "one for the prose paragraph, one for the blockquote's",
  );
});

/*
 * The two fixtures below are wrapped in a blockquote, and that is the whole point of them.
 *
 * At the root of a document there is no containable ancestor above a `pre` or above a `table`, so
 * a marker that WRONGLY treated display maths as inline, or that WRONGLY hoisted past a table
 * cell, would still mark nothing and the test would still pass. Both properties would be
 * non-load-bearing while looking covered. The blockquote gives the wrong behaviour somewhere to
 * land, so these assertions discriminate. Each one asserts that ancestor is there first.
 */

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
