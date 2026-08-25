// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import {
  MATH_BLOCK_CLASS,
  markMathBlocks,
  withMathBlockMarker,
} from "../src/components/assistant-ui/math-block-marker.ts";

/*
 * The marker, RUN over hast trees shaped the way Streamdown's pipeline actually shapes them.
 *
 * THE FIXTURE'S OWN PRECONDITIONS ARE ASSERTED FIRST, and two of them live in dependencies rather
 * than in this repo, which is the case that reading our own source would never reveal:
 *
 *   - `rehype-sanitize`'s `defaultSchema` decides whether a class on a `<p>` survives at all. If it
 *     ever started allowing `className` everywhere, the whole reason this marker runs where it runs
 *     would evaporate, and every test below would still pass while the code became needlessly
 *     convoluted. So the schema is read and asserted.
 *   - Streamdown renders a list item with `[&>p]:inline`. That is the entire justification for
 *     hoisting past a paragraph inside a list item. If Streamdown drops it, the hoist becomes wrong
 *     and this test says so by name.
 */

type HastNode = {
  type: string;
  tagName?: string;
  properties?: { className?: unknown; [key: string]: unknown };
  children?: HastNode[];
};

const el = (
  tagName: string,
  children: HastNode[] = [],
  properties: Record<string, unknown> = {},
): HastNode => ({ type: "element", tagName, properties, children });

const text = (value: string): HastNode =>
  ({ type: "text", value }) as unknown as HastNode;

/** What `remark-math` emits for inline maths AFTER the sanitizer has been over it. */
const inlineMath = (): HastNode =>
  el("code", [text("x^2")], { className: ["language-math"] });

/** What it emits for display maths: the same `code`, wrapped in a `pre`. */
const displayMath = (): HastNode =>
  el("pre", [el("code", [text("x^2")], { className: ["language-math"] })]);

const root = (children: HastNode[]): HastNode => ({ type: "root", children });

const classesOf = (node: HastNode): string[] => {
  const raw = node.properties?.className;
  return Array.isArray(raw) ? raw.map(String) : [];
};

const marked = (node: HastNode): boolean =>
  classesOf(node).includes(MATH_BLOCK_CLASS);

const countMarked = (node: HastNode): number => {
  let n = marked(node) ? 1 : 0;
  for (const child of node.children ?? []) n += countMarked(child);
  return n;
};

test("PRECONDITION: the sanitizer would strip this class from a paragraph", async () => {
  const { defaultSchema } = (await import("hast-util-sanitize")) as {
    defaultSchema: {
      attributes?: Record<string, ReadonlyArray<unknown>>;
    };
  };
  const attributes = defaultSchema.attributes ?? {};
  const flat = (name: string): string[] =>
    (attributes[name] ?? []).map((entry) =>
      Array.isArray(entry) ? String(entry[0]) : String(entry),
    );

  assert.ok(
    !flat("*").includes("className"),
    "the wildcard schema does not permit className, so a class must be added after the sanitizer",
  );
  assert.ok(
    !flat("p").includes("className"),
    "the paragraph schema does not permit className either",
  );
  assert.ok(
    flat("code").includes("className"),
    "PRECONDITION: `code` does keep a className, which is how `language-math` survives at all",
  );
});

test("PRECONDITION: Streamdown still renders a list item's paragraph inline", () => {
  // The chunk name carries a content hash, so the directory is scanned rather than a file named.
  // A rename must not turn this precondition into a silent pass.
  const dist = new URL("../node_modules/streamdown/dist/", import.meta.url);
  const files = readdirSync(dist).filter((name) => name.endsWith(".js"));
  assert.ok(
    files.length > 0,
    "PRECONDITION: the installed Streamdown build was found",
  );
  const found = files.some((name) =>
    readFileSync(new URL(name, dist), "utf8").includes("[&>p]:inline"),
  );
  assert.ok(
    found,
    "the hoist past a paragraph inside a list item is justified by this Streamdown class",
  );
});

test("inline maths marks the paragraph that holds it", () => {
  const paragraph = el("p", [text("so "), inlineMath(), text(" grows")]);
  const tree = root([paragraph]);
  assert.equal(marked(paragraph), false, "PRECONDITION: nothing is marked yet");

  assert.equal(markMathBlocks(tree), 1);
  assert.deepEqual(classesOf(paragraph), [MATH_BLOCK_CLASS]);
  assert.equal(
    countMarked(tree),
    1,
    "exactly one block, and it is the paragraph",
  );
});

test("an existing class list is preserved rather than replaced", () => {
  const paragraph = el("p", [inlineMath()], { className: ["prose"] });
  markMathBlocks(root([paragraph]));
  assert.deepEqual(classesOf(paragraph), ["prose", MATH_BLOCK_CLASS]);
});

test("two maths roots in one paragraph mark it once", () => {
  const paragraph = el("p", [inlineMath(), text(" and "), inlineMath()]);
  const tree = root([paragraph]);
  // The count is per maths root; the class is idempotent.
  assert.equal(markMathBlocks(tree), 2);
  assert.deepEqual(classesOf(paragraph), [MATH_BLOCK_CLASS]);
  assert.equal(countMarked(tree), 1);
});

test("inline wrappers are walked through to the block", () => {
  const paragraph = el("p", [el("em", [el("strong", [inlineMath()])])]);
  const tree = root([paragraph]);
  assert.equal(markMathBlocks(tree), 1);
  assert.ok(marked(paragraph));
  assert.equal(countMarked(tree), 1, "the em and the strong are not marked");
});

test("a paragraph inside a list item hoists to the list item", () => {
  const paragraph = el("p", [inlineMath()]);
  const item = el("li", [paragraph]);
  const tree = root([el("ul", [item])]);

  assert.equal(markMathBlocks(tree), 1);
  assert.ok(marked(item), "the list item takes the class");
  assert.equal(
    marked(paragraph),
    false,
    "the paragraph does not, because Streamdown renders it inline",
  );
});

test("a heading and a blockquote paragraph are markable", () => {
  const heading = el("h2", [inlineMath()]);
  const quoted = el("p", [inlineMath()]);
  const tree = root([heading, el("blockquote", [quoted])]);

  assert.equal(markMathBlocks(tree), 2);
  assert.ok(marked(heading));
  assert.ok(marked(quoted), "a blockquote's paragraph is a normal block");
  assert.equal(countMarked(tree), 2, "the blockquote itself is not marked");
});

test("display maths is left alone, because `.katex-display` is already a block", () => {
  const tree = root([displayMath()]);
  assert.equal(
    (tree.children ?? []).length,
    1,
    "PRECONDITION: the fixture is a pre-wrapped maths code element",
  );
  assert.equal(markMathBlocks(tree), 0);
  assert.equal(countMarked(tree), 0);
});

test("maths in a table cell is abandoned, not hoisted to the table", () => {
  const cell = el("td", [inlineMath()]);
  const tree = root([el("table", [el("tbody", [el("tr", [cell])])])]);
  assert.equal(
    markMathBlocks(tree),
    0,
    "size containment does not apply to internal table elements",
  );
  assert.equal(countMarked(tree), 0);
});

test("a maths root buried deeper than the hop bound marks nothing", () => {
  let node = inlineMath();
  const depth = 13;
  for (let i = 0; i < depth; i += 1) node = el("span", [node]);
  const paragraph = el("p", [node]);
  const tree = root([paragraph]);

  assert.ok(
    depth > 12,
    "PRECONDITION: the fixture is past the twelve-hop bound",
  );
  assert.equal(markMathBlocks(tree), 0);
  assert.equal(marked(paragraph), false);
});

test("a document with no maths is untouched", () => {
  const paragraph = el("p", [text("no maths here"), el("code", [text("x")])]);
  const tree = root([paragraph]);
  assert.equal(markMathBlocks(tree), 0);
  assert.equal(countMarked(tree), 0);
});

test("the composed attacher marks the tree and then runs the maths renderer", () => {
  const seen: string[] = [];
  let optionsSeen: unknown = "not called";
  const fakeMathAttacher = (options: unknown) => {
    optionsSeen = options;
    return (tree: HastNode) => {
      // PRECONDITION for this test: the marker must already have run by the time the maths
      // renderer sees the tree, which is the whole point of composing rather than appending.
      seen.push(marked(tree.children?.[0] as HastNode) ? "marked" : "unmarked");
    };
  };

  const attacher = withMathBlockMarker([
    fakeMathAttacher,
    { errorColor: "red" },
  ]);
  const transform = attacher.call(undefined) as (
    tree: HastNode,
    file: unknown,
  ) => unknown;

  assert.deepEqual(
    optionsSeen,
    { errorColor: "red" },
    "the maths options are preserved",
  );

  const paragraph = el("p", [inlineMath()]);
  transform(root([paragraph]), {});
  assert.deepEqual(seen, ["marked"]);
  assert.ok(marked(paragraph));
});

test("the composed attacher also accepts a bare attacher with no options", () => {
  let ran = 0;
  const attacher = withMathBlockMarker(() => () => {
    ran += 1;
  });
  const transform = attacher.call(undefined) as (
    tree: HastNode,
    file: unknown,
  ) => unknown;
  const paragraph = el("p", [inlineMath()]);
  transform(root([paragraph]), {});
  assert.equal(ran, 1);
  assert.ok(marked(paragraph));
});
