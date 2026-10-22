// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";

import {
  MATH_BLOCK_CLASS,
  MATH_DISPLAY_CLASS,
  guardEquationNumbers,
  markMathBlocks,
  withMathBlockMarker,
} from "../src/components/assistant-ui/math-block-marker.ts";

/* Asserts dependency preconditions: sanitize's className allowlist and Streamdown's [&>p]:inline. */

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

const inlineMath = (): HastNode =>
  el("code", [text("x^2")], { className: ["language-math"] });

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
  // The chunk name carries a content hash, so the directory is scanned.
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

test("maths inside a list item is abandoned, so the item keeps its number", () => {
  // Containing an li breaks its ::marker (style containment scopes the list-item counter).
  const paragraph = el("p", [inlineMath()]);
  const item = el("li", [paragraph]);
  const tree = root([el("ul", [item])]);

  assert.equal(markMathBlocks(tree), 0, "nothing is marked");
  assert.equal(marked(item), false, "the list item does NOT take the class");
  assert.equal(marked(paragraph), false, "nor does its inline paragraph");
});

test("maths directly inside a list item is abandoned too, not just via a paragraph", () => {
  const item = el("li", [inlineMath()]);
  const tree = root([el("ol", [item])]);

  assert.equal(markMathBlocks(tree), 0);
  assert.equal(marked(item), false);
});

test("the walk does not hoist PAST a list item and contain the whole list", () => {
  // Containing an ol/ul would lose every marker, so they stop the walk.
  const item = el("li", [el("p", [inlineMath()])]);
  const list = el("ol", [item]);
  const tree = root([el("div", [list])]);

  assert.equal(markMathBlocks(tree), 0);
  assert.equal(marked(list), false, "the list itself is not marked");
  assert.equal(countMarked(tree), 0, "and nothing above it is either");
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

/* Style containment scopes KaTeX equation counters on Chromium; this runs after KaTeX. */

const withClass = (tag: string, className: string, children: HastNode[] = []): HastNode => {
  const node = el(tag, children);
  node.properties = { className: [className] };
  return node;
};

test("a display with no equation number gets the display class", () => {
  const display = withClass("span", "katex-display", [withClass("span", "katex")]);
  const tree = root([display]);

  assert.deepEqual(guardEquationNumbers(tree), { marked: 1, unmarked: 0 });
  assert.ok(classesOf(display).includes(MATH_DISPLAY_CLASS));
  assert.ok(classesOf(display).includes("katex-display"), "and keeps its own class");
});

test("a display WITH an equation number does not", () => {
  const numbered = withClass("span", "katex-display", [
    withClass("span", "katex", [withClass("span", "eqn-num")]),
  ]);
  const tree = root([numbered]);

  assert.deepEqual(guardEquationNumbers(tree), { marked: 0, unmarked: 0 });
  assert.equal(classesOf(numbered).includes(MATH_DISPLAY_CLASS), false);
});

test("the MathML equation number counts too", () => {
  // katex.css has two counters, katexEqnNo and mmlEqnNo.
  const numbered = withClass("span", "katex-display", [
    withClass("span", "katex", [withClass("span", "mml-eqn-num")]),
  ]);
  assert.deepEqual(guardEquationNumbers(root([numbered])), { marked: 0, unmarked: 0 });
  assert.equal(classesOf(numbered).includes(MATH_DISPLAY_CLASS), false);
});

test("a marked block holding a numbered display is UNMARKED", () => {
  const numbered = withClass("span", "katex-display", [
    withClass("span", "katex", [withClass("span", "eqn-num")]),
  ]);
  const quote = el("blockquote", [numbered]);
  quote.properties = { className: [MATH_BLOCK_CLASS] };
  const tree = root([quote]);

  const counts = guardEquationNumbers(tree);
  assert.equal(counts.unmarked, 1);
  assert.equal(classesOf(quote).includes(MATH_BLOCK_CLASS), false, "the block loses containment");
});

test("a marked block holding an UNNUMBERED display keeps its class", () => {
  const plain = withClass("span", "katex-display", [withClass("span", "katex")]);
  const quote = el("blockquote", [plain]);
  quote.properties = { className: [MATH_BLOCK_CLASS] };

  const counts = guardEquationNumbers(root([quote]));
  assert.equal(counts.unmarked, 0);
  assert.ok(classesOf(quote).includes(MATH_BLOCK_CLASS));
  assert.ok(classesOf(plain).includes(MATH_DISPLAY_CLASS), "and the display is still contained");
});

test("running the guard twice adds nothing the second time", () => {
  // Streamdown re-renders settled bodies on mount, so this must be idempotent.
  const display = withClass("span", "katex-display", [withClass("span", "katex")]);
  const tree = root([display]);
  guardEquationNumbers(tree);
  assert.deepEqual(guardEquationNumbers(tree), { marked: 0, unmarked: 0 });
  assert.equal(
    classesOf(display).filter((c) => c === MATH_DISPLAY_CLASS).length,
    1,
  );
});
