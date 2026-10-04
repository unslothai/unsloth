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

test("maths inside a list item is abandoned, so the item keeps its number", () => {
  // An earlier revision marked the `li`, because Streamdown gives its paragraph `[&>p]:inline` and
  // an inline box cannot take size containment, which leaves the item as the only containable
  // ancestor. Containing it costs the item its `::marker`: `content-visibility: auto` applies style
  // containment, style containment scopes the automatic `list-item` counter, and the item can no
  // longer resolve `counter(list-item)` for its own marker. Photographed on WebKitGTK 2.50.4 the
  // number simply vanishes on the contained items. So there is no containable ancestor here at all
  // and the maths is abandoned, as it is inside a table cell.
  const paragraph = el("p", [inlineMath()]);
  const item = el("li", [paragraph]);
  const tree = root([el("ul", [item])]);

  assert.equal(markMathBlocks(tree), 0, "nothing is marked");
  assert.equal(marked(item), false, "the list item does NOT take the class");
  assert.equal(marked(paragraph), false, "nor does its inline paragraph");
});

test("maths directly inside a list item is abandoned too, not just via a paragraph", () => {
  // The `p` inside `li` shape has its own branch. This one reaches the `li` through the ordinary
  // tag walk, so it proves the exemption is in the tag sets rather than only in that branch.
  const item = el("li", [inlineMath()]);
  const tree = root([el("ol", [item])]);

  assert.equal(markMathBlocks(tree), 0);
  assert.equal(marked(item), false);
});

test("the walk does not hoist PAST a list item and contain the whole list", () => {
  // Containing the `ol` would lose every marker in it rather than one, so `ol` and `ul` are
  // uncontainable and stop the walk instead of being skipped over.
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

/*
 * EQUATION NUMBERS. `katex.css` resets `katexEqnNo` on `body` and increments it in
 * `.katex .eqn-num::before`. Style containment, which `content-visibility: auto` applies
 * unconditionally, scopes that increment per contained display. Measured on Chromium: three
 * numbered displays read (1) (2) (3) off and (1) (1) (1) on. The same fixture on WebKitGTK 2.50.4
 * does NOT reproduce it, 0 differing pixels against 120 for the list-marker case photographed in
 * the same frame, so the probe could see counter breakage and this engine simply does not do it.
 * Chromium is the engine most of the web UI runs in, so it decides.
 *
 * `guardEquationNumbers` runs AFTER KaTeX, because none of this markup exists before it.
 */

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
  // `katex.css` has two counters, `katexEqnNo` and `mmlEqnNo`, and only one of them appears in a
  // given output mode. Missing either would leave half the configurations broken.
  const numbered = withClass("span", "katex-display", [
    withClass("span", "katex", [withClass("span", "mml-eqn-num")]),
  ]);
  assert.deepEqual(guardEquationNumbers(root([numbered])), { marked: 0, unmarked: 0 });
  assert.equal(classesOf(numbered).includes(MATH_DISPLAY_CLASS), false);
});

test("a marked block holding a numbered display is UNMARKED", () => {
  // `markMathBlocks` runs before KaTeX, when a display is still `<pre><code>` and no `.eqn-num`
  // exists to see. A blockquote holding both inline maths and a numbered display would otherwise
  // scope the counter from above and break the numbering just as effectively as containing the
  // display itself would.
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
  // The control for the row above: without this, an implementation that unmarked every block
  // holding any display at all would pass and would give up containment it did not need to.
  const plain = withClass("span", "katex-display", [withClass("span", "katex")]);
  const quote = el("blockquote", [plain]);
  quote.properties = { className: [MATH_BLOCK_CLASS] };

  const counts = guardEquationNumbers(root([quote]));
  assert.equal(counts.unmarked, 0);
  assert.ok(classesOf(quote).includes(MATH_BLOCK_CLASS));
  assert.ok(classesOf(plain).includes(MATH_DISPLAY_CLASS), "and the display is still contained");
});

test("running the guard twice adds nothing the second time", () => {
  // Streamdown re-renders a settled body on every mount, so the pass has to be idempotent or the
  // class list grows without bound.
  const display = withClass("span", "katex-display", [withClass("span", "katex")]);
  const tree = root([display]);
  guardEquationNumbers(tree);
  assert.deepEqual(guardEquationNumbers(tree), { marked: 0, unmarked: 0 });
  assert.equal(
    classesOf(display).filter((c) => c === MATH_DISPLAY_CLASS).length,
    1,
  );
});
