// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Find in page. What has to hold is that the flat index and the document agree: every offset the
// search reports has to land on the character a reader sees, or the highlight paints over the wrong
// word and the walk sends them somewhere they did not ask to go.
//
// There is no DOM library in this project. The runner is `node --test` and every sibling test that
// needs a document hand-rolls one, which is what the flatten's structural types are for. The rest
// of the DOM half is covered in smoke-find-in-page.tsx.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  FIND_HIGHLIGHT,
  FIND_HIGHLIGHT_ACTIVE,
  MAX_PAINTED_RANGES,
  paintWindow,
} from "../src/features/find-in-page/lib/find-dom.ts";
import {
  BLOCK_SEPARATOR,
  type FindElementLike,
  type FindTextNodeLike,
  FIND_SKIP_ATTRIBUTE,
  MAX_INDEX_CHARS,
  buildTextIndex,
  endPositionAt,
  findMatches,
  foldChunk,
  normalizeQuery,
  segmentAt,
  startPositionAt,
} from "../src/features/find-in-page/lib/find-text-index.ts";

// --- the hand-rolled tree ----------------------------------------------------------------------

function text(data: string): FindTextNodeLike {
  return { nodeType: 3, data };
}

function el(
  tagName: string,
  childNodes: (FindTextNodeLike | FindElementLike)[] = [],
  attributes: Record<string, string> = {},
): FindElementLike {
  return {
    nodeType: 1,
    tagName,
    childNodes,
    getAttribute: (name) =>
      Object.hasOwn(attributes, name) ? attributes[name] : null,
  };
}

// --- the flatten -------------------------------------------------------------------------------

test("inline markup does not break a word", () => {
  // The case the whole feature turns on: markdown emphasis, code spans and links split a word
  // across text nodes, and a find that searched node by node would not see it.
  const root = el("DIV", [
    el("P", [text("un"), el("EM", [text("sloth")]), text(" studio")]),
  ]);
  const index = buildTextIndex(root);
  assert.equal(index.text, "unsloth studio");
  assert.equal(findMatches(index, "unsloth").length, 1);
});

test("a block boundary stops a match running across it", () => {
  const root = el("DIV", [
    el("P", [text("the end")]),
    el("P", [text("start of the next")]),
  ]);
  const index = buildTextIndex(root);
  assert.equal(index.text, `the end${BLOCK_SEPARATOR}start of the next`);
  // Neither the run-together form nor the spaced one: the two words are not adjacent on screen.
  assert.deepEqual(findMatches(index, "endstart"), []);
  assert.deepEqual(findMatches(index, "end start"), []);
});

test("no separator is written before the first character or after the last", () => {
  const root = el("DIV", [el("P", [text("only")])]);
  assert.equal(buildTextIndex(root).text, "only");
});

test("a subtree the walk must not read contributes nothing", () => {
  for (const [tag, attributes] of [
    ["SCRIPT", {}],
    ["STYLE", {}],
    ["TEXTAREA", {}],
    // The bar itself, so the query typed into it is not a match of itself.
    ["DIV", { [FIND_SKIP_ATTRIBUTE]: "" }],
    // A workspace the shell parks off-route rather than unmounting.
    ["DIV", { inert: "" }],
    ["DIV", { hidden: "" }],
    // The rest of the page for as long as a modal is up.
    ["DIV", { "aria-hidden": "true" }],
  ] as const) {
    const root = el("DIV", [
      el("P", [text("visible")]),
      el(tag, [text("buried")], attributes),
    ]);
    const index = buildTextIndex(root);
    assert.equal(
      index.text.includes("buried"),
      false,
      `${tag} ${JSON.stringify(attributes)} leaked into the index`,
    );
    assert.equal(index.text.includes("visible"), true);
  }
});

test("aria-hidden false does not hide a subtree", () => {
  const root = el("DIV", [
    el("P", [text("shown")], { "aria-hidden": "false" }),
  ]);
  assert.equal(buildTextIndex(root).text, "shown");
});

// --- offsets -----------------------------------------------------------------------------------

test("an offset maps back to the node and character it came from", () => {
  const first = text("Unsloth ");
  const second = text("Studio");
  const root = el("DIV", [el("P", [first, el("B", [second])])]);
  const index = buildTextIndex(root);

  const [match] = findMatches(index, "studio");
  assert.ok(match);
  const start = startPositionAt(index.segments, match.start);
  const end = endPositionAt(index.segments, match.end);
  assert.equal(start?.node, second);
  assert.equal(start?.offset, 0);
  // The match finishes the node, so the end boundary is its length -- the one offset no segment
  // holds, and the one a Range needs there.
  assert.equal(end?.node, second);
  assert.equal(end?.offset, 6);
});

test("a match spanning two nodes ends in the second", () => {
  const first = text("uns");
  const second = text("loth");
  const index = buildTextIndex(el("DIV", [el("P", [first, second])]));
  const [match] = findMatches(index, "unsloth");
  assert.ok(match);
  assert.equal(startPositionAt(index.segments, match.start)?.node, first);
  assert.equal(endPositionAt(index.segments, match.end)?.node, second);
  assert.equal(endPositionAt(index.segments, match.end)?.offset, 4);
});

test("an offset on a separator belongs to no node", () => {
  const index = buildTextIndex(
    el("DIV", [el("P", [text("a")]), el("P", [text("b")])]),
  );
  assert.equal(index.text.length, 3);
  assert.equal(segmentAt(index.segments, 1), -1);
  assert.equal(startPositionAt(index.segments, 1), null);
});

test("a case fold that would change a run's length is not applied", () => {
  // Turkish İ folds to two code units. Folding it would shift every offset after it by one, so the
  // run keeps its case instead and the offsets stay true -- which is what this asserts, by finding
  // a word that sits AFTER one.
  assert.equal("İ".toLowerCase().length, 2, "premise: the fold grows this run");
  assert.equal(foldChunk("İ"), "İ");

  const marker = text("İ");
  const after = text("Unsloth");
  const index = buildTextIndex(el("DIV", [el("P", [marker, after])]));
  const [match] = findMatches(index, "unsloth");
  assert.ok(match);
  const start = startPositionAt(index.segments, match.start);
  assert.equal(start?.node, after);
  assert.equal(start?.offset, 0);
});

test("a non-breaking space answers to the space key", () => {
  // Spelled with a char code rather than pasted: a literal U+00A0 in a source file looks like a
  // space to every reader and every diff, which is the confusion this guards against.
  const nbsp = String.fromCharCode(0x00a0);
  const index = buildTextIndex(
    el("DIV", [el("P", [text(`Unsloth${nbsp}Studio`)])]),
  );
  assert.equal(findMatches(index, "unsloth studio").length, 1);
  // Substituted one for one, so every offset after it is untouched.
  assert.equal(index.text.length, "Unsloth Studio".length);
  assert.equal(index.text.includes(nbsp), false);
});

// --- the search --------------------------------------------------------------------------------

test("matching ignores case in both directions", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("Unsloth STUDIO")])]));
  assert.equal(findMatches(index, "unsloth").length, 1);
  assert.equal(findMatches(index, "Studio").length, 1);
  assert.equal(findMatches(index, "sTuDiO").length, 1);
});

test("matches do not overlap", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("aaaa")])]));
  assert.deepEqual(findMatches(index, "aa"), [
    { start: 0, end: 2 },
    { start: 2, end: 4 },
  ]);
});

test("the match list stops at the limit it was given", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("aaaaaaaa")])]));
  assert.equal(findMatches(index, "a", 3).length, 3);
});

test("an empty query matches nothing", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("unsloth")])]));
  assert.deepEqual(findMatches(index, ""), []);
  assert.equal(normalizeQuery(""), null);
});

test("a pasted separator cannot match across a block boundary", () => {
  // Not typeable, but a paste could carry one, and it would otherwise match the very boundary the
  // separator is there to keep closed.
  const index = buildTextIndex(
    el("DIV", [el("P", [text("a")]), el("P", [text("b")])]),
  );
  assert.equal(normalizeQuery(`a${BLOCK_SEPARATOR}b`), null);
  assert.deepEqual(findMatches(index, `a${BLOCK_SEPARATOR}b`), []);
});

// --- the bounds --------------------------------------------------------------------------------

test("a document past the ceiling is flattened as far as it goes and says so", () => {
  const chunk = "x".repeat(100_000);
  const paragraphs = Array.from({ length: 60 }, () => el("P", [text(chunk)]));
  const index = buildTextIndex(el("DIV", paragraphs));
  assert.equal(index.truncated, true);
  assert.ok(index.text.length <= MAX_INDEX_CHARS);
  // What was read is still usable, which is the point of stopping rather than giving up.
  assert.ok(index.segments.length > 0);
  assert.ok(findMatches(index, "xxx").length > 0);
});

test("a document inside the ceiling is not marked truncated", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("unsloth")])]));
  assert.equal(index.truncated, false);
});

// --- the paint window --------------------------------------------------------------------------

test("every match is painted while there are few enough of them", () => {
  assert.deepEqual(paintWindow(12, 4, 400), { from: 0, to: 12 });
});

test("the paint window is capped and always holds the active match", () => {
  const total = 5_000;
  for (const active of [0, 1, 199, 200, 2_500, 4_799, 4_998, 4_999]) {
    const { from, to } = paintWindow(total, active, MAX_PAINTED_RANGES);
    assert.equal(to - from, MAX_PAINTED_RANGES, `width at ${active}`);
    assert.ok(from >= 0 && to <= total, `bounds at ${active}`);
    assert.ok(
      active >= from && active < to,
      `active ${active} fell outside ${from}..${to}`,
    );
  }
});

/** The body of one CSS rule, for the tests that pin the bar to the composer. */
function cssRule(css: string, selector: string): string {
  const at = css.indexOf(`${selector} {`);
  assert.notEqual(at, -1, `${selector} is gone`);
  return css.slice(at, css.indexOf("}", at));
}

// --- the wiring --------------------------------------------------------------------------------

test("the stylesheet paints the two highlights the code registers", async () => {
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  for (const name of [FIND_HIGHLIGHT, FIND_HIGHLIGHT_ACTIVE]) {
    assert.ok(
      css.includes(`::highlight(${name})`),
      `${name} is registered but never painted`,
    );
  }
});

test("the bar keeps itself out of the region it searches", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(bar, new RegExp(`${FIND_SKIP_ATTRIBUTE}=`));
  // And the mutation filter reads the same attribute, so the counter re-rendering does not order a
  // re-index of the conversation.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(engine, /FIND_SKIP_ATTRIBUTE/);
});

test("light mode is the chatbox's background, under a slightly heavier shadow", async () => {
  const css = await readFile(new URL("../src/index.css", import.meta.url), "utf8");
  const composer = cssRule(css, ".unsloth-composer-surface");
  const bar = cssRule(css, ".find-bar-surface");

  // The composer is the reference, not a copied colour: if it restyles, this fails rather than
  // leaving the one floating panel a shade off from the one below it.
  const background = /background-color:\s*(#[0-9a-f]{6});/i.exec(composer);
  assert.ok(background);
  assert.match(bar, new RegExp(`background-color:\\s*${background[1]};`, "i"));

  // The shadow is the composer's, a touch larger and darker, because that one sits at the bottom
  // of the page with nothing under it and this one floats over content.
  const shape = (rule: string) => {
    const hit =
      /box-shadow:\s*0 (\d+)px (\d+)px -\d+px rgba\(0, 0, 0, ([\d.]+)\);/.exec(rule);
    assert.ok(hit, "box-shadow is not in the shape this test reads");
    return { y: Number(hit[1]), blur: Number(hit[2]), alpha: Number(hit[3]) };
  };
  const from = shape(composer);
  const to = shape(bar);
  assert.ok(to.blur > from.blur, `blur ${to.blur} is not larger than ${from.blur}`);
  assert.ok(to.alpha > from.alpha, `alpha ${to.alpha} is not darker than ${from.alpha}`);
  // Slightly. A drop shadow twice the composer's would read as a different material.
  assert.ok(to.blur <= from.blur * 1.5, `blur ${to.blur} is more than slightly larger`);
  assert.ok(to.alpha <= from.alpha * 1.5, `alpha ${to.alpha} is more than slightly darker`);
  assert.ok(to.y >= from.y);
});

test("dark mode sits above the cards it floats over", async () => {
  const css = await readFile(new URL("../src/index.css", import.meta.url), "utf8");
  const value = (selector: string, property: string) => {
    const hit = new RegExp(`${property}:\\s*([^;]+);`).exec(cssRule(css, selector));
    assert.ok(hit, `${selector} has no ${property}`);
    return hit[1].trim();
  };
  const grey = (hex: string) => Number.parseInt(hex.slice(1, 3), 16);
  // The thread's message cards are `--card`. A bar at that value dissolves into whatever scrolls
  // under it, so it sits above them, and below `--border`, past which it reads as an edge.
  const bar = grey(value(".dark .find-bar-surface", "background-color"));
  const card = grey(value(".dark", "--card"));
  const border = grey(value(".dark", "--border"));
  assert.ok(bar > card, `bar ${bar} is not lighter than --card ${card}`);
  assert.ok(bar < border, `bar ${bar} is not darker than --border ${border}`);
  // And a halo in the page background, not a dark edge around a borderless panel.
  assert.match(value(".dark .find-bar-surface", "box-shadow"), /var\(--background\)/);
});

test("the bar has no border, and its buttons have a hover that shows", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const surface = /className="(find-bar-surface[^"]*)"/.exec(bar);
  assert.ok(surface, "the bar no longer wears the shared surface class");
  assert.equal(
    /\bborder\b/.test(surface[1]),
    false,
    "the bar took a border back",
  );
  // The ghost variant's own `--muted/50` hover lands within a shade of this surface, so every
  // button in the bar overrides it.
  assert.match(bar, /hover:bg-black\/\[0\.06\] dark:hover:bg-white\/10/);
  assert.equal((bar.match(/className=\{FIND_BUTTON_CLASS\}/g) ?? []).length, 3);
});

test("a long query rewinds to its first character when focus leaves", async () => {
  // Typing past the width of the field scrolls it, and a bar left showing the tail of a word says
  // nothing about what was searched for.
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(bar, /onBlur=\{rewindToStart\}/);
  assert.match(bar, /input\.setSelectionRange\(0, 0\);/);
  assert.match(bar, /input\.scrollLeft = 0;/);
  // The walk buttons must not trigger it: they cancel their own mousedown, so the field never
  // loses focus and the caret stays where the reader left it.
  assert.match(bar, /onMouseDown=\{keepFocusInField\}/);
});

test("the observer watches the attributes a workspace switch flips", async () => {
  // Chat and Images are both kept alive by the shell, so switching between them adds and removes
  // nothing -- it flips `inert`. A childList observer would never hear it, and the bar would go on
  // counting the workspace the user just left.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(engine, /attributeFilter: \[[^\]]*"inert"/);
  // And not the whole attribute stream: `class` changes on every hover.
  assert.equal(/attributes: true,\s*\n\s*attributeFilter/.test(engine), true);
  assert.equal(engine.includes('attributeFilter: ["class"'), false);
});

test("nothing of the engine is mounted while the bar is closed", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // The index, the observer and the highlights all live in `useFindInPage`, and the only component
  // that calls it is behind this return. A hook moved above it would run them on every route.
  assert.match(bar, /if \(!enabled \|\| !open\) return null;/);
  const engineCallers = bar.match(/useFindInPage\(/g) ?? [];
  assert.equal(engineCallers.length, 1);
});
