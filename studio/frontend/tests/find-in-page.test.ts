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
  mutatesSearchableText,
  paintWindow,
  resolvePortalSurfaces,
  selectRangeFallback,
} from "../src/features/find-in-page/lib/find-dom.ts";
import {
  BLOCK_SEPARATOR,
  FIND_SKIP_ATTRIBUTE,
  type FindElementLike,
  type FindTextNodeLike,
  MAX_INDEX_CHARS,
  MAX_MATCHES,
  MAX_NODE_CHARS,
  PORTAL_RESERVE_CHARS,
  buildTextIndex,
  dropProbeFurthestFrom,
  endPositionAt,
  findMatches,
  foldText,
  normalizeQuery,
  segmentAt,
  startPositionAt,
} from "../src/features/find-in-page/lib/find-text-index.ts";
import { useFindInPageStore } from "../src/features/find-in-page/stores/find-in-page-store.ts";

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

test("dotted I folds to a plain i, and the offsets still hold", () => {
  // Its default fold is two code units, which one index character cannot stand for. The Turkic fold
  // is a bare `i`, which fits and is what a search wants: the platform's own find matches all four
  // spellings of Istanbul against a visible dotted one, and now so does this.
  assert.equal("İ".toLowerCase().length, 2, "premise: the default fold grows");
  assert.equal(foldText("İ"), "i");

  const spellings = buildTextIndex(
    el("DIV", [el("P", [text("Welcome to İstanbul")])]),
  );
  for (const query of ["istanbul", "İstanbul", "ISTANBUL", "Istanbul"]) {
    assert.equal(findMatches(spellings, query).length, 1, query);
  }

  // And the offsets stay true, which is what folding it in place buys.
  const marker = text("İ");
  const after = text("Unsloth");
  const index = buildTextIndex(el("DIV", [el("P", [marker, after])]));
  const [match] = findMatches(index, "unsloth");
  assert.ok(match);
  const start = startPositionAt(index.segments, match.start);
  assert.equal(start?.node, after);
  assert.equal(start?.offset, 0);
});

test("an expanding fold does not change the letters around it", () => {
  // Greek final sigma is decided by what follows it, so the fold of a run is not the folds of its
  // code points strung together. One expanding character in a node used to force the per-code-point
  // path for the whole run, turning every sigma in it into the wrong letter: text plainly on screen
  // then matched nothing.
  const dottedI = String.fromCharCode(0x0130); // Turkish dotted I, whose fold is two units
  const greek = `\u039f\u0394\u039f\u03a3 ${dottedI} \u039f\u03a3`; // "ΟΔΟΣ İ ΟΣ"
  const index = buildTextIndex(el("DIV", [el("P", [text(greek)])]));
  // The length has to hold, or every offset after it maps to the wrong character.
  assert.equal(index.text.length, greek.length);
  assert.equal(index.segments[0].length, greek.length);
  // Both sigmas are word-final here, and both fold to the one form a query folds to.
  assert.equal(findMatches(index, "\u039f\u03a3").length, 2);
  assert.equal(findMatches(index, "\u039f\u0394\u039f\u03a3").length, 1);
  // And the character that would have grown is one plain `i`, still one wide.
  assert.equal(index.text, "\u03bf\u03b4\u03bf\u03c3 i \u03bf\u03c3");
});

test("casing context carries across inline markup", () => {
  // Split by an `<em>`, a word-final sigma folds medial per node and final over the run, so the two
  // used to disagree about a word plainly on screen. Both sigmas now fold to one letter, which
  // settles it whichever way the flatten arrives at the run.
  const index = buildTextIndex(
    el("DIV", [el("P", [text("\u039f"), el("EM", [text("\u03a3")])])]),
  );
  assert.equal(index.text, "\u03bf\u03c3");
  assert.equal(findMatches(index, "\u039f\u03a3").length, 1);
  // The offset map still lands on the right nodes, which is what folding in place buys.
  assert.equal(index.segments.length, 2);
  assert.equal(index.segments[1].start, 1);
});

test("several dotted I in a run fold without drift", () => {
  const dottedI = String.fromCharCode(0x0130);
  const raw = `${dottedI}a${dottedI}\u039f\u03a3${dottedI}b`;
  const index = buildTextIndex(el("DIV", [el("P", [text(raw)])]));
  assert.equal(index.text.length, raw.length);
  // Each one is a plain `i`, and the sigma between them still reads its own context: a cased letter
  // follows it, so it stays medial.
  assert.equal(index.text, "iai\u03bf\u03c3ib");
});

test("either sigma finds the other, whichever one is on screen", () => {
  // `toLowerCase` picks the final form by position, so uppercase Greek ending in sigma folded one
  // way and a query typed with the medial sigma folded the other, and half the spellings a reader
  // can produce found nothing. Measured: chromium, firefox and webkit all match `ΟΣ` from either.
  const index = buildTextIndex(
    el("DIV", [el("P", [text("\u039f\u0394\u039f\u03a3 \u039f\u03a3")])]),
  );
  for (const query of [
    "\u03bf\u03c3", // medial, which is what a keyboard gives mid-word
    "\u03bf\u03c2", // final, which is what it gives at the end of one
    "\u039f\u03a3", // and the uppercase the document itself is written in
  ]) {
    assert.equal(
      findMatches(index, query).length,
      2,
      `${escape(query)} found nothing`,
    );
  }
  // One letter in the index, so the offsets still stand for what is written.
  const run = "\u03a3\u03a3\u03a3";
  const sigmas = buildTextIndex(el("DIV", [el("P", [text(run)])]));
  assert.equal(sigmas.text, "\u03c3\u03c3\u03c3");
  assert.equal(sigmas.text.length, run.length);
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

test("a dotted I does not make the rest of its run case-sensitive", () => {
  // Its default fold grows, and giving up on the whole run for that left ordinary words in the
  // same node unmatchable.
  const index = buildTextIndex(el("DIV", [el("P", [text("HELLO İ")])]));
  assert.equal(findMatches(index, "hello").length, 1);
  // And the offsets still line up, which is what the whole-run fallback was protecting.
  const [match] = findMatches(index, "hello");
  assert.equal(startPositionAt(index.segments, match.start)?.offset, 0);
});

test("a text node bigger than its share contributes its prefix", () => {
  // One Bash step's log arrives as one text node. Dropping it whole indexed nothing at all.
  const node = text(`unsloth ${"x".repeat(MAX_NODE_CHARS + 10)}`);
  const index = buildTextIndex(el("DIV", [el("P", [node])]));
  assert.equal(index.truncated, true);
  assert.equal(index.text.length, MAX_NODE_CHARS);
  assert.equal(findMatches(index, "unsloth").length, 1);
  // The prefix maps back to the node it came from, so a match in it is reachable.
  assert.equal(startPositionAt(index.segments, 0)?.node, node);
});

test("an oversized node does not take the whole budget with it", () => {
  // It used to: the node claimed every remaining character, the walk stopped, and the messages the
  // reader was looking at were not in the index at all. A share each, and the walk goes on.
  const log = el("PRE", [text("x".repeat(MAX_INDEX_CHARS + 1000))]);
  const onScreen = el("P", [
    text("the message in front of the reader says unsloth"),
  ]);
  const index = buildTextIndex(el("DIV", [log, onScreen]));
  assert.equal(index.truncated, true);
  assert.equal(findMatches(index, "in front of the reader").length, 1);
  // The log is still there, up to its share.
  assert.equal(index.text.startsWith("x".repeat(MAX_NODE_CHARS)), true);
});

test("a popover over a document at the ceiling is still searchable", () => {
  // The workspace filling the budget used to end the walk, and the portal roots come after it, so
  // the one surface the reader is actually looking at fell out of the index entirely. That is the
  // case portal support exists for, so it gets a reserve rather than the leftovers.
  const filler = Array.from({ length: 50 }, () =>
    el("P", [text("x".repeat(MAX_NODE_CHARS))]),
  );
  const popover = el("DIV", [el("P", [text("a model named unsloth zephyr")])]);
  const index = buildTextIndex(el("DIV", filler), [popover]);
  assert.equal(index.truncated, true);
  assert.equal(findMatches(index, "unsloth zephyr").length, 1);
});

test("the reserve is only held back when there is a portal to hold it for", () => {
  // With nothing portaled the workspace gets the whole budget, so the ceiling means what it says.
  const filler = Array.from({ length: 50 }, () =>
    el("P", [text("x".repeat(MAX_NODE_CHARS))]),
  );
  const alone = buildTextIndex(el("DIV", filler));
  assert.equal(alone.text.length, MAX_INDEX_CHARS);
  // And with one, the workspace gives up only the reserve, not more.
  const withPopover = buildTextIndex(el("DIV", filler), [
    el("DIV", [el("P", [text("unsloth")])]),
  ]);
  assert.ok(
    withPopover.text.length > MAX_INDEX_CHARS - PORTAL_RESERVE_CHARS,
    `index was ${withPopover.text.length}`,
  );
  assert.ok(withPopover.text.length <= MAX_INDEX_CHARS);
});

test("an element the engine is not painting is skipped", () => {
  // Attributes miss the common case: a responsive `hidden lg:flex` is a class.
  const hidden = {
    ...el("DIV", [text("buried")]),
    checkVisibility: () => false,
  };
  const shown = { ...el("DIV", [text("shown")]), checkVisibility: () => true };
  const index = buildTextIndex(el("DIV", [hidden, shown]));
  assert.equal(index.text.includes("buried"), false);
  assert.equal(index.text.includes("shown"), true);
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

test("content-visibility skipping is not treated as invisibility", () => {
  // A `content-visibility: auto` subtree the reader has not scrolled to is SKIPPED, not hidden,
  // and asking `checkVisibility` about that would drop the far half of a Hub README from the
  // index. Nothing would put it back: scrolling renders the subtree without mutating the DOM.
  const asked: unknown[] = [];
  const probe = {
    ...el("DIV", [text("readme")]),
    checkVisibility: (options?: unknown) => {
      asked.push(options);
      return true;
    },
  };
  const index = buildTextIndex(el("DIV", [probe]));
  assert.equal(index.text.includes("readme"), true);
  assert.equal(asked.length, 1);
  assert.deepEqual(asked[0], {
    contentVisibilityAuto: false,
    opacityProperty: false,
    checkOpacity: false,
    visibilityProperty: true,
    checkVisibilityCSS: true,
  });
});

test("both spellings of every visibility option are asked for", () => {
  // `visibilityProperty`/`opacityProperty` are renames of `checkVisibilityCSS`/`checkOpacity`, an
  // engine reads only the name it knows, and Web IDL drops an unknown member silently. The modern
  // name alone is a no-op on Chrome 105-120 and Firefox 106-121, which would then index and
  // highlight `visibility: hidden` text.
  const seen: Record<string, unknown>[] = [];
  const probe = {
    ...el("DIV", [text("readme")]),
    checkVisibility: (options?: Record<string, unknown>) => {
      if (options) seen.push(options);
      return true;
    },
  };
  buildTextIndex(el("DIV", [probe]));
  assert.equal(seen.length, 1);
  const options = seen[0];
  for (const [modern, historic] of [
    ["visibilityProperty", "checkVisibilityCSS"],
    ["opacityProperty", "checkOpacity"],
  ]) {
    assert.equal(modern in options, true, `${modern} is missing`);
    assert.equal(historic in options, true, `${historic} is missing`);
    // The two names mean the same thing, so they must never disagree.
    assert.equal(
      options[modern],
      options[historic],
      `${modern} != ${historic}`,
    );
  }
});

test("an engine that honours only the historic option names still hides hidden text", () => {
  // Simulates Chrome 105-120 / Firefox 106-121: `checkVisibility` exists, but the modern option
  // names are unknown to it and therefore ignored.
  const legacyEngine = (style: { visibility?: string }) => ({
    checkVisibility: (options?: Record<string, unknown>) =>
      !(options?.checkVisibilityCSS === true && style.visibility === "hidden"),
  });
  const hidden = {
    ...el("SPAN", [text("invisible")]),
    ...legacyEngine({ visibility: "hidden" }),
  };
  const index = buildTextIndex(el("DIV", [hidden]));
  assert.equal(index.text.includes("invisible"), false);
});

test("an inline SVG is skipped despite reporting a lowercase tag", () => {
  // Only HTML elements report their tag uppercased. SVG and MathML keep their source casing, so a
  // Mermaid diagram's `<svg>` answers "svg" and walked past a set spelled in HTML casing, putting
  // its labels in the index as matches a Range cannot reliably paint.
  const svg = el("svg", [el("text", [text("mermaid label")])]);
  const index = buildTextIndex(el("DIV", [svg, el("P", [text("prose")])]));
  assert.equal(index.text.includes("mermaid"), false);
  assert.equal(index.text.includes("prose"), true);
});

test("a portaled surface is indexed after the scope, behind a boundary", () => {
  // A popover renders to the body, outside the scope, and a reader sees it as part of the page.
  const scope = el("DIV", [el("P", [text("in the thread")])]);
  const portal = el("DIV", [el("P", [text("in the popover")])]);
  const index = buildTextIndex(scope, [portal]);
  assert.equal(index.text, `in the thread${BLOCK_SEPARATOR}in the popover`);
  assert.equal(findMatches(index, "in the popover").length, 1);
  // Separate surfaces, so nothing runs out of one and into the other.
  assert.deepEqual(findMatches(index, "thread in"), []);
});

test("every portaled surface is separated from the one before it", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("a")])]), [
    el("DIV", [text("b")]),
    el("DIV", [text("c")]),
  ]);
  assert.equal(index.text, `a${BLOCK_SEPARATOR}b${BLOCK_SEPARATOR}c`);
});

test("a portaled surface with nothing to contribute leaves no separator behind", () => {
  // The boundary is pending until text follows it, so a surface that is skipped, or empty, does
  // not leave a gap at the end of the index for a match to be measured against.
  const scope = el("DIV", [el("P", [text("only")])]);
  const index = buildTextIndex(scope, [
    el("DIV", [text("parked")], { inert: "" }),
    el("DIV"),
  ]);
  assert.equal(index.text, "only");
});

/** Run `body` with `getComputedStyle` answering from `styles`, and `{}` for anything else. */
function withStyles(
  styles: Map<
    unknown,
    {
      display?: string;
      visibility?: string;
      whiteSpace?: string;
      clip?: string;
      clipPath?: string;
    }
  >,
  body: () => void,
): void {
  const view = globalThis as { getComputedStyle?: unknown };
  const saved = view.getComputedStyle;
  view.getComputedStyle = (element: unknown) => styles.get(element) ?? {};
  try {
    body();
  } finally {
    view.getComputedStyle = saved;
  }
}

// --- the observer's own filter -----------------------------------------------------------------

/** The two bits of `Element` the filter touches, so a record can be handed over without a DOM. */
function skipNode(options: {
  skipped?: boolean;
  /** Whatever takes this element out of the index, spelled as the selector spells it. */
  mark?: string;
  parent?: ReturnType<typeof skipNode> | null;
}): {
  nodeType: number;
  parentElement: Element | null;
  closest: (s: string) => Element | null;
} {
  const node = {
    nodeType: 1,
    mark:
      options.mark ??
      (options.skipped === true ? `[${FIND_SKIP_ATTRIBUTE}]` : null),
    parent: options.parent ?? null,
    get parentElement() {
      return node.parent as unknown as Element | null;
    },
    closest(selector: string): Element | null {
      let at: typeof node | null = node;
      while (at) {
        if (at.mark !== null && selector.includes(at.mark)) {
          return at as unknown as Element;
        }
        at = at.parent as typeof node | null;
      }
      return null;
    },
  };
  return node as unknown as ReturnType<typeof skipNode>;
}

function record(
  target: ReturnType<typeof skipNode>,
  type = "childList",
  attributeName: string | null = null,
) {
  return { target, type, attributeName } as unknown as Parameters<
    typeof mutatesSearchableText
  >[0];
}

test("the selection fallback hands the caret back to the field", async () => {
  // Moving the selection into ordinary text takes the caret with it on WebKit and Blink: the field
  // still reports as active but every keystroke is swallowed, so the query freezes at one
  // character. Measured with the registry removed, the field held "u" and never grew. This matters
  // exactly where the fallback runs: Firefox below 140, or the desktop build's WebKitGTK.
  const engine = await readFile(
    new URL("../src/features/find-in-page/lib/find-dom.ts", import.meta.url),
    "utf8",
  );
  const fallback = engine.slice(
    engine.indexOf("export function selectRangeFallback"),
  );
  const body = fallback.slice(0, fallback.indexOf("\n}"));
  assert.match(body, /holdCaret\(\)/);
  assert.match(body, /releaseCaret\(/);
  // The caret has to be taken BEFORE the selection moves and given back after, or there is nothing
  // left to restore.
  assert.ok(
    body.indexOf("holdCaret()") < body.indexOf("selection.addRange"),
    "the caret must be captured before the selection is moved",
  );
  assert.ok(
    body.indexOf("releaseCaret(") > body.indexOf("selection.addRange"),
    "the caret must be restored after the selection is moved",
  );
});

test("a workspace generating off-route does not rebuild the index", () => {
  // `__root.tsx` keeps Chat, Images, Video and Audio mounted under `hidden` and `inert` precisely
  // so a long generation is not cancelled by navigating away, and they sit INSIDE the scope. Every
  // character such a reply streams is a mutation the bar used to answer with a full flatten, which
  // then correctly excluded that text: the whole rebuild was for nothing, once per throttle for as
  // long as the generation ran.
  for (const mark of ["[inert]", "[hidden]", '[aria-hidden="true"]']) {
    const parked = skipNode({ mark });
    const streamed = skipNode({ parent: parked });
    assert.equal(
      mutatesSearchableText(record(streamed, "characterData")),
      false,
      `a reply streaming under ${mark} still asked for a rebuild`,
    );
  }
  // And an ordinary reply in the workspace on screen still does.
  const live = skipNode({ parent: skipNode({}) });
  assert.equal(mutatesSearchableText(record(live, "characterData")), true);
});

test("parking a workspace is itself a change, whichever attribute says so", () => {
  // The same trap as the skip attribute below: `closest` matches the element it starts at, so the
  // record announcing that a workspace just went `inert` would answer "inside skipped content" and
  // be dropped, leaving the workspace the reader just left in the count.
  for (const mark of ["[inert]", "[hidden]", '[aria-hidden="true"]']) {
    const parked = skipNode({ mark, parent: skipNode({}) });
    assert.equal(
      mutatesSearchableText(record(parked, "attributes", "inert")),
      true,
      `${mark} being added was filtered out`,
    );
  }
});

test("adding the skip attribute is what reindexes, not only removing it", () => {
  // `closest` matches where it starts, so the record announcing that an element became skippable
  // answers "inside skipped content" and is thrown away, leaving the region counted and painted
  // until some unrelated mutation happens by. Removal always worked, which is what hid this.
  const parent = skipNode({});
  const marked = skipNode({ skipped: true, parent });
  assert.equal(
    mutatesSearchableText(record(marked, "attributes", FIND_SKIP_ATTRIBUTE)),
    true,
    "gaining the attribute must schedule a rebuild",
  );

  const unmarked = skipNode({ skipped: false, parent });
  assert.equal(
    mutatesSearchableText(record(unmarked, "attributes", FIND_SKIP_ATTRIBUTE)),
    true,
    "losing the attribute must still schedule a rebuild",
  );
});

test("ordinary mutations inside skipped content are still ignored", () => {
  // The point of the filter: the bar floats inside the region it searches, so its own counter
  // re-rendering must not order a re-index of itself.
  const bar = skipNode({ skipped: true });
  const inside = skipNode({ parent: bar });
  assert.equal(mutatesSearchableText(record(inside)), false);
  assert.equal(mutatesSearchableText(record(bar)), false);
});

test("a mutation in ordinary content always reindexes", () => {
  const thread = skipNode({});
  const message = skipNode({ parent: thread });
  assert.equal(mutatesSearchableText(record(message)), true);
  assert.equal(mutatesSearchableText(record(message, "characterData")), true);
});

test("a detached target counts as a change rather than being dropped", () => {
  // No parent to ask, so the conservative answer is the safe one.
  const orphan = skipNode({ skipped: true, parent: null });
  assert.equal(
    mutatesSearchableText(record(orphan, "attributes", FIND_SKIP_ATTRIBUTE)),
    true,
  );
});

test("an attribute that is not the skip flag is judged from the target itself", () => {
  // Chrome stays chrome: `inert` flipping inside a skipped region is still skipped.
  const bar = skipNode({ skipped: true });
  const inside = skipNode({ parent: bar });
  assert.equal(
    mutatesSearchableText(record(inside, "attributes", "inert")),
    false,
  );
});

test("a display:contents wrapper that is itself invisible keeps its own text out", () => {
  // `skipsSubtree` lets a boxless wrapper through on purpose, since `checkVisibility` calls
  // anything with no box invisible and the shell wraps visible content in one. But `visibility`
  // inherits and only ELEMENT children are re-checked, so a direct text child of a hidden
  // `contents` wrapper was indexed, counted and painted while nobody could see it.
  const ghost = el("SPAN", [text("invisible")]);
  (ghost as { checkVisibility?: () => boolean }).checkVisibility = () => false;
  withStyles(
    new Map([[ghost, { display: "contents", visibility: "hidden" }]]),
    () => {
      const index = buildTextIndex(el("DIV", [ghost]));
      assert.equal(index.text.includes("invisible"), false);
      assert.deepEqual(findMatches(index, "invisible"), []);
    },
  );
});

test("a visible display:contents wrapper is still searched", () => {
  // The other half: the rescue has to keep working, or most of what there is to search goes.
  const wrapper = el("SPAN", [text("findable")]);
  (wrapper as { checkVisibility?: () => boolean }).checkVisibility = () =>
    false;
  withStyles(new Map([[wrapper, { display: "contents" }]]), () => {
    const index = buildTextIndex(el("DIV", [wrapper]));
    assert.equal(index.text.includes("findable"), true);
    assert.equal(findMatches(index, "findable").length, 1);
  });
});

test("an element child that restores visibility inside a hidden contents wrapper is kept", () => {
  // Scoped to the wrapper's OWN text: `visibility: visible` paints again and the walk does not
  // turn back, so that child has to survive.
  const inner = el("SPAN", [text("restored")]);
  const ghost = el("SPAN", [text("invisible"), inner]);
  (ghost as { checkVisibility?: () => boolean }).checkVisibility = () => false;
  withStyles(
    new Map<unknown, { display?: string; visibility?: string }>([
      [ghost, { display: "contents", visibility: "hidden" }],
      [inner, { display: "inline", visibility: "visible" }],
    ]),
    () => {
      const index = buildTextIndex(el("DIV", [ghost]));
      assert.equal(index.text.includes("invisible"), false);
      assert.equal(index.text.includes("restored"), true);
    },
  );
});

test("the match window anchor is resolved only once the cap bites", () => {
  // `viewportOffset` reads layout, and an argument is evaluated whether or not the callee wants
  // it, so inline it ran on every keystroke however few matches there were. As a thunk it is paid
  // only where it changes the answer: when the cap actually cuts the list short.
  const index = buildTextIndex(el("P", [text("a a a a a a a a")]));

  let asked = 0;
  const anchor = () => {
    asked += 1;
    return 6;
  };

  const underCap = findMatches(index, "a", 100, anchor);
  assert.equal(underCap.length, 8);
  assert.equal(asked, 0, "an under-cap query must not read layout");

  const capped = findMatches(index, "a", 3, anchor);
  assert.equal(asked, 1, "a capped query resolves the anchor exactly once");
  // A thunk and the number it returns must pick the same window.
  assert.deepEqual(capped, findMatches(index, "a", 3, 6));
});

test("a decomposed dotted I is found by the ordinary query", () => {
  // U+0130 decomposes to `I` + a combining dot, which folds to `i` + that dot. `i` + dot has no
  // precomposed form, so NFC cannot put it back and the plain query missed a word on screen while
  // the precomposed spelling of the same word matched.
  const decomposed = "\u0049\u0307stanbul";
  // The fold is what strands it: `I` + dot lowercases to `i` + dot, and THAT has no precomposed
  // form, so no amount of normalizing the query reaches it.
  assert.equal("\u0069\u0307".normalize("NFC"), "\u0069\u0307");
  const index = buildTextIndex(el("P", [text(`Welcome to ${decomposed}`)]));
  for (const query of ["istanbul", "ISTANBUL", "\u0130stanbul"]) {
    assert.equal(findMatches(index, query).length, 1, query);
  }
});

test("the dotted variant costs nothing on a document without combining marks", () => {
  // It is only offered when the index carries a combining dot, so an ordinary thread keeps the
  // single-variant `indexOf` path.
  const index = buildTextIndex(el("P", [text("indexing is fine here")]));
  assert.equal(findMatches(index, "indexing").length, 1);
  assert.equal(findMatches(index, "i").length, 4);
});

test("a query too large to compile falls back instead of throwing", () => {
  // Every engine caps the pattern size it will compile and the spec sets none, so there is no
  // length that is right everywhere. Measured on V8: a whitespace-bearing query throws at 15,651
  // characters, and the throw came out through the keystroke and took the bar down with it.
  const index = buildTextIndex(el("P", [text("a small thread about unsloth")]));
  const huge = "some log line with spaces ".repeat(4000);
  assert.ok(huge.length > 15_651, "premise: past the measured V8 ceiling");
  assert.doesNotThrow(() => findMatches(index, huge));
  assert.deepEqual(findMatches(index, huge), []);
});

test("a needle longer than the haystack is rejected before any of the work", () => {
  const index = buildTextIndex(el("P", [text("short")]));
  assert.deepEqual(findMatches(index, "x".repeat(500)), []);
  // Measured against the shortest spelling: a decomposed query is longer than the precomposed
  // text it is meant to find, so the raw length is the wrong thing to test.
  const cafe = buildTextIndex(el("P", [text("caf\u00e9")]));
  assert.equal(findMatches(cafe, "cafe\u0301").length, 1);
  // And a needle that still fits is unaffected.
  assert.equal(findMatches(index, "short").length, 1);
});

test("a numeric anchor still means what it always did", () => {
  // The thunk is additive. Every existing caller passes a number and must be unaffected.
  const index = buildTextIndex(el("P", [text("b b b b b b b b")]));
  assert.deepEqual(findMatches(index, "b", 3, 0), findMatches(index, "b", 3));
  assert.deepEqual(
    findMatches(index, "b", 3, 10),
    findMatches(index, "b", 3, () => 10),
  );
});

test("a word matches whichever way either side spells it", () => {
  // The same word composed and decomposed. macOS hands back decomposed filenames while a model
  // writes composed prose, so one thread holds both, and the platform's own find matches either
  // from either. Measured: all four pairings hit.
  const composed = "caf\u00e9";
  const decomposed = "cafe\u0301";
  assert.notEqual(composed, decomposed);
  for (const written of [composed, decomposed]) {
    for (const typed of [composed, decomposed]) {
      const index = buildTextIndex(
        el("DIV", [el("P", [text(`a ${written} b`)])]),
      );
      const matches = findMatches(index, typed);
      assert.equal(
        matches.length,
        1,
        `text ${escape(written)} and query ${escape(typed)} did not meet`,
      );
      // And the offsets are the document's, not a normalized copy's: the match has to cover
      // exactly the characters that were written, whatever length that spelling is.
      assert.deepEqual(matches[0], { start: 2, end: 2 + written.length });
      assert.equal(index.text.slice(matches[0].start, matches[0].end), written);
    }
  }
});

test("an occurrence that mixes the two spellings is still one word", () => {
  // Alternating whole spellings of the query only reaches text that is all-composed or
  // all-decomposed. Joining two text nodes joins two sources, so one visible word can be neither,
  // and then no spelling the query CAN be written in matches it. Every engine's own find reaches
  // this: measured `true` on chromium, firefox and webkit for both all-composed and all-decomposed
  // queries against a mixed occurrence.
  const composed = "é";
  const decomposed = "é";
  const mixed = `caf${composed}caf${decomposed}`;
  for (const typed of [
    `caf${composed}caf${composed}`,
    `caf${decomposed}caf${decomposed}`,
    mixed,
  ]) {
    const index = buildTextIndex(el("DIV", [el("P", [text(`a ${mixed} b`)])]));
    const matches = findMatches(index, typed);
    assert.equal(
      matches.length,
      1,
      `query ${escape(typed)} missed a mixed word`,
    );
    // Still the document's own offsets, so the highlight covers what was written.
    assert.deepEqual(matches[0], { start: 2, end: 2 + mixed.length });
    assert.equal(index.text.slice(matches[0].start, matches[0].end), mixed);
  }

  // The shape that produces it here: one word, two inline nodes, one source each.
  const split = buildTextIndex(
    el("DIV", [el("P", [text(`caf${composed}`), text(`caf${decomposed}`)])]),
  );
  assert.equal(findMatches(split, `caf${composed}caf${composed}`).length, 1);
});

test("the index itself is left in the form the document wrote", () => {
  // Normalizing it is the other way to fix the above, and it would change its length: every offset
  // in the index stands for one character of a text node, so a shorter index misplaces them all.
  const decomposed = "cafe\u0301";
  const index = buildTextIndex(el("DIV", [el("P", [text(decomposed)])]));
  assert.equal(index.text, decomposed);
  assert.equal(index.text.length, decomposed.length);
});

test("spelling variants do not loosen whitespace inside a fence", () => {
  // The variants share the pattern path with the flexible-whitespace one, and inside a `<pre>` the
  // whitespace on screen is the whitespace in the node. A variant is exact; a flexed run is not.
  const fence = el("PRE", [text("caf\u00e9   au lait")]);
  withStyles(new Map([[fence, { whiteSpace: "pre" }]]), () => {
    const index = buildTextIndex(el("DIV", [fence]));
    assert.equal(findMatches(index, "caf\u00e9 au lait").length, 0);
    assert.equal(findMatches(index, "cafe\u0301   au lait").length, 1);
  });
});

test("an engine with no checkVisibility falls back to the computed properties", () => {
  // `checkVisibility` landed in Safari 17.4, and WebKitGTK is already supported here: it is the
  // engine `selectRangeFallback` exists for. The optional call answers undefined there, and read as
  // "not false" that put every `display: none` subtree in the app back into the index.
  for (const style of [
    { display: "none" },
    { visibility: "hidden" },
    { visibility: "collapse" },
  ]) {
    const buried = el("DIV", [text("buried")]);
    const root = el("DIV", [el("P", [text("visible")]), buried]);
    withStyles(new Map([[buried, style]]), () => {
      const index = buildTextIndex(root);
      assert.equal(
        index.text.includes("buried"),
        false,
        `${JSON.stringify(style)} leaked into the index`,
      );
      assert.equal(index.text.includes("visible"), true);
    });
  }
});

test("with no checkVisibility, a hidden boxless wrapper is still descended into", () => {
  // Where the two mechanisms meet. `display: contents` is boxless rather than hidden, so the
  // fallback must not skip it whole; `hidesOwnText` drops the text it holds directly, and a child
  // that turns visibility back on is painted and still has to be found.
  const shown = el("SPAN", [text("turned back on")]);
  const wrapper = el("DIV", [text("the wrapper's own text"), shown]);
  withStyles(
    new Map([
      [wrapper, { display: "contents", visibility: "hidden" }],
      [shown, { visibility: "visible" }],
    ]),
    () => {
      const index = buildTextIndex(el("DIV", [wrapper]));
      assert.equal(index.text.includes("the wrapper"), false);
      assert.equal(index.text.includes("turned back on"), true);
    },
  );
});

test("the fallback does not mistake a boxless wrapper for a hidden one", () => {
  // `display: contents` is the case the whole visibility branch was written around: it has no box
  // and is not hidden, and the shell hands a grid its children through one.
  const wrapper = el("DIV", [el("P", [text("inside a wrapper")])]);
  withStyles(new Map([[wrapper, { display: "contents" }]]), () => {
    assert.equal(buildTextIndex(el("DIV", [wrapper])).text, "inside a wrapper");
  });
});

test("two spans the CSS renders as blocks do not run together", () => {
  // No tag name says these are blocks, and Tailwind stacks them all over the app: a source's title
  // over its URL in the research panel is exactly this shape. Run together they invent a word that
  // is on screen nowhere, under one highlight spanning two rows.
  const first = el("SPAN", [text("Open")]);
  const second = el("SPAN", [text("AI models")]);
  withStyles(
    new Map([
      [first, { display: "block" }],
      [second, { display: "block" }],
    ]),
    () => {
      const index = buildTextIndex(el("DIV", [first, second]));
      assert.equal(index.text.includes("openai"), false);
      assert.deepEqual(findMatches(index, "openai"), []);
      // Each row still matches on its own.
      assert.equal(findMatches(index, "open").length, 1);
      assert.equal(findMatches(index, "ai models").length, 1);
    },
  );
  // An inline span is not a boundary, so markup inside a sentence still reads as one word.
  const inline = el("SPAN", [text("slo")]);
  withStyles(new Map([[inline, { display: "inline" }]]), () => {
    const index = buildTextIndex(el("P", [text("un"), inline, text("th")]));
    assert.equal(findMatches(index, "unsloth").length, 1);
  });
});

test("whitespace is only flexible where the page collapses it", () => {
  // The flexible run exists for prose, where a markdown soft wrap puts a newline in the node that
  // renders as a space. In a code fence the whitespace on screen IS the whitespace in the node, so
  // a query typed with one space must not land on three. The platform's own find draws the same
  // line: matching across a wrap in a paragraph, and not inside a `<pre>`.
  const fence = el("PRE", [text("unsloth   fast")]);
  const prose = el("P", [text("unsloth\n   fast")]);
  withStyles(
    new Map([
      [fence, { whiteSpace: "pre" }],
      [prose, { whiteSpace: "normal" }],
    ]),
    () => {
      const fenced = buildTextIndex(el("DIV", [fence]));
      const wrapped = buildTextIndex(el("DIV", [prose]));
      assert.deepEqual(findMatches(fenced, "unsloth fast"), []);
      assert.equal(findMatches(fenced, "unsloth   fast").length, 1);
      // Prose is unchanged: one space still crosses the wrap.
      assert.equal(findMatches(wrapped, "unsloth fast").length, 1);
      // And the flag rides on the segment, not the element, so it survives the offset map.
      assert.equal(fenced.segments[0].preserved, true);
      assert.equal(wrapped.segments[0].preserved, false);
    },
  );
});

test("preserved whitespace is inherited by the nodes inside it", () => {
  const fence = el("PRE", [el("CODE", [text("a   b")])]);
  withStyles(new Map([[fence, { whiteSpace: "pre" }]]), () => {
    const index = buildTextIndex(el("DIV", [fence]));
    // `CODE` gets `{}` from the fake, so this only passes if the walk carries the mode down.
    assert.equal(index.segments[0].preserved, true);
    assert.deepEqual(findMatches(index, "a b"), []);
  });
});

test("a boxless wrapper is walked through, not skipped", () => {
  // `display: contents` generates no box, and no box is the first thing `checkVisibility` calls
  // invisible. The shell (sidebar.tsx) and the training page (studio-page.tsx) each wrap their
  // content in one, so reading that answer as hidden empties the index for most of the app.
  const wrapper = {
    ...el("DIV", [el("P", [text("training")])]),
    checkVisibility: () => false,
  };
  const collapsed = {
    ...el("DIV", [el("P", [text("offscreen")])]),
    checkVisibility: () => false,
  };
  const display = new Map<unknown, string>([
    [wrapper, "contents"],
    [collapsed, "none"],
  ]);
  const view = globalThis as { getComputedStyle?: unknown };
  const saved = view.getComputedStyle;
  view.getComputedStyle = (element: unknown) => ({
    display: display.get(element) ?? "block",
  });
  try {
    const index = buildTextIndex(el("DIV", [wrapper, collapsed]));
    assert.equal(index.text.includes("training"), true);
    // A wrapper with a box that says invisible is still hidden.
    assert.equal(index.text.includes("offscreen"), false);
  } finally {
    view.getComputedStyle = saved;
  }
});

test("a query spanning whitespace matches the phrase as it renders", () => {
  // HTML collapses runs of whitespace, so a markdown paragraph soft-wrapped mid-sentence renders
  // as one line while its text node still holds the newline.
  const index = buildTextIndex(
    el("DIV", [el("P", [text("A soft wrapped\n      phrase about unsloth.")])]),
  );
  assert.equal(findMatches(index, "wrapped phrase").length, 1);
  // The match is as wide as the run it covered, so the highlight lands on the whole phrase.
  const [match] = findMatches(index, "wrapped phrase");
  assert.equal(match.end - match.start, "wrapped\n      phrase".length);
});

test("a query spanning whitespace still cannot cross a block boundary", () => {
  const index = buildTextIndex(
    el("DIV", [el("P", [text("the end")]), el("P", [text("start here")])]),
  );
  assert.deepEqual(findMatches(index, "end start"), []);
});

test("a regex metacharacter in a query is a literal", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("a.b and axb c")])]));
  // Only reaches the pattern path because of the space, which is the point.
  assert.equal(findMatches(index, "a.b and").length, 1);
  assert.deepEqual(findMatches(index, "axb and"), []);
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

test("a clipped node does not run into the next one", () => {
  // What the clip threw away is still in the document. Without a boundary the retained prefix and
  // the next node touch, a query across the seam matches, and the Range it maps back to spans every
  // discarded character in between: measured at 8503 characters of unrelated highlight.
  const clipped = text(
    `${"x".repeat(MAX_NODE_CHARS)}${"discarded ".repeat(500)}`,
  );
  const next = text("yz");
  const index = buildTextIndex(el("DIV", [el("P", [clipped, next])]));
  assert.equal(index.truncated, true);
  assert.deepEqual(findMatches(index, "xy"), []);
  // The separator is what closes it, and both sides are still findable on their own.
  assert.equal(index.text.includes(BLOCK_SEPARATOR), true);
  assert.equal(findMatches(index, "yz").length, 1);
});

test("the ceiling holds across a block boundary", () => {
  // A node landing exactly on the ceiling used to let the next block's separator push `length`
  // past it, and the negative `room` that followed made `slice(0, room)` take all but the last
  // character of the following node: a 500,000 character overshoot on a 4,000,000 cap.
  // Filled a node at a time now that no single one may take the lot, landing exactly on the cap.
  const blocks = Array.from({ length: MAX_INDEX_CHARS / MAX_NODE_CHARS }, () =>
    el("P", [text("x".repeat(MAX_NODE_CHARS))]),
  );
  const index = buildTextIndex(
    el("DIV", [...blocks, el("P", [text("y".repeat(500_000))])]),
  );
  assert.equal(index.text.length, MAX_INDEX_CHARS);
  assert.equal(index.truncated, true);
  assert.deepEqual(findMatches(index, "yyy", 5), []);
});

test("nothing lands past the ceiling however the walk arrives at it", () => {
  // Every shape that reaches the cap: one oversized node, many small ones, and a boundary in the
  // middle. None may report an index longer than the cap it was given.
  const shapes = [
    [el("P", [text("x".repeat(MAX_INDEX_CHARS + 1_000))])],
    Array.from({ length: 9 }, () => el("P", [text("x".repeat(500_000))])),
    [
      el("P", [text("x".repeat(MAX_INDEX_CHARS - 1))]),
      el("P", [text("y".repeat(1_000))]),
    ],
  ];
  for (const [i, children] of shapes.entries()) {
    const index = buildTextIndex(el("DIV", children));
    assert.ok(
      index.text.length <= MAX_INDEX_CHARS,
      `shape ${i} indexed ${index.text.length}, past the ${MAX_INDEX_CHARS} cap`,
    );
    assert.equal(index.truncated, true, `shape ${i} did not report truncation`);
  }
});

test("a document inside the ceiling is not marked truncated", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("unsloth")])]));
  assert.equal(index.truncated, false);
});

// --- the probe over the cap --------------------------------------------------------------------

/** One node holding `count` occurrences of "x", each at its own offset. */
function documentOfMatches(count: number): FindElementLike {
  return el("DIV", [text("x-".repeat(count))]);
}

/** What `search` does: ask for one over the cap, remember the anchor, then trim. */
function walkAsTheHookDoes(
  index: ReturnType<typeof buildTextIndex>,
  at: number,
) {
  let anchoredAt: number | null = null;
  const matches = findMatches(index, "x", MAX_MATCHES + 1, () => {
    anchoredAt = at;
    return at;
  });
  const capped = matches.length > MAX_MATCHES;
  if (capped) dropProbeFurthestFrom(matches, anchoredAt);
  return { matches, capped };
}

test("the last match in the document is reachable from the bottom of it", () => {
  // The probe asked for over the cap used to come off the tail unconditionally. Once the reader is
  // far enough down the window IS the tail, so that threw away the occurrence beside them.
  const index = buildTextIndex(documentOfMatches(MAX_MATCHES + 1_000));
  const all = findMatches(index, "x", Number.POSITIVE_INFINITY, 0);
  const last = all[all.length - 1].start;

  const { matches, capped } = walkAsTheHookDoes(index, index.text.length);
  assert.equal(capped, true);
  assert.equal(matches.length, MAX_MATCHES);
  assert.ok(
    matches.some((match) => match.start === last),
    "the final occurrence must still be walkable",
  );
});

test("the first match is still reachable from the top, which is the case that worked", () => {
  const index = buildTextIndex(documentOfMatches(MAX_MATCHES + 1_000));
  const { matches } = walkAsTheHookDoes(index, 0);
  assert.equal(matches.length, MAX_MATCHES);
  assert.equal(matches[0].start, 0);
});

test("the window holds the match nearest the reader, wherever they are", () => {
  const index = buildTextIndex(documentOfMatches(MAX_MATCHES + 2_500));
  const all = findMatches(index, "x", Number.POSITIVE_INFINITY, 0);
  for (const fraction of [0, 0.25, 0.5, 0.75, 1]) {
    const at = Math.floor(index.text.length * fraction);
    const nearest =
      all.find((match) => match.start >= at) ?? all[all.length - 1];
    const { matches } = walkAsTheHookDoes(index, at);
    assert.ok(
      matches.some((match) => match.start === nearest.start),
      `the match beside the reader is missing at ${fraction}`,
    );
  }
});

test("the trim takes the far end, and the tail when there is no anchor to judge by", () => {
  const window = () => [
    { start: 100, end: 101 },
    { start: 200, end: 201 },
    { start: 300, end: 301 },
  ];
  const above = window();
  dropProbeFurthestFrom(above, 320, 2);
  assert.deepEqual(
    above.map((match) => match.start),
    [200, 300],
    "a reader past the window gives up the head",
  );

  const below = window();
  dropProbeFurthestFrom(below, 90, 2);
  assert.deepEqual(
    below.map((match) => match.start),
    [100, 200],
    "a reader above the window gives up the tail",
  );

  const unanchored = window();
  dropProbeFurthestFrom(unanchored, null, 2);
  assert.deepEqual(
    unanchored.map((match) => match.start),
    [100, 200],
    "no anchor resolved means the window started at the top",
  );
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
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  const composer = cssRule(css, ".unsloth-composer-surface");
  const bar = cssRule(css, ".find-bar-surface");

  // The composer is the reference, not a copied colour: if it restyles, this fails rather than
  // leaving the one floating panel a shade off from the one below it.
  const background = /background-color:\s*(#[0-9a-f]{6});/i.exec(composer);
  assert.ok(background);
  assert.match(bar, new RegExp(`background-color:\\s*${background[1]};`, "i"));

  // The shadow is the composer's, spread wider and softened: that one sits at the bottom of the
  // page with nothing under it, this one floats over content and needs more to sit on, not more
  // weight.
  const shape = (rule: string) => {
    const hit =
      /box-shadow:\s*0 (\d+)px (\d+)px (-?\d+)px rgba\(0, 0, 0, ([\d.]+)\);/.exec(
        rule,
      );
    assert.ok(hit, "box-shadow is not in the shape this test reads");
    return {
      y: Number(hit[1]),
      blur: Number(hit[2]),
      spread: Number(hit[3]),
      alpha: Number(hit[4]),
    };
  };
  const from = shape(composer);
  const to = shape(bar);
  assert.ok(
    to.blur > from.blur,
    `blur ${to.blur} is not wider than ${from.blur}`,
  );
  assert.ok(
    to.spread > from.spread,
    `spread ${to.spread} is not wider than ${from.spread}`,
  );
  // Wider, never heavier: the width is what lifts it off the page, not the ink.
  assert.ok(
    to.alpha < from.alpha,
    `alpha ${to.alpha} is not softer than ${from.alpha}`,
  );
  // But still a shadow, and still in the composer's family.
  assert.ok(
    to.alpha >= from.alpha * 0.7,
    `alpha ${to.alpha} has faded to nothing`,
  );
  assert.ok(
    to.blur <= from.blur * 2,
    `blur ${to.blur} is more than slightly wider`,
  );
  assert.ok(to.y >= from.y);
});

test("dark mode sits above the cards it floats over", async () => {
  const css = await readFile(
    new URL("../src/index.css", import.meta.url),
    "utf8",
  );
  const value = (selector: string, property: string) => {
    const hit = new RegExp(`${property}:\\s*([^;]+);`).exec(
      cssRule(css, selector),
    );
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
  assert.match(
    value(".dark .find-bar-surface", "box-shadow"),
    /var\(--background\)/,
  );
});

test("the bar stays out of a backgrounded scope, and off the document origin", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Every modal, not just Settings: Radix marks the shell aria-hidden for as long as one is up,
  // and `enabled` is read at render, which a dialog opening need not cause.
  assert.match(
    bar,
    /isSurfaceBackgrounded\(`\[\$\{FIND_SCOPE_ATTRIBUTE\}\]`\)/,
  );
  // Fixed, not absolute: on a route whose outer container scrolls, an absolutely positioned bar
  // sits at the top of a scope taller than the window and scrolls out of reach.
  const surface = /className="(find-bar-surface[^"]*)"/.exec(bar);
  assert.ok(surface);
  assert.match(surface[1], /\bfixed\b/);
  assert.equal(/\babsolute\b/.test(surface[1]), false);
  // And capped, so a narrow window cannot push it off the left edge.
  assert.match(surface[1], /max-w-\[calc\(100vw-2rem\)\]/);
});

test("the reveal looks again while the scroll is still moving", async () => {
  // A `content-visibility: auto` subtree contributes its placeholder height to scrollHeight until
  // it renders, so the first scroll is clamped short and reaching toward the block is what makes it
  // render. Measured in a real viewport: 3415px short on all three engines without this. The node
  // suite cannot see a scroll, so what is pinned here is the shape the browser harness relies on.
  const dom = await readFile(
    new URL("../src/features/find-in-page/lib/find-dom.ts", import.meta.url),
    "utf8",
  );
  // The scroll reports whether it moved anything, which is the whole signal.
  assert.match(
    dom,
    /export function scrollRangeIntoView\(range: Range\): boolean/,
  );
  const reveal = dom.slice(dom.indexOf("function revealPass("));
  const body = reveal.slice(0, reveal.indexOf("\n}\n"));
  // Stops as soon as a pass moves nothing, and is bounded so nothing can spin.
  assert.match(
    body,
    /if \(!scrollRangeIntoView\(range\) \|\| tries <= 1\) return;/,
  );
  assert.match(body, /tries - 1/);
  assert.match(dom, /revealRangeWhenPainted\(range: Range, tries = \d\)/);
  // Next frame, not a timer: what is being waited for is a paint.
  assert.match(body, /requestAnimationFrame\(/);
  // And a range whose nodes a streaming reply has replaced is dropped rather than scrolled to.
  assert.match(body, /range\.startContainer\.isConnected/);
  // The engine asks for the retrying one, or the second look never happens.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(engine, /revealRangeWhenPainted\(activeRange\)/);
  assert.equal(/scrollRangeIntoView\(activeRange\)/.test(engine), false);
});

test("a dismissed or superseded search abandons its queued reveal passes", async () => {
  // The reveal chain walks up to seven more frames after the first scroll. The workspace stays
  // mounted when the bar closes, so `isConnected` stays true and the old chain would keep scrolling
  // the reader toward a match they already dismissed. A generation token retires it.
  const dom = await readFile(
    new URL("../src/features/find-in-page/lib/find-dom.ts", import.meta.url),
    "utf8",
  );
  // Every new reveal retires the previous one, so two navigations cannot scroll against each other.
  const entry = dom.slice(
    dom.indexOf("export function revealRangeWhenPainted"),
  );
  assert.match(
    entry.slice(0, entry.indexOf("\n}\n")),
    /cancelRevealPasses\(\)/,
  );
  // And the queued frame checks the token it was queued under before scrolling again.
  const pass = dom.slice(dom.indexOf("function revealPass("));
  assert.match(
    pass.slice(0, pass.indexOf("\n}\n")),
    /if \(generation !== revealGeneration\) return;/,
  );
  // Teardown retires the chain: closing the bar is exactly when the reader stops asking to move.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(engine, /cancelRevealPasses\(\);/);
});

test("a query too large to compile falls back instead of throwing on the first scan", () => {
  // V8 compiles a regex lazily, so an oversized pattern is accepted by the constructor and throws
  // `SyntaxError` on the first `exec` instead. Guarding only the constructor left the throw coming
  // out through the keystroke that caused it, which tears the bar out of the DOM. Whitespace is
  // what gets a query there: every run becomes `\\s+`, so a spaced paste doubles on the way in.
  const index = buildTextIndex(el("DIV", [el("P", [text("b".repeat(60000))])]));
  // Longer than the compiler will take once the whitespace flexes, shorter than the haystack, so
  // the length guard cannot short-circuit it before the pattern is built.
  const query = "a ".repeat(10000).trim();
  assert.ok(query.length < index.text.length);
  // The premise: this pattern really does survive construction and die on use.
  const escaped = query.replace(/\s+/g, "\\s+");
  let lazy = false;
  try {
    new RegExp(escaped, "g").exec("");
  } catch {
    lazy = true;
  }
  assert.equal(
    lazy,
    true,
    "premise: the pattern throws at compile-on-first-use",
  );
  // No throw, and no matches: the literal scan is exact, so a spaced query simply finds nothing.
  assert.deepEqual(findMatches(index, query, 10), []);
});

test("a match with no geometry is aimed at through its nearest laid-out ancestor", async () => {
  const dom = await readFile(
    new URL("../src/features/find-in-page/lib/find-dom.ts", import.meta.url),
    "utf8",
  );
  // Text inside a skipped subtree has a collapsed rect while the subtree's own box keeps its
  // placeholder geometry, so the walk aims at that instead of giving up.
  assert.match(
    dom,
    /export function revealRect\(range: Range\): DOMRect \| null/,
  );
  assert.match(dom, /export function rangeTop\(range: Range\): number \| null/);
  // And both readers go through it rather than reading the range rect directly.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(engine, /const top = rangeTop\(range\);/);
  assert.equal(/range\.getBoundingClientRect\(\)/.test(engine), false);
});

test("a fresh query starts from the scroll container's top, not the window's", async () => {
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // The thread viewport starts below the navbar and the chat header, so a match clipped just off
  // the top of it still has a positive window-relative `top`.
  assert.match(engine, /top >= scrollViewportTop\(range\)/);
  assert.equal(/top >= 0/.test(engine), false);
});

test("re-indexing while the document changes is a throttle, and says so", async () => {
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // A debounce would leave the count frozen and new text unfindable for as long as a reply takes
  // to write. The name has to match the behaviour, which is what went wrong before.
  assert.match(engine, /REINDEX_INTERVAL_MS/);
  assert.equal(/REINDEX_DEBOUNCE_MS/.test(engine), false);
  assert.match(engine, /A throttle rather than a debounce/);
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
  // `open` too: toggling a `<details>` changes that and nothing else, while the body inside it
  // goes from visible to not, so a collapsible opened after indexing would stay unfindable.
  assert.match(engine, /attributeFilter: \[[^\]]*"open"/);
  // And not the whole attribute stream: `class` changes on every hover. Scanned rather than
  // matched, since the comments in between make a regex for this one backtrack badly.
  const opensAttributes = engine.indexOf("attributes: true,");
  const opensFilter = engine.indexOf("attributeFilter:", opensAttributes);
  assert.ok(opensAttributes !== -1 && opensFilter > opensAttributes);
  const between = engine.slice(
    opensAttributes + "attributes: true,".length,
    opensFilter,
  );
  assert.equal(
    between
      .split("\n")
      .every((line) => line.trim() === "" || line.trim().startsWith("//")),
    true,
    "nothing else is being observed between the flag and its filter",
  );
  assert.equal(engine.includes('attributeFilter: ["class"'), false);
});

/** Stand in for the document `resolvePortalSurfaces` queries: node has no version of one. */
function withPortals<T>(surfaces: Element[], body: () => T): T {
  const view = globalThis as { document?: unknown };
  const had = "document" in view;
  const saved = view.document;
  view.document = { querySelectorAll: () => surfaces };
  try {
    return body();
  } finally {
    if (had) view.document = saved;
    else delete view.document;
  }
}

/** A portaled surface, which is asked only what state it is in and what it contains. */
function surface(state: string | null, children: Element[] = []): Element {
  const node = {
    getAttribute: (name: string) => (name === "data-state" ? state : null),
    contains: (other: Element) => other === node || children.includes(other),
  };
  return node as unknown as Element;
}

test("a portaled surface is searched unless it is on its way out", () => {
  const open = surface("open");
  const closing = surface("closed");
  // A plain menu, which says nothing about its state because it is only rendered while open.
  const plain = surface(null);
  const scope = { contains: () => false } as unknown as Element;
  assert.deepEqual(
    withPortals([open, closing, plain], () => resolvePortalSurfaces(scope)),
    [open, plain],
  );
});

test("a surface inside the scope, or inside one already taken, is not indexed twice", () => {
  const inner = surface("open");
  const outer = surface("open", [inner]);
  const own = surface("open");
  const scope = {
    contains: (other: Element) => other === own,
  } as unknown as Element;
  assert.deepEqual(
    withPortals([own, outer, inner], () => resolvePortalSurfaces(scope)),
    [outer],
  );
});

test("the observer watches the document, since a portal lands outside the scope", async () => {
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // A popover renders to the body: an observer on the scope alone never hears one open or close.
  assert.match(engine, /scope\?\.ownerDocument\?\.body \?\? scope/);
  // And `data-state` is the only thing a dismissed one changes. It keeps its box until the
  // animation that follows finishes, so without this it stays in the count after it is gone.
  assert.match(engine, /attributeFilter: \[[^\]]*"data-state"/);
});

test("the rows progressive completion adds are re-anchored, not renumbered", async () => {
  // Streaming APPENDS, so keeping the ordinal is right there. Progressive completion PREPENDS the
  // older half of a thread, and match 3 of the tail is not match 3 of the whole conversation:
  // keeping the number moves the highlight to an older match off screen, and the next step from
  // there walks the reader backwards.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    engine,
    /completeProgressiveMounts\([\s\S]*?\.then\(\(\) => \{[\s\S]*?search\(false, reindex\(\)\);/,
  );
  // Through `reindex`, which answers false when the completion brought nothing in, so a settled
  // thread's 400ms probe does not take back an Enter pressed while it ran.
  assert.equal(engine.includes("rebuild("), false);
});

test("Escape closes the bar from the walk buttons, not just the field", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // On the WINDOW, in the capture phase, for the lifetime of the open bar. A handler on the bar
  // only reaches presses that started inside it, so clicking a message to read it left a bar
  // Escape would not close -- and with a tool request waiting, that same unprevented Escape went
  // on to `declineToolRequest`, which is bare Escape and is not shielded by `isTextEntryFocused`
  // on a message body. Closing a find bar must not be able to answer a tool request.
  const effect = bar.slice(bar.indexOf("const onEscape ="));
  const body = effect.slice(0, effect.indexOf("window.addEventListener"));
  assert.match(body, /event\.key !== "Escape"/);
  assert.match(body, /event\.preventDefault\(\);/);
  assert.match(body, /event\.stopPropagation\(\);/);
  assert.match(body, /close\(\);/);
  // Capture, so it runs ahead of the registry's own keydown listener rather than racing it.
  assert.match(effect, /window\.addEventListener\("keydown", onEscape, true\)/);
  assert.match(
    effect,
    /window\.removeEventListener\("keydown", onEscape, true\)/,
  );
  // Two presses it deliberately does not take: a modal above the bar owns Escape, and an open
  // popover, menu or listbox is dismissed by its own Escape first.
  assert.match(body, /isSurfaceBackgrounded\(/);
  assert.match(body, /resolvePortalSurfaces\(/);
  // And nothing left on the landmark, which would take the inside presses before the window does.
  const landmark = bar.slice(bar.indexOf('role="search"'));
  assert.equal(
    landmark.slice(0, landmark.indexOf(">")).includes("onKeyDown"),
    false,
  );
});

test("only threads this search can read are forced to finish mounting", async () => {
  // The shell keeps every workspace mounted and marks the off-route ones `inert`. Completing
  // globally would make a retained conversation mount every row it withheld, on a route where the
  // walk then skips all of it.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    engine,
    /completeProgressiveMounts\(\(viewport\) =>\s*\n?\s*indexReaches\(scope, viewport\)/,
  );
  const progressive = await readFile(
    new URL(
      "../src/components/assistant-ui/progressive-messages.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // The loop's exit has to read the filtered set too, or a completer this caller declined holds it
  // open forever.
  assert.match(
    progressive,
    /if \(wanted\(\)\.length === 0 && \(observed \|\| Date\.now\(\) >= deadline\)\)/,
  );
});

test("the chord is left to the browser when the scope is behind a modal", async () => {
  // `useShortcut` prevents the event BEFORE calling the handler, so declining from inside the
  // handler kills the chord for everyone: no bar, and no native find either.
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(bar, /claims: \(\) => !isSurfaceBackgrounded\(/);
  const shortcut = await readFile(
    new URL("../src/features/settings/hooks/use-shortcut.ts", import.meta.url),
    "utf8",
  );
  const consume = shortcut.indexOf("event.preventDefault();");
  assert.ok(consume > 0);
  assert.ok(shortcut.lastIndexOf("latestRef.current.claims?.()", consume) > 0);
});

test("the Enter that commits an IME candidate is left alone", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Ahead of preventDefault, or the composition is discarded before the guard is reached.
  const enter = bar.slice(bar.indexOf('event.key === "Enter"'));
  const guard = enter.indexOf("isImeComposing(event.nativeEvent)");
  const prevent = enter.indexOf("event.preventDefault()");
  assert.ok(guard > 0 && guard < prevent);
});

test("closing the bar hands focus back to where it came from", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Captured above the effect that focuses the field, or it reads the field it is about to fill.
  const capture = bar.indexOf("const active = document.activeElement");
  const takeFocus = bar.indexOf("input.select();");
  assert.ok(capture > 0 && capture < takeFocus);
  assert.match(bar, /origin\.focus\(\);/);
  // First answer only, and never the bar. StrictMode replays the effect in development, and by the
  // second run the field has focus: read plainly, the bar would aim focus at its own input.
  assert.match(bar, /originRef\.current === null &&/);
  // Against the bar's own element, not `data-find-skip`. The composer carries that attribute, and
  // it is the single most likely place the chord is pressed from: matching on it would refuse to
  // record exactly the origin this exists to restore.
  assert.match(bar, /barRef\.current\?\.contains\(active\) !== true/);
  assert.equal(bar.includes("closest(`[${FIND_SKIP_ATTRIBUTE}]`)"), false);
  // And only when closing dropped focus: the reader having moved it is not an invitation.
  assert.match(
    bar,
    /if \(focused !== null && focused !== document\.body\) return;/,
  );
});

test("the chat composer is out of the searchable scope", async () => {
  // Its draft lives in a textarea the index cannot read, so all it leaves find is the pill labels:
  // a search for "code" or "images" would land on the toolbar rather than the conversation.
  const thread = await readFile(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  const root = thread.slice(thread.indexOf("<ComposerPrimitive.Root"));
  assert.match(
    root.slice(0, root.indexOf(">")),
    /\{\.\.\.\{ \[FIND_SKIP_ATTRIBUTE\]: "" \}\}/,
  );
});

test("the reader is kept on the occurrence, not on the number", async () => {
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // Written where the active match is settled, read where the next list is built.
  assert.match(
    engine,
    /activeStartRef\.current = active >= 0 \? matches\[active\]\.start : null;/,
  );
  // Read BEFORE the new list is installed, or it is the new list's answer being read back.
  const read = engine.indexOf("const wasAt = activeStartRef.current;");
  const install = engine.indexOf("matchesRef.current = matches;", read - 400);
  assert.ok(read > 0 && read < install);
  assert.match(engine, /ordinalOfStart\(matches, wasAt\)/);
  // And when the occurrence is gone, the reader's position decides rather than a stale number.
  assert.match(
    engine,
    /at === -1 \? firstMatchFromViewport\(index, matches\) : at/,
  );
});

test("the ordinal survives an append and nothing else", async () => {
  // One rule decides who keeps their place. A streaming reply only adds at the tail, so match 20 is
  // still match 20. History arriving above, a workspace switch flipping `inert`, a breakpoint
  // revealing a column: each renumbers the list, and the reader's number then points at unrelated
  // text. An unchanged document is an append of nothing, which is why a rebuild that finds no news
  // leaves an Enter pressed while it ran alone.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    engine,
    /const before = indexRef\.current\.text;[\s\S]*?return !indexRef\.current\.text\.startsWith\(before\);/,
  );
  // Every rebuild goes through it: no call site decides for itself that the numbering held.
  assert.equal(engine.includes("search(false, false)"), false);
  assert.equal((engine.match(/search\(false, reindex\(\)\)/g) ?? []).length, 2);
  // Except a fresh open, which starts from the reader whatever the index says.
  assert.match(
    engine,
    /reindex\(\);\n\s*\/\/[^\n]*\n\s*search\(false, true\);/,
  );
});

test("a breakpoint that changes what is rendered invalidates the index", async () => {
  // Crossing one hides or reveals whole columns (`hidden lg:flex`) with nothing in the DOM to
  // observe, and the index reads computed visibility, so the observer alone cannot see it.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(engine, /window\.addEventListener\("resize", invalidate\);/);
  assert.match(engine, /window\.removeEventListener\("resize", invalidate\);/);
  // Through the same throttle as a mutation, so dragging a window edge costs one rebuild an
  // interval rather than one a frame.
  assert.match(
    engine,
    /const invalidate = \(\) => \{[\s\S]*?REINDEX_INTERVAL_MS\);/,
  );
});

test("leaving the shell forgets the search", () => {
  // The store is module-global and keeps the query across a close on purpose. Signing out unmounts
  // the shell, and the next person to sign in in the same tab must not be handed the last one's
  // search, open and focused.
  const store = useFindInPageStore;
  store.getState().setQuery("someone else's search");
  store.getState().requestFocus();
  assert.equal(store.getState().open, true);
  store.getState().reset();
  assert.deepEqual(
    { open: store.getState().open, query: store.getState().query },
    { open: false, query: "" },
  );
});

test("the shell unmounting is what calls it, not the bar closing", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // As an unmount cleanup. Not `enabled`: a dialog turns that off, and the search should still be
  // there when it closes.
  assert.match(bar, /useEffect\(\(\) => reset, \[reset\]\);/);
  // And `close` still keeps the query, which is what makes reopening offer the last search.
  const store = await readFile(
    new URL(
      "../src/features/find-in-page/stores/find-in-page-store.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(store, /close: \(\) => set\(\{ open: false \}\),/);
});

test("the capped window follows the reader, not the top of the document", () => {
  // A single common letter in a long thread has far more matches than the cap. Keeping the first
  // `limit` of them keeps only the top of the document, so a reader at the bottom is walked away
  // from every occurrence beside them, to one they were not looking for.
  const body = `${"q".repeat(20_000)} needle ${"q".repeat(20_000)}`;
  const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const anchor = index.text.indexOf("needle");
  const limit = 100;

  const fromTheTop = findMatches(index, "q", limit);
  assert.equal(fromTheTop.length, limit);
  assert.equal(fromTheTop[fromTheTop.length - 1].end <= anchor, true);

  const aroundTheReader = findMatches(index, "q", limit, anchor);
  assert.equal(aroundTheReader.length, limit);
  // Half either side, so the walk goes forward from where the reader is and back the other way.
  assert.equal(
    aroundTheReader.some((match) => match.start < anchor),
    true,
  );
  assert.equal(
    aroundTheReader.some((match) => match.start > anchor),
    true,
  );
  const nearest = aroundTheReader.find((match) => match.start > anchor);
  assert.ok(nearest && nearest.start - anchor < 20);
});

test("a capped window slides as matches are appended", () => {
  // Which is why an ordinal cannot be kept across a rebuild even when the document only grew. The
  // window recentres, so the same number is a different occurrence.
  // The reader near the end, which is where a reply streams in: too few matches after them to fill
  // half the window, so the window is pinned to the end of the list and the end is what moves.
  const limit = 100;
  const before = `${"q".repeat(200)} needle ${"q".repeat(20)}`;
  const after = `${before}${"q".repeat(200)}`;
  const anchor = before.indexOf("needle");
  const first = findMatches(
    buildTextIndex(el("DIV", [el("P", [text(before)])])),
    "q",
    limit,
    anchor,
  );
  const second = findMatches(
    buildTextIndex(el("DIV", [el("P", [text(after)])])),
    "q",
    limit,
    anchor,
  );
  assert.equal(first.length, limit);
  assert.equal(second.length, limit);
  // Same ordinal, different occurrence: the number moved under the reader.
  assert.notEqual(first[limit - 1].start, second[limit - 1].start);
  // The occurrence itself is still in the list, at a different index. That is what the engine
  // follows instead of the number.
  assert.equal(
    second.some((match) => match.start === first[limit - 1].start),
    true,
  );
});

test("the window is only computed when the cap bites", () => {
  // Under the cap this is the single pass it always was, whatever the anchor says.
  const index = buildTextIndex(el("DIV", [el("P", [text("q q q q q")])]));
  assert.deepEqual(
    findMatches(index, "q", 100, 8),
    findMatches(index, "q", 100, 0),
  );
});

test("stopping the count early does not move the window", () => {
  // `total` is only counted to keep the window off the end of the list, and it stops as soon as it
  // is high enough to no longer do that. A stop that came too soon would drag the window back
  // toward the top, which is the whole thing the anchor exists to prevent.
  const body = "q".repeat(500);
  const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const at = 200;
  const window = findMatches(index, "q", 50, at);
  assert.equal(window.length, 50);
  // Centred on the reader: 25 kept behind them, the rest ahead.
  assert.equal(window[0].start, at - 25);
  assert.equal(window[window.length - 1].end, at + 25);
  // And the matches past the window, which the count stops before reaching, are really there.
  assert.equal(findMatches(index, "q", 500, 0).length, 500);
});

test("the window stops at the ends of the list", () => {
  const body = `needle ${"q".repeat(500)}`;
  const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  // Anchored at the very start: nothing to keep before it, so the window is the first `limit`.
  const atTheTop = findMatches(index, "q", 50, 1);
  assert.equal(atTheTop[0].start, index.text.indexOf("q"));
  // Anchored past the end: the window is the last `limit`, not a slice running off it.
  const atTheEnd = findMatches(index, "q", 50, index.text.length);
  assert.equal(atTheEnd.length, 50);
  assert.equal(atTheEnd[atTheEnd.length - 1].end, index.text.length);
});

test("clipped accessibility text is not searchable", () => {
  // Tailwind's `sr-only` keeps a real box at full opacity and clips it to nothing, so
  // `checkVisibility` calls it visible. The app has 46 of them; counted, they are matches with a
  // highlight clipped away along with the text.
  const label = el("SPAN", [text("Data input")]);
  const shown = el("SPAN", [text("Data output")]);
  withStyles(
    new Map([
      [label, { clipPath: "inset(50%)" }],
      [shown, { clipPath: "none" }],
    ]),
    () => {
      const index = buildTextIndex(el("DIV", [label, shown]));
      assert.equal(index.text.includes("data input"), false);
      assert.equal(index.text.includes("data output"), true);
    },
  );
  // The legacy spelling of the same idiom.
  const legacy = el("SPAN", [text("Data input")]);
  withStyles(new Map([[legacy, { clip: "rect(0px, 0px, 0px, 0px)" }]]), () => {
    assert.equal(buildTextIndex(el("DIV", [legacy])).text, "");
  });
});

test("the counter says '+' only when the cap actually cut something off", () => {
  // `findMatches` stops at the limit it is given, so a count equal to the cap cannot say whether it
  // is the total or a floor: a page holding exactly MAX_MATCHES read as "more than MAX_MATCHES".
  // Asking for one past it is the whole difference; the extra match is thrown away.
  const occurrences = (n: number) =>
    findMatches(
      buildTextIndex(el("DIV", [el("P", [text("a".repeat(n))])])),
      "a",
      MAX_MATCHES + 1,
    ).length;
  assert.equal(occurrences(MAX_MATCHES - 1) > MAX_MATCHES, false);
  assert.equal(occurrences(MAX_MATCHES) > MAX_MATCHES, false);
  assert.equal(occurrences(MAX_MATCHES + 1) > MAX_MATCHES, true);
  // The count itself cannot tell the last two apart, which is why the flag exists.
  assert.equal(
    findMatches(
      buildTextIndex(el("DIV", [el("P", [text("a".repeat(MAX_MATCHES))])])),
      "a",
    ).length,
    findMatches(
      buildTextIndex(el("DIV", [el("P", [text("a".repeat(MAX_MATCHES + 1))])])),
      "a",
    ).length,
  );
});

test("the cap flag is what the bar renders, not the count", async () => {
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    engine,
    /findMatches\(\s*\n\s*index,\s*\n\s*queryRef\.current,\s*\n\s*MAX_MATCHES \+ 1,/,
  );
  assert.match(engine, /cappedRef\.current = matches\.length > MAX_MATCHES;/);
  // Trimmed back to the cap, so nothing downstream sees the probe match -- and trimmed from the
  // end the reader is further from, since a window anchored near the bottom of a long thread ends
  // at the document's last match rather than starting at its first.
  assert.match(
    engine,
    /if \(cappedRef\.current\) dropProbeFurthestFrom\(matches, anchoredAt\);/,
  );
  // The anchor is captured as the thunk resolves it, so the trim costs no second layout read and
  // still costs nothing under the cap, where the thunk is never called.
  assert.match(engine, /let anchoredAt: number \| null = null;/);
  assert.match(engine, /anchoredAt = viewportOffset\(index\);/);
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(bar, /\$\{capped \? "\+" : ""\}/);
  assert.equal(bar.includes("count >= MAX_MATCHES"), false);
});

test("Escape is left to the IME while it is composing", async () => {
  // Escape dismisses a candidate. Consumed here, it closes the bar out from under a word still
  // being typed, and the candidate window never sees the key it was aimed at.
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const escape = bar.slice(bar.indexOf("const onEscape ="));
  const guard = escape.indexOf("isImeComposing(event)");
  const consume = escape.indexOf("event.preventDefault()");
  assert.ok(guard > 0 && guard < consume);
  // Safe to let through: the global listener stands aside for a composing event before it looks
  // for a binding, so the bare-Escape decline cannot take it either.
  const shortcut = await readFile(
    new URL("../src/features/settings/hooks/use-shortcut.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    shortcut,
    /if \(isImeComposing\(event\)\) return;\n\s*const hit = bindings\.find/,
  );
});

test("the selection fallback only clears what it put there", () => {
  // The engines without a highlight registry get a selection instead, and the bar paints once on
  // opening, before any query is typed. Clearing unconditionally there throws away whatever the
  // reader had highlighted to copy, and closing cannot give it back.
  // Boundary points, since engines differ on whether `getRangeAt` hands back the object that was
  // added. Two distinct start containers stand in for two distinct selections.
  const span = (name: string) =>
    ({
      startContainer: name,
      startOffset: 0,
      endContainer: name,
      endOffset: 1,
    }) as unknown as Range;
  const ranges: Range[] = [];
  const selection = {
    get rangeCount(): number {
      return ranges.length;
    },
    getRangeAt: (i: number): Range => ranges[i],
    removeAllRanges: (): void => {
      ranges.length = 0;
    },
    addRange: (range: Range): void => {
      ranges.push(range);
    },
  };
  const view = globalThis as { window?: unknown };
  const saved = view.window;
  view.window = { getSelection: () => selection };
  try {
    ranges.push(span("what the reader selected"));
    selectRangeFallback(null);
    assert.deepEqual(ranges, [span("what the reader selected")]);

    selectRangeFallback(span("the active match"));
    assert.deepEqual(ranges, [span("the active match")]);
    selectRangeFallback(null);
    // Annotated, or `deepEqual`'s assertion signature narrows `ranges` to `never[]` from here on.
    assert.deepEqual(ranges, [] as Range[]);

    // And a selection the reader made while the bar was open, over the match this had put there.
    selectRangeFallback(span("the active match"));
    ranges.length = 0;
    ranges.push(span("dragged over something else"));
    selectRangeFallback(null);
    assert.deepEqual(ranges, [span("dragged over something else")]);
  } finally {
    view.window = saved;
  }
});

test("the generated-image actions are out of the index too", async () => {
  // Same shape as the badge below, and the same reason. From `sm` up the action bar over a
  // generated image is transparent until the card is hovered, so its "Edit" was counted and walked
  // to under a highlight nobody can see. Every place that mounts persistently transparent text has
  // to say so, since the index cannot tell one from a message still fading in.
  const tool = await readFile(
    new URL(
      "../src/components/assistant-ui/tool-ui-image-generation.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const at = tool.indexOf("sm:group-hover/generated-image:opacity-100");
  assert.notEqual(at, -1);
  assert.match(
    tool.slice(Math.max(at - 700, 0), at),
    /\{\.\.\.\{ \[FIND_SKIP_ATTRIBUTE\]: "" \}\}/,
  );
});

test("a hover-only badge is out of the index", async () => {
  // The response model label is mounted transparent and unclickable until the message is hovered.
  // That is an affordance, not an entrance animation, so a match in it would be counted and walked
  // to under a highlight nobody can see. Marked at the call site rather than by turning the opacity
  // check on, which would drop a message still fading in.
  const sheet = await readFile(
    new URL(
      "../src/components/assistant-ui/message-response-details-sheet.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const badge = sheet.slice(sheet.indexOf("aui-response-model-badge") - 400);
  assert.match(
    badge.slice(0, badge.indexOf("aui-response-model-badge")),
    /\{\.\.\.\{ \[FIND_SKIP_ATTRIBUTE\]: "" \}\}/,
  );
  // The general rule stays off, so a fade-in is still findable.
  const index = await readFile(
    new URL(
      "../src/features/find-in-page/lib/find-text-index.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(index, /opacityProperty: false/);
});

test("a container query resizing the scope invalidates the index", async () => {
  // A container query does not need the window to change: Images is an `@container` with labels on
  // `@[50rem]`, so pinning or collapsing the sidebar crosses that breakpoint with no window resize
  // and no mutation inside the scope.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(engine, /new ResizeObserver\(\(\) => \{/);
  assert.match(engine, /sized\.observe\(scope\);/);
  assert.match(engine, /sized\?\.disconnect\(\);/);
  // The first delivery reports the size the scope already had, which is not news and would cost a
  // flatten on every open.
  assert.match(
    engine,
    /if \(!measured\) \{\s*\n\s*measured = true;\s*\n\s*return;/,
  );
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
