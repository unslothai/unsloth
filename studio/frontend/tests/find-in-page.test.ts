// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Find in page. What has to hold is that the flat index and the document agree: every offset
// the search reports lands on the character a reader sees. There is no DOM library here, so
// the runner is `node --test` over a hand-rolled document.

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  FIND_HIGHLIGHT,
  FIND_HIGHLIGHT_ACTIVE,
  MAX_PAINTED_RANGES,
  clearHighlights,
  mutatesSearchableText,
  paintHighlights,
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
  renumbersMatches,
  segmentAt,
  startPositionAt,
} from "../src/features/find-in-page/lib/find-text-index.ts";

/** The feature's component module as one string. */
async function readComponentSource(): Promise<string> {
  return await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
}

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

test("inline markup does not break a word", () => {
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
    ["DIV", { [FIND_SKIP_ATTRIBUTE]: "" }],
    ["DIV", { inert: "" }],
    ["DIV", { hidden: "" }],
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

test("KaTeX indexes its painted tree instead of its clipped accessibility mirror", () => {
  const mathmlText = text("x2");
  const paintedText = text("x2");
  const root = el(
    "SPAN",
    [
      el("SPAN", [mathmlText], { class: "katex-mathml" }),
      el("SPAN", [paintedText], {
        class: "katex-html",
        "aria-hidden": "true",
      }),
    ],
    { class: "katex" },
  );

  const index = buildTextIndex(root);
  const [match] = findMatches(index, "x2");
  assert.ok(match);
  assert.equal(
    index.segments.some((segment) => segment.node === mathmlText),
    false,
    "the clipped MathML mirror must not create ranges",
  );
  assert.equal(
    index.segments.some((segment) => segment.node === paintedText),
    true,
    "the painted KaTeX tree must create the range",
  );
  assert.equal(startPositionAt(index.segments, match.start)?.node, paintedText);
  assert.equal(endPositionAt(index.segments, match.end)?.node, paintedText);
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
  // The end boundary is the node's length: the one offset no segment holds.
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
  // The Turkic fold is a bare `i`, which fits in one unit.
  assert.equal("İ".toLowerCase().length, 2, "premise: the default fold grows");
  assert.equal(foldText("İ"), "i");

  const spellings = buildTextIndex(
    el("DIV", [el("P", [text("Welcome to İstanbul")])]),
  );
  for (const query of ["istanbul", "İstanbul", "ISTANBUL", "Istanbul"]) {
    assert.equal(findMatches(spellings, query).length, 1, query);
  }

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
  // The fold of a run is not the folds of its code points strung together.
  const dottedI = String.fromCharCode(0x0130); // Turkish dotted I, whose fold is two units
  const greek = `\u039f\u0394\u039f\u03a3 ${dottedI} \u039f\u03a3`; // "ΟΔΟΣ İ ΟΣ"
  const index = buildTextIndex(el("DIV", [el("P", [text(greek)])]));
  // The length has to hold, or every offset after it maps to the wrong character.
  assert.equal(index.text.length, greek.length);
  assert.equal(index.segments[0].length, greek.length);
  assert.equal(findMatches(index, "\u039f\u03a3").length, 2);
  assert.equal(findMatches(index, "\u039f\u0394\u039f\u03a3").length, 1);
  assert.equal(index.text, "\u03bf\u03b4\u03bf\u03c3 i \u03bf\u03c3");
});

test("casing context carries across inline markup", () => {
  const index = buildTextIndex(
    el("DIV", [el("P", [text("\u039f"), el("EM", [text("\u03a3")])])]),
  );
  assert.equal(index.text, "\u03bf\u03c3");
  assert.equal(findMatches(index, "\u039f\u03a3").length, 1);
  assert.equal(index.segments.length, 2);
  assert.equal(index.segments[1].start, 1);
});

test("several dotted I in a run fold without drift", () => {
  const dottedI = String.fromCharCode(0x0130);
  const raw = `${dottedI}a${dottedI}\u039f\u03a3${dottedI}b`;
  const index = buildTextIndex(el("DIV", [el("P", [text(raw)])]));
  assert.equal(index.text.length, raw.length);
  assert.equal(index.text, "iai\u03bf\u03c3ib");
});

test("either sigma finds the other, whichever one is on screen", () => {
  // `toLowerCase` picks the final form by position, so half the spellings folded the other way.
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
  const run = "\u03a3\u03a3\u03a3";
  const sigmas = buildTextIndex(el("DIV", [el("P", [text(run)])]));
  assert.equal(sigmas.text, "\u03c3\u03c3\u03c3");
  assert.equal(sigmas.text.length, run.length);
});

test("a non-breaking space answers to the space key", () => {
  // Spelled with a char code: a literal U+00A0 in source looks like a space to every reader.
  const nbsp = String.fromCharCode(0x00a0);
  const index = buildTextIndex(
    el("DIV", [el("P", [text(`Unsloth${nbsp}Studio`)])]),
  );
  assert.equal(findMatches(index, "unsloth studio").length, 1);
  assert.equal(index.text.length, "Unsloth Studio".length);
  assert.equal(index.text.includes(nbsp), false);
});

test("a dotted I does not make the rest of its run case-sensitive", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("HELLO İ")])]));
  assert.equal(findMatches(index, "hello").length, 1);
  const [match] = findMatches(index, "hello");
  assert.equal(startPositionAt(index.segments, match.start)?.offset, 0);
});

test("a text node bigger than its share contributes its prefix", () => {
  const node = text(`unsloth ${"x".repeat(MAX_NODE_CHARS + 10)}`);
  const index = buildTextIndex(el("DIV", [el("P", [node])]));
  assert.equal(index.truncated, true);
  assert.equal(index.text.length, MAX_NODE_CHARS);
  assert.equal(findMatches(index, "unsloth").length, 1);
  assert.equal(startPositionAt(index.segments, 0)?.node, node);
});

test("an oversized node does not take the whole budget with it", () => {
  const log = el("PRE", [text("x".repeat(MAX_INDEX_CHARS + 1000))]);
  const onScreen = el("P", [
    text("the message in front of the reader says unsloth"),
  ]);
  const index = buildTextIndex(el("DIV", [log, onScreen]));
  assert.equal(index.truncated, true);
  assert.equal(findMatches(index, "in front of the reader").length, 1);
  assert.equal(index.text.startsWith("x".repeat(MAX_NODE_CHARS)), true);
});

test("a popover over a document at the ceiling is still searchable", () => {
  // A workspace filling the budget used to end the walk, leaving the portal unindexed.
  const filler = Array.from({ length: 50 }, () =>
    el("P", [text("x".repeat(MAX_NODE_CHARS))]),
  );
  const popover = el("DIV", [el("P", [text("a model named unsloth zephyr")])]);
  const index = buildTextIndex(el("DIV", filler), [popover]);
  assert.equal(index.truncated, true);
  assert.equal(findMatches(index, "unsloth zephyr").length, 1);
});

test("the reserve is only held back when there is a portal to hold it for", () => {
  const filler = Array.from({ length: 50 }, () =>
    el("P", [text("x".repeat(MAX_NODE_CHARS))]),
  );
  const alone = buildTextIndex(el("DIV", filler));
  assert.equal(alone.text.length, MAX_INDEX_CHARS);
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
  const hidden = {
    ...el("DIV", [text("buried")]),
    checkVisibility: () => false,
  };
  const shown = { ...el("DIV", [text("shown")]), checkVisibility: () => true };
  const index = buildTextIndex(el("DIV", [hidden, shown]));
  assert.equal(index.text.includes("buried"), false);
  assert.equal(index.text.includes("shown"), true);
});

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
  // A paste could carry one, and it would match the very boundary the separator keeps closed.
  const index = buildTextIndex(
    el("DIV", [el("P", [text("a")]), el("P", [text("b")])]),
  );
  assert.equal(normalizeQuery(`a${BLOCK_SEPARATOR}b`), null);
  assert.deepEqual(findMatches(index, `a${BLOCK_SEPARATOR}b`), []);
});

test("content-visibility skipping is not treated as invisibility", () => {
  // Such a subtree is SKIPPED, not hidden, and nothing would put it back.
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
  // An engine reads only the name it knows, and Web IDL drops an unknown member silently.
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
    assert.equal(
      options[modern],
      options[historic],
      `${modern} != ${historic}`,
    );
  }
});

test("an engine that honours only the historic option names still hides hidden text", () => {
  // Chrome 105-120 / Firefox 106-121: `checkVisibility` exists, but ignores the modern names.
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
  // SVG and MathML keep their source casing, so `<svg>` answers "svg".
  const svg = el("svg", [el("text", [text("mermaid label")])]);
  const index = buildTextIndex(el("DIV", [svg, el("P", [text("prose")])]));
  assert.equal(index.text.includes("mermaid"), false);
  assert.equal(index.text.includes("prose"), true);
});

test("a portaled surface is indexed after the scope, behind a boundary", () => {
  const scope = el("DIV", [el("P", [text("in the thread")])]);
  const portal = el("DIV", [el("P", [text("in the popover")])]);
  const index = buildTextIndex(scope, [portal]);
  assert.equal(index.text, `in the thread${BLOCK_SEPARATOR}in the popover`);
  assert.equal(findMatches(index, "in the popover").length, 1);
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
  // The caret goes with the selection on WebKit and Blink: the field still reports as active
  // while every keystroke is swallowed.
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
  // The caret has to be taken BEFORE the selection moves, or there is nothing left to restore.
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
  // The shell keeps off-route workspaces mounted under `hidden` and `inert` INSIDE the scope.
  for (const mark of ["[inert]", "[hidden]", '[aria-hidden="true"]']) {
    const parked = skipNode({ mark });
    const streamed = skipNode({ parent: parked });
    assert.equal(
      mutatesSearchableText(record(streamed, "characterData")),
      false,
      `a reply streaming under ${mark} still asked for a rebuild`,
    );
  }
  const live = skipNode({ parent: skipNode({}) });
  assert.equal(mutatesSearchableText(record(live, "characterData")), true);
});

test("parking a workspace is itself a change, whichever attribute says so", () => {
  // `closest` matches where it starts, so the record saying a workspace went `inert` answers
  // "inside skipped content".
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
  // Same trap: gaining the attribute answers "inside skipped content" and is thrown away.
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
  const orphan = skipNode({ skipped: true, parent: null });
  assert.equal(
    mutatesSearchableText(record(orphan, "attributes", FIND_SKIP_ATTRIBUTE)),
    true,
  );
});

test("an attribute that is not the skip flag is judged from the target itself", () => {
  const bar = skipNode({ skipped: true });
  const inside = skipNode({ parent: bar });
  assert.equal(
    mutatesSearchableText(record(inside, "attributes", "inert")),
    false,
  );
});

test("a display:contents wrapper that is itself invisible keeps its own text out", () => {
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
  // `viewportOffset` reads layout, and an argument is evaluated whether or not it is wanted.
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
  assert.deepEqual(capped, findMatches(index, "a", 3, 6));
});

test("a decomposed dotted I is found by the ordinary query", () => {
  // U+0130 decomposes to `I` + a combining dot, which folds to `i` + that dot.
  const decomposed = "\u0049\u0307stanbul";
  assert.equal("\u0069\u0307".normalize("NFC"), "\u0069\u0307");
  const index = buildTextIndex(el("P", [text(`Welcome to ${decomposed}`)]));
  for (const query of ["istanbul", "ISTANBUL", "\u0130stanbul"]) {
    assert.equal(findMatches(index, query).length, 1, query);
  }
});

test("the dotted variant costs nothing on a document without combining marks", () => {
  const index = buildTextIndex(el("P", [text("indexing is fine here")]));
  assert.equal(findMatches(index, "indexing").length, 1);
  assert.equal(findMatches(index, "i").length, 4);
});

test("a query too large to compile falls back instead of throwing", () => {
  // Measured on V8: a whitespace-bearing query throws at 15,651 characters.
  const index = buildTextIndex(el("P", [text("a small thread about unsloth")]));
  const huge = "some log line with spaces ".repeat(4000);
  assert.ok(huge.length > 15_651, "premise: past the measured V8 ceiling");
  assert.doesNotThrow(() => findMatches(index, huge));
  assert.deepEqual(findMatches(index, huge), []);
});

test("a needle longer than the haystack is rejected before any of the work", () => {
  const index = buildTextIndex(el("P", [text("short")]));
  assert.deepEqual(findMatches(index, "x".repeat(500)), []);
  // A decomposed query is longer than the precomposed text it is meant to find.
  const cafe = buildTextIndex(el("P", [text("caf\u00e9")]));
  assert.equal(findMatches(cafe, "cafe\u0301").length, 1);
  assert.equal(findMatches(index, "short").length, 1);
});

test("a numeric anchor still means what it always did", () => {
  const index = buildTextIndex(el("P", [text("b b b b b b b b")]));
  assert.deepEqual(findMatches(index, "b", 3, 0), findMatches(index, "b", 3));
  assert.deepEqual(
    findMatches(index, "b", 3, 10),
    findMatches(index, "b", 3, () => 10),
  );
});

test("a word matches whichever way either side spells it", () => {
  // macOS hands back decomposed filenames while a model writes composed prose.
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
      assert.deepEqual(matches[0], { start: 2, end: 2 + written.length });
      assert.equal(index.text.slice(matches[0].start, matches[0].end), written);
    }
  }
});

test("an occurrence that mixes the two spellings is still one word", () => {
  // Alternating whole spellings reaches only all-composed or all-decomposed text, and joining
  // two text nodes joins two sources. Every engine's own find reaches this.
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
    assert.deepEqual(matches[0], { start: 2, end: 2 + mixed.length });
    assert.equal(index.text.slice(matches[0].start, matches[0].end), mixed);
  }

  const split = buildTextIndex(
    el("DIV", [el("P", [text(`caf${composed}`), text(`caf${decomposed}`)])]),
  );
  assert.equal(findMatches(split, `caf${composed}caf${composed}`).length, 1);
});

test("Hangul canonical spellings match without changing document offsets", () => {
  const composed = "가각";
  const decomposed = composed.normalize("NFD");
  const mixed = `가${"각".normalize("NFD")}`;

  for (const written of [composed, decomposed, mixed]) {
    for (const typed of [composed, decomposed, mixed]) {
      const node = text(`a ${written} b`);
      const index = buildTextIndex(el("P", [node]));
      const matches = findMatches(index, typed);
      assert.deepEqual(
        matches,
        [{ start: 2, end: 2 + written.length }],
        `text ${escape(written)} and query ${escape(typed)} did not meet`,
      );
      assert.equal(
        startPositionAt(index.segments, matches[0].start)?.node,
        node,
      );
      assert.equal(
        startPositionAt(index.segments, matches[0].start)?.offset,
        2,
      );
      assert.equal(
        endPositionAt(index.segments, matches[0].end)?.offset,
        2 + written.length,
      );
    }
  }

  const leading = text("ᄀ");
  const rest = text("ᅡ각");
  const split = buildTextIndex(el("P", [leading, el("EM", [rest])]));
  const [match] = findMatches(split, composed);
  assert.ok(
    match,
    "a decomposed syllable split across inline nodes must match",
  );
  assert.equal(startPositionAt(split.segments, match.start)?.node, leading);
  assert.equal(endPositionAt(split.segments, match.end)?.node, rest);
  assert.equal(
    endPositionAt(split.segments, match.end)?.offset,
    rest.data.length,
  );
});

test("an open Hangul syllable cannot match the prefix of a closed one", () => {
  const open = "가";
  const closed = "각";
  for (const written of [closed, closed.normalize("NFD"), `${open}\u11a8`]) {
    const index = buildTextIndex(el("P", [text(written)]));
    assert.deepEqual(
      findMatches(index, open),
      [],
      `open ${escape(open)} matched part of closed ${escape(written)}`,
    );
  }

  for (const written of [open, open.normalize("NFD")]) {
    const index = buildTextIndex(el("P", [text(written)]));
    assert.deepEqual(findMatches(index, open), [
      { start: 0, end: written.length },
    ]);
  }
});

test("the jamo boundary holds for Hangul that has only one spelling", () => {
  // Extended and Old Hangul jamo have no precomposed form, so NFC and NFD spell them the same way.
  // One spelling used to take the literal scan, which writes no boundary, and the open syllable
  // painted the first two jamo of a visibly closed one. Modern Hangul always has two spellings and
  // reached the pattern anyway, so this only ever showed up out here.
  const open = "ꥠힰ";
  const closed = "ꥠힰퟋ";
  assert.equal(open.normalize("NFC"), open);
  assert.equal(open.normalize("NFD"), open);

  const inClosed = buildTextIndex(el("P", [text(closed)]));
  assert.deepEqual(findMatches(inClosed, open), []);
  assert.deepEqual(findMatches(inClosed, closed), [
    { start: 0, end: closed.length },
  ]);

  const inOpen = buildTextIndex(el("P", [text(open)]));
  assert.deepEqual(findMatches(inOpen, open), [{ start: 0, end: open.length }]);
  assert.deepEqual(findMatches(inOpen, closed), []);
});

test("complete Old Hangul clusters cannot match inside a longer syllable", () => {
  const leadAndVowels = "ꥠힰힱ";
  const withTrailing = `${leadAndVowels}ퟋ`;
  const withTwoTrailing = `${withTrailing}ퟌ`;

  assert.deepEqual(
    findMatches(buildTextIndex(el("P", [text(withTrailing)])), leadAndVowels),
    [],
  );
  assert.deepEqual(
    findMatches(buildTextIndex(el("P", [text(withTwoTrailing)])), withTrailing),
    [],
  );
  assert.deepEqual(
    findMatches(buildTextIndex(el("P", [text(withTrailing)])), withTrailing),
    [{ start: 0, end: withTrailing.length }],
  );
});

test("a half-composed Hangul syllable is found and covered whole", () => {
  // Hangul composes in two steps, L+V into a syllable and then the trailing jamo into it, and text
  // can stop after the first. NFC and NFD both write past that spelling, so a document holding one
  // used to be invisible to every query, its own spelling included.
  const closed = "각";
  const spellings = [closed, closed.normalize("NFD"), `각`];
  for (const written of spellings) {
    const index = buildTextIndex(el("P", [text(written)]));
    for (const query of spellings) {
      assert.deepEqual(
        findMatches(index, query),
        [{ start: 0, end: written.length }],
        `${escape(query)} did not cover ${escape(written)}`,
      );
    }
    // The open syllable still stops at the trailing jamo, whichever way the closed one is written.
    assert.deepEqual(findMatches(index, "가"), []);
  }
});

test("the index itself is left in the form the document wrote", () => {
  // Normalizing the index would change its length, and every offset stands for one character.
  const decomposed = "cafe\u0301";
  const index = buildTextIndex(el("DIV", [el("P", [text(decomposed)])]));
  assert.equal(index.text, decomposed);
  assert.equal(index.text.length, decomposed.length);
});

test("spelling variants do not loosen whitespace inside a fence", () => {
  const fence = el("PRE", [text("caf\u00e9   au lait")]);
  withStyles(new Map([[fence, { whiteSpace: "pre" }]]), () => {
    const index = buildTextIndex(el("DIV", [fence]));
    assert.equal(findMatches(index, "caf\u00e9 au lait").length, 0);
    assert.equal(findMatches(index, "cafe\u0301   au lait").length, 1);
  });
});

test("an engine with no checkVisibility falls back to the computed properties", () => {
  // `checkVisibility` is undefined on WebKitGTK, and read as "not false" it indexed every
  // `display: none` subtree.
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
  const wrapper = el("DIV", [el("P", [text("inside a wrapper")])]);
  withStyles(new Map([[wrapper, { display: "contents" }]]), () => {
    assert.equal(buildTextIndex(el("DIV", [wrapper])).text, "inside a wrapper");
  });
});

test("two spans the CSS renders as blocks do not run together", () => {
  // No tag name says these are blocks, and run together they invent a word on screen nowhere.
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
      assert.equal(findMatches(index, "open").length, 1);
      assert.equal(findMatches(index, "ai models").length, 1);
    },
  );
  const inline = el("SPAN", [text("slo")]);
  withStyles(new Map([[inline, { display: "inline" }]]), () => {
    const index = buildTextIndex(el("P", [text("un"), inline, text("th")]));
    assert.equal(findMatches(index, "unsloth").length, 1);
  });
});

test("whitespace is only flexible where the page collapses it", () => {
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
      assert.equal(findMatches(wrapped, "unsloth fast").length, 1);
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
    assert.equal(index.text.includes("offscreen"), false);
  } finally {
    view.getComputedStyle = saved;
  }
});

test("a query spanning whitespace matches the phrase as it renders", () => {
  const index = buildTextIndex(
    el("DIV", [el("P", [text("A soft wrapped\n      phrase about unsloth.")])]),
  );
  assert.equal(findMatches(index, "wrapped phrase").length, 1);
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

test("a document past the ceiling is flattened as far as it goes and says so", () => {
  const chunk = "x".repeat(100_000);
  const paragraphs = Array.from({ length: 60 }, () => el("P", [text(chunk)]));
  const index = buildTextIndex(el("DIV", paragraphs));
  assert.equal(index.truncated, true);
  assert.ok(index.text.length <= MAX_INDEX_CHARS);
  assert.ok(index.segments.length > 0);
  assert.ok(findMatches(index, "xxx").length > 0);
});

test("a clipped node does not run into the next one", () => {
  // Without a boundary a match across the seam maps back to a Range spanning every discarded
  // character.
  const clipped = text(
    `${"x".repeat(MAX_NODE_CHARS)}${"discarded ".repeat(500)}`,
  );
  const next = text("yz");
  const index = buildTextIndex(el("DIV", [el("P", [clipped, next])]));
  assert.equal(index.truncated, true);
  assert.deepEqual(findMatches(index, "xy"), []);
  assert.equal(index.text.includes(BLOCK_SEPARATOR), true);
  assert.equal(findMatches(index, "yz").length, 1);
});

test("the ceiling holds across a block boundary", () => {
  // A node landing exactly on the ceiling let the next separator push `length` past it, and the
  // negative `room` made `slice(0, room)` take all but the last character.
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

test("registered ranges are cleared and repainted before deletion or replacement", () => {
  const events: string[] = [];
  class FakeHighlight {
    priority = 0;
    constructor(..._ranges: Range[]) {}
    clear(): void {
      events.push("clear");
    }
  }
  class FakeElement {
    private opacity = "";
    style = {
      get opacity() {
        return root.opacity;
      },
      set opacity(value: string) {
        events.push(`opacity:${value}`);
        root.opacity = value;
      },
    };
    get offsetHeight(): number {
      events.push("reflow");
      return 1;
    }
  }
  const root = new FakeElement();
  const entries = new Map<string, FakeHighlight>([
    [FIND_HIGHLIGHT, new FakeHighlight()],
    [FIND_HIGHLIGHT_ACTIVE, new FakeHighlight()],
  ]);
  const registry = {
    get: (name: string) => entries.get(name),
    set: (name: string, highlight: FakeHighlight) => {
      events.push(`set:${name}`);
      entries.set(name, highlight);
    },
    delete: (name: string) => {
      events.push(`delete:${name}`);
      return entries.delete(name);
    },
  };
  const scope = globalThis as {
    CSS?: unknown;
    Highlight?: unknown;
    HTMLElement?: unknown;
    document?: unknown;
  };
  const saved = {
    css: scope.CSS,
    document: scope.document,
    highlight: scope.Highlight,
    htmlElement: scope.HTMLElement,
    hadCss: "CSS" in scope,
    hadDocument: "document" in scope,
    hadHighlight: "Highlight" in scope,
    hadHtmlElement: "HTMLElement" in scope,
  };
  scope.CSS = { highlights: registry };
  scope.Highlight = FakeHighlight;
  scope.HTMLElement = FakeElement;
  scope.document = {
    querySelector: () => root,
    body: root,
    documentElement: root,
  };
  try {
    clearHighlights();
    assert.deepEqual(events, [
      "clear",
      `delete:${FIND_HIGHLIGHT_ACTIVE}`,
      "clear",
      `delete:${FIND_HIGHLIGHT}`,
      "opacity:0.999999",
      "reflow",
      "opacity:",
    ]);

    events.length = 0;
    entries.set(FIND_HIGHLIGHT, new FakeHighlight());
    entries.set(FIND_HIGHLIGHT_ACTIVE, new FakeHighlight());
    paintHighlights([{} as Range], {} as Range);
    assert.deepEqual(events, [
      "clear",
      `delete:${FIND_HIGHLIGHT}`,
      `set:${FIND_HIGHLIGHT}`,
      "clear",
      `delete:${FIND_HIGHLIGHT_ACTIVE}`,
      `set:${FIND_HIGHLIGHT_ACTIVE}`,
    ]);

    events.length = 0;
    paintHighlights([], null);
    assert.deepEqual(events, [
      "clear",
      `delete:${FIND_HIGHLIGHT}`,
      "clear",
      `delete:${FIND_HIGHLIGHT_ACTIVE}`,
      "opacity:0.999999",
      "reflow",
      "opacity:",
    ]);
  } finally {
    for (const [key, value, had] of [
      ["CSS", saved.css, saved.hadCss],
      ["document", saved.document, saved.hadDocument],
      ["Highlight", saved.highlight, saved.hadHighlight],
      ["HTMLElement", saved.htmlElement, saved.hadHtmlElement],
    ] as const) {
      if (had) Object.assign(scope, { [key]: value });
      else delete scope[key];
    }
  }
});

test("the bar keeps itself out of the region it searches", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(bar, new RegExp(`${FIND_SKIP_ATTRIBUTE}=`));
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

  // The composer is the reference, not a copied colour, so a restyle fails here.
  const background = /background-color:\s*(#[0-9a-f]{6});/i.exec(composer);
  assert.ok(background);
  assert.match(bar, new RegExp(`background-color:\\s*${background[1]};`, "i"));

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
  assert.ok(
    to.alpha < from.alpha,
    `alpha ${to.alpha} is not softer than ${from.alpha}`,
  );
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
  // A bar at `--card` dissolves into what scrolls under it; past `--border` it reads as an edge.
  const bar = grey(value(".dark .find-bar-surface", "background-color"));
  const card = grey(value(".dark", "--card"));
  const border = grey(value(".dark", "--border"));
  assert.ok(bar > card, `bar ${bar} is not lighter than --card ${card}`);
  assert.ok(bar < border, `bar ${bar} is not darker than --border ${border}`);
  assert.match(
    value(".dark .find-bar-surface", "box-shadow"),
    /var\(--background\)/,
  );
});

test("the bar stays out of a backgrounded scope, and off the document origin", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-bar.tsx",
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
  // Fixed, not absolute: on a route whose outer container scrolls, an absolute bar scrolls away.
  const surface = /className="(find-bar-surface[^"]*)"/.exec(bar);
  assert.ok(surface);
  assert.match(surface[1], /\bfixed\b/);
  assert.equal(/\babsolute\b/.test(surface[1]), false);
  assert.match(surface[1], /max-w-\[calc\(100vw-2rem\)\]/);

  // The counter grows from `1/2` to `1234/5000+` in a long chat. The pill must keep its nominal
  // responsive width and let the query field yield that already-reserved room instead of growing.
  // 22.25/28.25rem is exactly the previous short-counter width: fixed input + 12rem chrome.
  assert.match(surface[1], /(?:^|\s)w-\[22\.25rem\](?:\s|$)/);
  assert.match(surface[1], /(?:^|\s)sm:w-\[28\.25rem\](?:\s|$)/);
  const input = /<input[\s\S]*?className=\{cn\([\s\S]*?"([^"]*)"/.exec(bar);
  assert.ok(input);
  assert.match(input[1], /\bflex-1\b/);
  assert.equal(/\bw-(?:40|64)\b/.test(input[1]), false);
});

test("the reveal looks again while the scroll is still moving", async () => {
  // Such a subtree contributes placeholder height until it renders, clamping the first scroll
  // 3415px short on all three engines. The node suite cannot see a scroll.
  const dom = await readFile(
    new URL("../src/features/find-in-page/lib/find-dom.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    dom,
    /export function scrollRangeIntoView\(range: Range\): boolean/,
  );
  const reveal = dom.slice(dom.indexOf("function revealPass("));
  const body = reveal.slice(0, reveal.indexOf("\n}\n"));
  assert.match(
    body,
    /if \(!scrollRangeIntoView\(range\) \|\| tries <= 1\) return;/,
  );
  assert.match(body, /tries - 1/);
  assert.match(dom, /revealRangeWhenPainted\(range: Range, tries = \d\)/);
  assert.match(body, /requestAnimationFrame\(/);
  assert.match(body, /range\.startContainer\.isConnected/);
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
  // The workspace stays mounted when the bar closes, so `isConnected` stays true and the old
  // chain keeps scrolling to a dismissed match.
  const dom = await readFile(
    new URL("../src/features/find-in-page/lib/find-dom.ts", import.meta.url),
    "utf8",
  );
  const entry = dom.slice(
    dom.indexOf("export function revealRangeWhenPainted"),
  );
  assert.match(
    entry.slice(0, entry.indexOf("\n}\n")),
    /cancelRevealPasses\(\)/,
  );
  const pass = dom.slice(dom.indexOf("function revealPass("));
  assert.match(
    pass.slice(0, pass.indexOf("\n}\n")),
    /if \(generation !== revealGeneration\) return;/,
  );
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
  // V8 compiles lazily, so guarding the constructor alone left the throw coming out through the
  // keystroke. Whitespace is what gets a query there: every run becomes `\\s+`.
  const index = buildTextIndex(el("DIV", [el("P", [text("b".repeat(60000))])]));
  // Shorter than the haystack, so the length guard cannot short-circuit it.
  const query = "a ".repeat(10000).trim();
  assert.ok(query.length < index.text.length);
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
  assert.deepEqual(findMatches(index, query, 10), []);
});

test("a Hangul query finds the syllables it is looking at", () => {
  // NFD takes a syllable apart into Jamo, no combining marks, so the general rule made three
  // clusters that each composed back to themselves.
  const index = buildTextIndex(
    el("DIV", [el("P", [text("\uac00\ub098\ub2e4 hello \ud55c\uad6d\uc5b4")])]),
  );
  assert.deepEqual(findMatches(index, "\uac00", 10), [{ start: 0, end: 1 }]);
  assert.deepEqual(findMatches(index, "\uac00\ub098\ub2e4", 10), [
    { start: 0, end: 3 },
  ]);
  assert.deepEqual(findMatches(index, "\ud55c\uad6d\uc5b4", 10), [
    { start: 10, end: 13 },
  ]);
  assert.deepEqual(
    findMatches(index, "\uac00\ub098\ub2e4".normalize("NFD"), 10),
    [{ start: 0, end: 3 }],
  );
  const decomposed = buildTextIndex(
    el("DIV", [el("P", [text("\uac00\ub098\ub2e4".normalize("NFD"))])]),
  );
  assert.equal(findMatches(decomposed, "\uac00\ub098\ub2e4", 10).length, 1);
});

test("a Hangul syllable is matched whole, in every spelling it can be stored in", () => {
  // Three canonical spellings, not two. Besides fully composed and fully decomposed there is the
  // half-composed one, an LV syllable followed by a loose trailing Jamo, which is exactly what
  // joining two text nodes produces when the syllable straddles them.
  const GA = "\uac00"; // 가
  const GAG = "\uac01"; // 각, the same syllable closed by a trailing consonant
  const GAG_NFD = "\u1100\u1161\u11a8";
  const GAG_HALF = "\uac00\u11a8";
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));

  // Every spelling of the text is reachable from every spelling of the query.
  for (const body of [GAG, GAG_NFD, GAG_HALF]) {
    for (const query of [GAG, GAG_NFD, GAG_HALF]) {
      const hits = findMatches(index(body), query, 10);
      assert.equal(
        hits.length,
        1,
        `${escape(body)} searched for ${escape(query)}`,
      );
      // And the highlight covers the whole syllable as that text spells it, never part of it.
      assert.deepEqual(hits[0], { start: 0, end: body.length });
    }
  }

  // An open syllable must not stop short of the trailing Jamo that closes the one on screen.
  // Decomposed, 가 would otherwise match the first two thirds of 각 and highlight part of a letter,
  // while composed text never could, since there the whole syllable is one code point.
  for (const body of [GAG, GAG_NFD, GAG_HALF]) {
    assert.deepEqual(findMatches(index(body), GA, 10), [], escape(body));
  }
  // The open syllable is still found where it really is open.
  assert.deepEqual(findMatches(index(GA), GA, 10), [{ start: 0, end: 1 }]);
});

test("Hangul clusters keep every Jamo, and only a real syllable is fenced", () => {
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));

  // A grapheme can carry more than one trailing Jamo. The half-composed spelling has to keep the
  // whole suffix: dropping all but the first both shortened the match and let it land on text that
  // lacks the query's final Jamo.
  const TWO_TRAILING = "\uac00\u11a8\u11a8";
  assert.deepEqual(findMatches(index(TWO_TRAILING), TWO_TRAILING, 10), [
    { start: 0, end: 3 },
  ]);
  assert.deepEqual(
    findMatches(index("\uac01\u11a8"), "\u1100\u1161\u11a8\u11a8", 10),
    [{ start: 0, end: 2 }],
    "the same grapheme, spelt half-composed in the text",
  );
  // ... but never on text that is missing the last Jamo the query asked for.
  assert.deepEqual(
    findMatches(index("\uac00\u11a8"), "\uac00\u11a8\u11a8", 10),
    [],
  );

  // A bare leading Jamo is its own grapheme, not a syllable waiting to be closed, so a trailing
  // Jamo after one belongs to something else and must not be fenced off.
  assert.deepEqual(
    findMatches(index("\uac00\u1100\u11a8"), "\uac00\u1100", 10),
    [{ start: 0, end: 2 }],
  );
});

test("a Hangul match starts and stops on grapheme boundaries", () => {
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const GAG = "\uac01"; // 각
  const GA = "\uac00"; // 가
  const LEAD = "\u1100"; // a bare leading Jamo

  // Not stopping inside one. A query carrying its own trailing Jamo could still match the prefix
  // of a grapheme that carries two, so the fence belongs after closed syllables as well as open.
  for (const body of [`${GAG}\u11a8`, `${LEAD}\u1161\u11a8\u11a8`]) {
    assert.deepEqual(findMatches(index(body), GAG, 10), [], escape(body));
  }
  // Nor starting inside one: a grapheme can carry more than one leading Jamo, and a match that
  // begins at the second highlights only its tail.
  for (const body of [`${LEAD}${GA}`, `${LEAD}${LEAD}\u1161`]) {
    assert.deepEqual(findMatches(index(body), GA, 10), [], escape(body));
  }
  // The fence is on the first cluster only, so a query that itself opens with a bare leading Jamo
  // still finds exactly that text.
  assert.deepEqual(findMatches(index(`${LEAD}${GA}`), `${LEAD}${GA}`, 10), [
    { start: 0, end: 2 },
  ]);
  // And an ordinary syllable is still found where it really stands alone.
  assert.deepEqual(findMatches(index(`${GA} hello`), GA, 10), [
    { start: 0, end: 1 },
  ]);
});

test("an engine without lookbehind falls back rather than throwing", () => {
  // The grapheme fence uses lookbehind, which JavaScriptCore only shipped in Safari 16.4. The
  // pattern is built from a string, so an older engine throws at construction, where `matchPattern`
  // already catches it and hands the search to the literal scan.
  const real = globalThis.RegExp;
  const refuseLookbehind = function (source: string, flags?: string) {
    if (typeof source === "string" && source.includes("(?<")) {
      throw new SyntaxError("Invalid regular expression");
    }
    return new real(source, flags);
  };
  refuseLookbehind.prototype = real.prototype;
  globalThis.RegExp = refuseLookbehind as unknown as RegExpConstructor;
  try {
    const index = buildTextIndex(
      el("DIV", [el("P", [text("\uac00\ub098\ub2e4 hello")])]),
    );
    // Exact queries still work; what is lost is only the flexing the pattern would have added.
    assert.deepEqual(findMatches(index, "\uac00\ub098\ub2e4", 10), [
      { start: 0, end: 3 },
    ]);
    assert.deepEqual(findMatches(index, "hello", 10), [{ start: 4, end: 9 }]);
  } finally {
    globalThis.RegExp = real;
  }
});

test("no Hangul match ever begins or ends inside a grapheme", () => {
  // Five rounds of review found five ways to stop or start half way through a Hangul syllable,
  // each a range that had not been thought of. So this asserts the property rather than the cases,
  // against `Intl.Segmenter` as the authority on where a grapheme ends.
  const segmenter = new Intl.Segmenter("ko", { granularity: "grapheme" });
  const boundaries = (body: string) => {
    const edges = new Set([0]);
    let at = 0;
    for (const { segment } of segmenter.segment(body)) {
      at += segment.length;
      edges.add(at);
    }
    return edges;
  };

  // Leading, vowel and trailing Jamo from the main block and from Extended-A and Extended-B.
  const leads = ["\u1100", "\u1101", "\ua960"];
  const vowels = ["\u1161", "\u1162", "\ud7b0"];
  const trails = ["\u11a8", "\u11a9", "\ud7cb"];
  const corpus = new Set([
    "hello \uac00",
    "\uac00 hello",
    "\uac00\ub098\ub2e4",
    "caf\u00e9",
    "cafe\u0301",
    "\uac00\u0301", // a syllable and a combining mark are one grapheme
    "\uac00\u200d\ub098", // ... and so is a joiner between two
    "\u0600\uac00", // a Prepend joins whatever follows it
    "\u1100\uac00\ub098", // two leading Jamo, then a second syllable
  ]);
  for (const lead of leads) {
    corpus.add(lead);
    for (const vowel of vowels) {
      const open = lead + vowel;
      corpus.add(open);
      corpus.add(open.normalize("NFC"));
      for (const other of vowels) corpus.add(open + other);
      for (const other of leads) {
        corpus.add(lead + other + vowel);
        corpus.add(lead + other + vowel + "\ub098");
      }
      corpus.add(open.normalize("NFC") + "\u0301");
      corpus.add("\u0600" + open.normalize("NFC"));
      for (const trail of trails) {
        const closed = open + trail;
        corpus.add(closed);
        corpus.add(closed.normalize("NFC"));
        corpus.add(open.normalize("NFC") + trail);
        for (const other of trails) corpus.add(closed + other);
      }
    }
  }

  let checked = 0;
  for (const body of corpus) {
    const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
    const edges = boundaries(index.text);
    for (const query of corpus) {
      for (const hit of findMatches(index, query, 50)) {
        checked += 1;
        assert.ok(
          edges.has(hit.start) && edges.has(hit.end),
          `${escape(body)} searched for ${escape(query)} gave ${hit.start}..${hit.end}`,
        );
      }
    }
    // And every string still finds itself, which is what a fence is easiest to break.
    assert.ok(
      findMatches(index, body, 10).length >= 1,
      `${escape(body)} cannot find itself`,
    );
    // A fence that is too eager is the other failure, and it does not show up as a bad range: the
    // match simply goes missing. Every leading run of whole graphemes must still be findable.
    let prefix = "";
    for (const { segment } of segmenter.segment(index.text)) {
      prefix += segment;
      if (prefix === index.text) break;
      assert.ok(
        findMatches(index, prefix, 10).length >= 1,
        `${escape(body)} cannot find its own prefix ${escape(prefix)}`,
      );
    }
  }
  assert.ok(checked > 200, `only ${checked} matches exercised`);
});

test("the grapheme fences do not depend on lookbehind", async () => {
  // JavaScriptCore only shipped lookbehind in Safari 16.4, and a pattern using one throws on older
  // engines straight into the unfenced literal scan, quietly undoing both boundaries. So the start
  // of the fence is checked in code and the pattern carries none.
  const source = await readFile(
    new URL(
      "../src/features/find-in-page/lib/find-text-index.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal(source.includes("(?<"), false, "no lookbehind in the pattern");

  const real = globalThis.RegExp;
  const refuse = function (pattern: string, flags?: string) {
    if (typeof pattern === "string" && pattern.includes("(?<")) {
      throw new SyntaxError("Invalid regular expression");
    }
    return new real(pattern, flags);
  };
  refuse.prototype = real.prototype;
  globalThis.RegExp = refuse as unknown as RegExpConstructor;
  try {
    const closed = buildTextIndex(el("DIV", [el("P", [text("\uac01\u11a8")])]));
    assert.deepEqual(findMatches(closed, "\uac01", 10), []);
    const led = buildTextIndex(el("DIV", [el("P", [text("\u1100\uac00")])]));
    assert.deepEqual(findMatches(led, "\uac00", 10), []);
    const plain = buildTextIndex(
      el("DIV", [el("P", [text("\uac00\ub098\ub2e4")])]),
    );
    assert.deepEqual(findMatches(plain, "\uac00", 10), [{ start: 0, end: 1 }]);
  } finally {
    globalThis.RegExp = real;
  }
});

test("the grapheme boundary is the platform's answer, not a list of ranges", () => {
  // Enumerating the ranges by hand kept missing one: Hangul Jamo, combining marks, Prepend,
  // spacing marks, skin tones. The question goes to `Intl.Segmenter` instead, which knows the whole
  // of UAX 29. These are the cases that were wrong before it did.
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  for (const [body, query] of [
    ["\uac00\u093e", "\uac00"], // a spacing mark
    ["\uac00\u{1f3fb}", "\uac00"], // an astral modifier
    ["\u{1193f}\uac00", "\uac00"], // a supplementary Prepend
    ["\uac00\u0301", "\uac00"], // a combining mark
    ["\u0600\uac00", "\uac00"], // a BMP Prepend
    ["\uac01\u11a8", "\uac01"], // one trailing Jamo too few
    ["\u1100\uac00", "\uac00"], // starting after a leading Jamo
  ] as const) {
    assert.deepEqual(findMatches(index(body), query, 10), [], escape(body));
  }
  // Whitespace ends a grapheme, so a query that ends in a space is not held to what follows it.
  assert.deepEqual(findMatches(index("\uac00 \u11a8"), "\uac00 ", 10), [
    { start: 0, end: 2 },
  ]);
  // An emoji sequence is one grapheme too, and this was never Hangul-specific.
  assert.deepEqual(
    findMatches(
      index("\u{1f469}\u200d\u{1f469}\u200d\u{1f466}"),
      "\u{1f469}",
      10,
    ),
    [],
  );
});

test("a query that needs no pattern is not given one", () => {
  // Forcing Hangul through the regex path so the fences could apply meant a large paste built a
  // pattern that V8 accepted and then refused to run, and the throw escaped the search entirely.
  // With the boundary asked of the segmenter instead, no query needs a pattern it did not earn.
  const body = "\u11a8".repeat(50_000);
  const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  assert.deepEqual(findMatches(index, body, 10), [{ start: 0, end: 50_000 }]);
});

test("plain text does not pay for the boundary check", () => {
  // The segmenter is asked only where something could actually join, which nothing below U+0300
  // can. Latin prose therefore costs one comparison per match rather than a segmentation.
  const source = readFileSync(
    new URL(
      "../src/features/find-in-page/lib/find-text-index.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(source, /const JOINS_GRAPHEME = \/\[\^\\u0000-\\u02ff\]\//);
  const guard = source.slice(source.indexOf("function alignsToGraphemes"));
  const before = guard.indexOf("JOINS_GRAPHEME");
  const asks = guard.indexOf("graphemeSegmenter()");
  assert.ok(before > 0 && before < asks, "the cheap test comes first");
});

test("the cheap boundary test looks at both sides of each edge", () => {
  // The shortcut asked what sat outside the match and not what sat at its edges, so a query that
  // itself ends in a character joining forwards slipped through: a Prepend at the start of the
  // text has nothing before it and an ordinary letter after it, and both outside looks passed.
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  assert.deepEqual(findMatches(index("\u0600a"), "\u0600", 10), []);
  // The same from the other side, where the match begins with a mark that joins backwards.
  assert.deepEqual(findMatches(index("a\u0301"), "\u0301", 10), []);
});

test("text with no whitespace for hundreds of characters is still fenced", () => {
  // The window is anchored at the nearest whitespace or block separator, and where there is none
  // in reach the match used to be waved through. A log line, a URL or a long identifier has none
  // for hundreds of characters, which put the original hole straight back.
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const run = "a".repeat(257);
  assert.deepEqual(findMatches(index(`${run}\uac01\u11a8`), "\uac01", 10), []);
  // ... while a match that really is whole is still found out there.
  assert.deepEqual(findMatches(index(`${run}\uac01 x`), "\uac01", 10), [
    { start: 257, end: 258 },
  ]);
});

test("whitespace is not treated as a grapheme boundary", () => {
  // A combining mark joins a preceding space and a Prepend joins a following one, so anchoring at
  // a space cut off the context that decides the join. Only the block separator can be picked up
  // from, being a control, which UAX 29 breaks on either side unconditionally.
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  assert.deepEqual(findMatches(index("a \u0301"), "\u0301", 10), []);
  assert.deepEqual(findMatches(index("\u0600 "), "\u0600", 10), []);
});

test("a long run of regional indicators keeps its parity", () => {
  // Flags pair off by the parity of the run they sit in, and a run has no length limit, so no
  // fixed amount of preceding context is enough to work out where the pairs fall.
  const flags = "\u{1f1e6}".repeat(129) + "\u{1f1e8}\u{1f1e9}";
  const index = buildTextIndex(el("DIV", [el("P", [text(flags)])]));
  // The 129th indicator pairs with the one after it, so the last two are not a grapheme of their own.
  assert.deepEqual(findMatches(index, "\u{1f1e8}\u{1f1e9}", 10), []);
});

test("a capped search does not pay twice for the same segmentation", () => {
  // A capped search walks its candidates more than once, to count them and again to take the
  // window around the viewport. The segmentation is made once and kept for as long as the index
  // lives, and is asked one offset at a time rather than block by block.
  const body = "\uac00\ub098\ub2e4".repeat(20_000);
  const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const anchor = () => index.text.length - 10;
  const first = Date.now();
  assert.equal(
    findMatches(index, "\uac00", MAX_MATCHES + 1, anchor).length,
    MAX_MATCHES + 1,
  );
  const cost = Date.now() - first;
  const second = Date.now();
  findMatches(index, "\uac00", MAX_MATCHES + 1, anchor);
  // The second search reuses the segmentation, so it cannot be slower than the first was.
  assert.ok(
    Date.now() - second <= cost + 50,
    `second search took ${Date.now() - second}ms against ${cost}ms`,
  );
});

test("a cluster is offered its longest spelling first", () => {
  // Both spellings of one cluster, and alternation takes the first that fits. Shortest first, the
  // bare `i` won, the match ended between the letter and its dot, and the fence threw the whole
  // occurrence away rather than reaching for the longer spelling.
  const dotted = buildTextIndex(el("DIV", [el("P", [text("İstanbul")])]));
  assert.deepEqual(findMatches(dotted, "i", 10), [{ start: 0, end: 2 }]);
  assert.deepEqual(findMatches(dotted, "istanbul", 10), [{ start: 0, end: 9 }]);
});

test("a clipped tail cannot vouch for its own last offset", () => {
  // A node cut at `MAX_NODE_CHARS` ends where the walk stopped, not where the text does. What was
  // left out is still on the page, and here it is the trailing Jamo that closes the syllable the
  // index now ends on, so a match on the bare vowel form would paint over half a character.
  const node = text(`${"a".repeat(MAX_NODE_CHARS - 1)}각`);
  const index = buildTextIndex(el("DIV", [el("P", [node])]));
  assert.deepEqual([...index.unsafe], [MAX_NODE_CHARS]);
  assert.equal(index.text.length, MAX_NODE_CHARS);
  assert.deepEqual(findMatches(index, "가", 10), []);
  // An index that ends because the document does keeps its last offset.
  const whole = buildTextIndex(el("DIV", [el("P", [text("가")])]));
  assert.equal(whole.unsafe.size, 0);
  assert.deepEqual(findMatches(whole, "가", 10), [{ start: 0, end: 1 }]);
});

test("the cheap boundary test does not shortcut past a clipped end", () => {
  // Latin text takes four comparisons and skips the segmenter, and one of the four asks what
  // follows the match. Nothing does, at a clipped end, so the shortcut read that as room to spare
  // and returned an `x` whose combining mark had been left on the page.
  const node = text(`unsloth${"x".repeat(MAX_NODE_CHARS - 8)}z\u0301`);
  const index = buildTextIndex(el("DIV", [el("P", [node])]));
  assert.deepEqual([...index.unsafe], [MAX_NODE_CHARS]);
  assert.equal(index.text.length, MAX_NODE_CHARS);
  assert.deepEqual(findMatches(index, "z", 10), []);
  // Everything before the end is still found on the fast path, which is most of a document.
  assert.deepEqual(findMatches(index, "unsloth", 10), [{ start: 0, end: 7 }]);
});

test("a full ceiling behind a block boundary still ends on a boundary", () => {
  // A separator was already due, so what is left out is behind a break and cannot reach back: the
  // last character indexed is whole. Treating a full ceiling like a cut lost it.
  const nodes = Array.from(
    { length: MAX_INDEX_CHARS / MAX_NODE_CHARS },
    (_unused, at) =>
      text(
        at === MAX_INDEX_CHARS / MAX_NODE_CHARS - 1
          ? `${"y".repeat(MAX_NODE_CHARS - 1)}Q`
          : "x".repeat(MAX_NODE_CHARS),
      ),
  );
  const index = buildTextIndex(
    el("DIV", [el("P", nodes), el("P", [text("left out")])]),
  );
  assert.equal(index.text.length, MAX_INDEX_CHARS);
  assert.equal(index.truncated, true);
  assert.equal(index.unsafe.size, 0);
  assert.deepEqual(findMatches(index, "q", 10), [
    { start: MAX_INDEX_CHARS - 1, end: MAX_INDEX_CHARS },
  ]);
});

test("a cut in the middle of the document is unsafe on both sides", () => {
  // The separator a cut writes is not the boundary it looks like: what was dropped is still on the
  // page between the two and can carry across it. One flag for the end of the walk forgot the cut
  // as soon as anything else was indexed. Spelt out, so the cut falls inside the syllable.
  const index = buildTextIndex(
    el("DIV", [
      el("P", [
        text(`${"a".repeat(MAX_NODE_CHARS - 1)}\uac00\u11a8`),
        text("tail"),
      ]),
    ]),
  );
  // The near side only. What was dropped is a trailing Jamo, which reaches back into the vowel form
  // before it, but forward only into another trailing Jamo, and `tail` starts with none.
  assert.deepEqual([...index.unsafe], [MAX_NODE_CHARS]);
  assert.equal(index.text[MAX_NODE_CHARS - 1], "\uac00");
  assert.deepEqual(findMatches(index, "\uac00", 10), []);
  assert.equal(findMatches(index, "tail", 10).length, 1);
});

test("what was dropped decides the far side of a cut, not what was kept", () => {
  // Nothing below U+0300 joins backwards, which made an ASCII character look safe, but a Prepend
  // in the dropped text joins forwards into it. Settled while the index is built, which is the only
  // time the dropped text can be seen.
  const reaching = buildTextIndex(
    el("DIV", [
      el("P", [text(`${"x".repeat(MAX_NODE_CHARS)}\u0600`), text("a")]),
    ]),
  );
  // The near side is settled by the same reading: a Prepend joins forwards, not back into the `x`
  // before it, so only the far side is in doubt.
  assert.deepEqual([...reaching.unsafe], [MAX_NODE_CHARS + 1]);
  assert.deepEqual(findMatches(reaching, "a", 10), []);
  // A dropped tail that cannot reach forward leaves the next node whole, even in Hangul.
  const settled = buildTextIndex(
    el("DIV", [el("P", [text(`${"x".repeat(MAX_NODE_CHARS)}zz`), text("가")])]),
  );
  assert.deepEqual([...settled.unsafe], []);
  assert.equal(findMatches(settled, "가", 10).length, 1);
});

test("the dropped tail is read back far enough to answer for itself", () => {
  // A linker with marks after it still joins what follows, so the junction has to see back past
  // the marks to the linker, and past a ZWJ to its pictograph. Both chains are unbounded in UAX 29,
  // so there is a window, and outrunning it is called unknown rather than guessed at.
  const conjunct = buildTextIndex(
    el("DIV", [
      el("P", [
        text(`${"x".repeat(MAX_NODE_CHARS)}\u0915\u094d\u0301`),
        text("\u0924!!"),
      ]),
    ]),
  );
  assert.deepEqual([...conjunct.unsafe], [MAX_NODE_CHARS + 1]);
  assert.deepEqual(findMatches(conjunct, "\u0924", 10), []);
  // A tail of marks longer than the window keeps nothing to hang them on, so it stays unsafe.
  const beyond = buildTextIndex(
    el("DIV", [
      el("P", [
        text(`${"x".repeat(MAX_NODE_CHARS)}\u0915${"\u0301".repeat(64)}`),
        text("\u0924!!"),
      ]),
    ]),
  );
  assert.deepEqual([...beyond.unsafe], [MAX_NODE_CHARS + 1]);
});

test("no match a cut leaves behind disagrees with the page it was cut from", () => {
  // The property behind the four cases above, over junctions built from the shapes that have gone
  // wrong: every match is mapped back through the segment table to the uncut page text and has to
  // land where the platform breaks. Fixed sequence, so a failure repeats.
  const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
  const at = (code: number) => String.fromCodePoint(code);
  const pieces = [
    "z",
    " ",
    "؀",
    "가",
    "각",
    "ᄀ",
    "ᅡ",
    "ᆨ",
    at(0x1f1e6),
    at(0x1f1e7),
    "‍",
    "‌",
    at(0x1f469),
    at(0x1f44d),
    at(0x1f3fb),
    "क",
    "्",
    "त",
    "́",
    "ा",
    "a",
    at(0x1193f),
    "️",
  ];
  let seed = 15200;
  const pick = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return pieces[seed % pieces.length];
  };
  let checked = 0;
  let usableFarSide = 0;
  for (let round = 0; round < 60; round += 1) {
    let dropped = "";
    for (let i = 0; i < 1 + (round % 4); i += 1) dropped += pick();
    let resumed = "";
    for (let i = 0; i < 1 + (round % 3); i += 1) resumed += pick();
    const first = `${"x".repeat(MAX_NODE_CHARS)}${dropped}`;
    const second = `${resumed}QQ`;
    const index = buildTextIndex(
      el("DIV", [el("P", [text(first), text(second)])]),
    );
    const page = first + second;
    const marks = new Uint8Array(page.length + 1);
    for (const { index: offset } of segmenter.segment(page)) marks[offset] = 1;
    marks[page.length] = 1;
    if (!index.unsafe.has(MAX_NODE_CHARS + 1)) usableFarSide += 1;
    const toPage = (offset: number) => {
      for (const segment of index.segments) {
        if (
          offset >= segment.start &&
          offset <= segment.start + segment.length
        ) {
          const base =
            segment.node === index.segments[0].node ? 0 : first.length;
          return base + (offset - segment.start);
        }
      }
      return -1;
    };
    for (const query of new Set([resumed, "QQ", "xxx", dropped])) {
      for (const hit of findMatches(index, query, 20)) {
        const start = toPage(hit.start);
        const end = toPage(hit.end);
        if (start < 0 || end < 0) continue;
        checked += 1;
        assert.equal(
          marks[start],
          1,
          `${escape(query)} starts inside a grapheme`,
        );
        assert.equal(marks[end], 1, `${escape(query)} ends inside a grapheme`);
      }
    }
  }
  assert.ok(checked > 200, `only ${checked} matches checked`);
  // Refusing everything would satisfy the above, so the other direction is asserted too.
  assert.ok(
    usableFarSide > 10,
    `only ${usableFarSide} junctions kept a usable far side`,
  );
});

test("what the dropped tail was is not the question; what it meets is", () => {
  // Refusing on the class of the last dropped code point could only ever say no. A closed syllable
  // reaches forward into a trailing Jamo and nothing else, and an even run of indicators pairs off
  // among itself, so the next node begins exactly where it appears to.
  const cut = (dropped: string, resumed: string) =>
    buildTextIndex(
      el("DIV", [
        el("P", [
          text(`${"x".repeat(MAX_NODE_CHARS)}${dropped}`),
          text(`${resumed}QQ`),
        ]),
      ]),
    );
  const at = (code: number) => String.fromCodePoint(code);
  for (const [dropped, resumed] of [
    ["각", "hello"],
    ["ᆨ", "hello"],
    [at(0x1f1e6) + at(0x1f1e7), at(0x1f1e8)],
  ] as const) {
    assert.equal(
      findMatches(cut(dropped, resumed), resumed, 5).length,
      1,
      `${escape(dropped)} then ${escape(resumed)}`,
    );
  }
  // And still fenced where the two really do join.
  for (const [dropped, resumed] of [
    ["ᄀ", "가"],
    ["가", "ᆨ"],
    [at(0x1f1e6), at(0x1f1e7)],
    [at(0x1f469) + "\u200d", at(0x1f469)],
  ] as const) {
    assert.deepEqual(
      findMatches(cut(dropped, resumed), resumed, 5),
      [],
      `${escape(dropped)} then ${escape(resumed)}`,
    );
  }
});

test("an odd run of regional indicators displaces the whole run after a cut", () => {
  // Parity is counted from the separator, so a run the cut fell inside has every boundary in it
  // moved, not only the one at the seam. Found by the junction fuzz, not by a review.
  const at = (code: number) => String.fromCodePoint(code);
  const flags = at(0x1f1e7) + at(0x1f1e6) + at(0x1f1e7);
  const index = buildTextIndex(
    el("DIV", [
      el("P", [
        text(`${"x".repeat(MAX_NODE_CHARS)}${at(0x1f1e6)}`),
        text(`${flags}a`),
      ]),
    ]),
  );
  // Every indicator in the resumed run, not just the first: the ones the text reads as boundaries
  // are inside a flag, and the boundary between them is where the page really breaks.
  // Asked of the offsets rather than of the whole set: doubt is held as runs now, so an indicator
  // covers the unit inside its surrogate pair as well as its start, and that unit begins nothing
  // either way.
  assert.equal(index.unsafe.has(MAX_NODE_CHARS + 1), true);
  assert.equal(index.unsafe.has(MAX_NODE_CHARS + 5), true);
  assert.equal(index.unsafe.has(MAX_NODE_CHARS + 3), false);
  assert.equal(index.shifted.has(MAX_NODE_CHARS + 3), true);
  // No indicator here stands alone on the page, so none is findable on its own either way.
  assert.deepEqual(findMatches(index, at(0x1f1e7), 10), []);
  // The flag the displacement makes, which the run being called unknown used to throw away.
  assert.equal(findMatches(index, at(0x1f1e6) + at(0x1f1e7), 10).length, 1);
});

test("a cut whose context outran its window does not outlive the cut", () => {
  // Outrunning the window makes a junction unknown, which is the right answer for that junction and
  // for no other. Held as a flag beside the context rather than inside it, it survived the block
  // boundary that cleared the context and went on refusing the start of every later block.
  const at = (code: number) => String.fromCodePoint(code);
  const flag = at(0x1f1e6) + at(0x1f1e7);
  const index = buildTextIndex(
    el("DIV", [
      el("SPAN", [text(`${"x".repeat(MAX_NODE_CHARS)}${"\u0301".repeat(40)}`)]),
      el("P", [text(flag)]),
    ]),
  );
  assert.deepEqual([...index.unsafe], [MAX_NODE_CHARS]);
  assert.equal(findMatches(index, flag, 10).length, 1);
  // The window itself still does its job at the junction it belongs to.
  const inline = buildTextIndex(
    el("DIV", [
      el("P", [
        text(`${"x".repeat(MAX_NODE_CHARS)}\u0915${"\u0301".repeat(40)}`),
        text("\u0924!!"),
      ]),
    ]),
  );
  assert.deepEqual([...inline.unsafe], [MAX_NODE_CHARS + 1]);
});

test("a portal is its own surface, whatever the workspace ended on", () => {
  // Nothing a portal paints carries on from text cut out of the workspace behind it. The block
  // branch clears the cut leaving any block tag, which covered a `DIV` scope and `DIV` portal and
  // hid it for everything else: the boundary belongs to the portal, not to the tags either side.
  const index = buildTextIndex(
    el("SPAN", [text(`${"x".repeat(MAX_NODE_CHARS)}\u1100`)]),
    [el("SPAN", [text("\uac00 hello")])],
  );
  assert.deepEqual([...index.unsafe], []);
  assert.equal(findMatches(index, "\uac00", 10).length, 1);
});

test("the retained edge of a cut is settled by what was dropped next", () => {
  // The whole node is still in hand at the cut, so refusing its last offset outright threw away a
  // word whose end the very next character proves. A space cannot continue anything.
  const proved = buildTextIndex(
    el("DIV", [
      el("P", [
        text(`${"y".repeat(MAX_NODE_CHARS - 10)}uniqueword trailing`),
        text("z"),
      ]),
    ]),
  );
  assert.equal(proved.truncated, true);
  assert.deepEqual([...proved.unsafe], []);
  assert.equal(findMatches(proved, "uniqueword", 5).length, 1);
  // And still refused where the dropped character does carry on.
  const joined = buildTextIndex(
    el("DIV", [
      el("P", [text(`${"y".repeat(MAX_NODE_CHARS)}\u0301tail`), text("z")]),
    ]),
  );
  assert.deepEqual([...joined.unsafe], [MAX_NODE_CHARS]);
});

test("an unknown chain reaches only what a rule out there could take", () => {
  // Running out of window makes the anchor unknown, and the rules that turn on it, GB9c and GB11,
  // want a letter or a pictograph on their right. A flag is neither, so it is answerable, and
  // treating the whole junction as unknown put every indicator in the run behind it out of reach.
  const at = (code: number) => String.fromCodePoint(code);
  const flag = at(0x1f1e6) + at(0x1f1e7);
  const index = buildTextIndex(
    el("DIV", [
      el("P", [
        text(`${"x".repeat(MAX_NODE_CHARS)}${"\u0301".repeat(40)}`),
        text(flag),
      ]),
    ]),
  );
  // The near side alone: a combining mark carries on from the `x`, a flag does not carry back.
  assert.deepEqual([...index.unsafe], [MAX_NODE_CHARS]);
  assert.equal(findMatches(index, flag, 5).length, 1);
});

test("the linker set is the one the platform joins on", () => {
  // Derived against the segmenter rather than a Unicode table read at some other version: the
  // Tulu-Tigalari conjoiner is newer than the table the first pass was filtered through, so it was
  // missing while the segmenter had known about it all along.
  const at = (code: number) => String.fromCodePoint(code);
  const conjoined = at(0x11380) + at(0x113d0) + at(0x11381);
  assert.equal(
    [
      ...new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(
        conjoined,
      ),
    ].length,
    1,
  );
  const index = buildTextIndex(
    el("DIV", [
      el("P", [
        text(`${"x".repeat(MAX_NODE_CHARS)}${at(0x11380)}${at(0x113d0)}`),
        text(`${at(0x11381)}QQ`),
      ]),
    ]),
  );
  assert.deepEqual(findMatches(index, at(0x11381), 5), []);
});

test("an odd run longer than the window is not read as even", () => {
  // Parity reaches as far as its run does. A context that ran out of window keeps an even tail of a
  // run that is odd behind it, which read as a boundary and ended a match inside the visible flag.
  const at = (code: number) => String.fromCodePoint(code);
  const index = buildTextIndex(
    el("DIV", [el("P", [text(`xx${at(0x1f1e6).repeat(50_000)}`)])]),
  );
  assert.equal(index.text.length, MAX_NODE_CHARS);
  assert.deepEqual([...index.unsafe], [MAX_NODE_CHARS]);
  for (const match of findMatches(index, at(0x1f1e6), MAX_MATCHES)) {
    assert.notEqual(match.end, MAX_NODE_CHARS);
  }
});

test("the time between two searches is not charged to either", () => {
  // The budget is wall time, and it is measured in blocks, so a search that ends inside one leaves
  // the clock running. The next search closes that block and is billed for everything in between,
  // which is the reader thinking. Two cheap queries and a pause bought a whole-index scan.
  const probe = `
    let scans = 0;
    const Real = Intl.Segmenter;
    Intl.Segmenter = class {
      constructor(...args) { this.inner = new Real(...args); }
      segment(input) {
        const segments = this.inner.segment(input);
        return {
          containing: (at) => segments.containing(at),
          [Symbol.iterator]: () => { scans += 1; return segments[Symbol.iterator](); },
        };
      }
    };
    const { buildTextIndex, findMatches, MAX_MATCHES } = await import(${JSON.stringify(
      new URL(
        "../src/features/find-in-page/lib/find-text-index.ts",
        import.meta.url,
      ).href,
    )});
    const el = (tagName, childNodes) => ({
      nodeType: 1, tagName, childNodes, getAttribute: () => null,
    });
    // Hangul, so every candidate is past the fast path and does ask the segmenter.
    const syllable = (at) => String.fromCodePoint(0xac00 + (at % 11172));
    let body = "";
    for (let at = 0; at < 100000; at += 1) body += syllable(at);
    const index = buildTextIndex(el("DIV", [el("P", [{ nodeType: 3, data: body }])]));
    const wait = (ms) => new Promise((done) => setTimeout(done, ms));
    // Each query needs well under a block of checks, so on its own none can reach the budget.
    for (let round = 1; round <= 4; round += 1) {
      findMatches(index, syllable(round * 13) + syllable(round * 13 + 1), MAX_MATCHES);
      await wait(60);
    }
    if (scans !== 0) throw new Error("bought a scan out of the pauses between searches");
  `;
  const run = spawnSync(
    process.execPath,
    ["--experimental-strip-types", "--input-type=module", "--eval", probe],
    { encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
});

test("what a seek is allowed to cost is time, not a number of them", () => {
  // A seek is not one price: 0.2us into a page of Hangul and 1236us into a page of flags, six
  // thousand to one on the same length of text, so any count is far too small for one and far too
  // large for the other. Counting let a flag-heavy page spend twenty thousand of the expensive kind
  // on its first search. Counted here, since the point is that they stop after a handful.
  const probe = `
    let seeks = 0;
    const Real = Intl.Segmenter;
    Intl.Segmenter = class {
      constructor(...args) { this.inner = new Real(...args); }
      segment(input) {
        const segments = this.inner.segment(input);
        return {
          containing: (at) => { seeks += 1; return segments.containing(at); },
          [Symbol.iterator]: () => segments[Symbol.iterator](),
        };
      }
    };
    const { buildTextIndex, findMatches, MAX_MATCHES, MAX_NODE_CHARS } = await import(${JSON.stringify(
      new URL(
        "../src/features/find-in-page/lib/find-text-index.ts",
        import.meta.url,
      ).href,
    )});
    const el = (tagName, childNodes) => ({
      nodeType: 1, tagName, childNodes, getAttribute: () => null,
    });
    const flag = (at) => String.fromCodePoint(0x1f1e6 + (at % 26));
    let run = "";
    for (let at = 0; at < MAX_NODE_CHARS / 2; at += 1) run += flag(at);
    const nodes = [];
    for (let at = 0; at < 13; at += 1) nodes.push({ nodeType: 3, data: run });
    const index = buildTextIndex(el("DIV", [el("P", nodes)]));
    const started = Date.now();
    // A pair that straddles two flags, so every candidate is rejected and every one asks.
    findMatches(index, flag(1) + flag(2), MAX_MATCHES);
    const took = Date.now() - started;
    // A count-based cap spends about 20,000 of these; a budget stops inside a few blocks of them.
    if (seeks > 2000) throw new Error("seeks: " + seeks + " in " + took + "ms");
  `;
  const run = spawnSync(
    process.execPath,
    ["--experimental-strip-types", "--input-type=module", "--eval", probe],
    { encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
});

test("a capped search does not segment the whole index to answer its first pass", () => {
  // The scan that replaces the seeks costs a pass over the whole index, so what it is allowed to
  // replace matters as much as that it exists. On a flat count a bounded pass reaches the threshold
  // on a big enough index and buys the whole document to answer ten thousand cheap questions.
  const probe = `
    let scans = 0;
    let seeks = 0;
    const Real = Intl.Segmenter;
    Intl.Segmenter = class {
      constructor(...args) { this.inner = new Real(...args); }
      segment(input) {
        const segments = this.inner.segment(input);
        return {
          containing: (at) => { seeks += 1; return segments.containing(at); },
          [Symbol.iterator]: () => { scans += 1; return segments[Symbol.iterator](); },
        };
      }
    };
    const { buildTextIndex, findMatches, MAX_MATCHES, MAX_NODE_CHARS } = await import(${JSON.stringify(
      new URL(
        "../src/features/find-in-page/lib/find-text-index.ts",
        import.meta.url,
      ).href,
    )});
    const el = (tagName, childNodes) => ({
      nodeType: 1, tagName, childNodes, getAttribute: () => null,
    });
    const nodes = [];
    for (let at = 0; at < 2000000; at += MAX_NODE_CHARS) {
      nodes.push({ nodeType: 3, data: "\uac00".repeat(MAX_NODE_CHARS) });
    }
    const index = buildTextIndex(el("DIV", [el("P", nodes)]));
    // What a seek costs on THIS machine, since that is what the budget is spent in and it varies
    // by orders of magnitude between engines and hosts. Measured through the same wrapper, then
    // discounted so the count it feeds is not itself the thing under test.
    const sample = new Intl.Segmenter(undefined, { granularity: "grapheme" }).segment(index.text);
    sample.containing(0);
    const startedSample = performance.now();
    for (let at = 0; at < 200; at += 1) sample.containing((at * 9871) % index.text.length);
    const perSeek = (performance.now() - startedSample) / 200;
    seeks = 0;
    // Anchored at the top, so the cap stops the search after one bounded pass.
    const found = findMatches(index, "\uac00", MAX_MATCHES, 0);
    if (found.length !== MAX_MATCHES) throw new Error("expected a capped search, got " + found.length);
    if (seeks > 4 * MAX_MATCHES) throw new Error("seeks: " + seeks);
    // The budget is what one scan of this index would cost, so a bounded pass reaches it only
    // where seeking is expensive enough that the scan is the better buy anyway. Assert the waste
    // this guards against, which is scanning while the seeks were still the cheaper option.
    const budgetMs = index.text.length / 16000;
    if (scans !== 0 && seeks * perSeek < budgetMs / 2) {
      throw new Error(
        "scanned after " + seeks + " seeks costing " + (seeks * perSeek).toFixed(1) +
        "ms, well inside a budget of " + budgetMs.toFixed(0) + "ms"
      );
    }
  `;
  const run = spawnSync(
    process.execPath,
    ["--experimental-strip-types", "--input-type=module", "--eval", probe],
    { encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
});

test("a query that matches everywhere stops seeking the segmenter per candidate", () => {
  // `containing` seeks, which is why it replaced segmenting whole blocks, and a seek per candidate
  // undoes that: a capped search anchored near the end walks the candidates up to three times, so a
  // page of one repeated syllable asked for millions. Counted, so the assertion is not the clock.
  const probe = `
    let seeks = 0;
    const Real = Intl.Segmenter;
    Intl.Segmenter = class {
      constructor(...args) { this.inner = new Real(...args); }
      segment(input) {
        const segments = this.inner.segment(input);
        return {
          containing: (at) => { seeks += 1; return segments.containing(at); },
          [Symbol.iterator]: () => segments[Symbol.iterator](),
        };
      }
    };
    const { buildTextIndex, findMatches, MAX_MATCHES, MAX_NODE_CHARS } = await import(${JSON.stringify(
      new URL(
        "../src/features/find-in-page/lib/find-text-index.ts",
        import.meta.url,
      ).href,
    )});
    const el = (tagName, childNodes) => ({
      nodeType: 1, tagName, childNodes, getAttribute: () => null,
    });
    const total = 400000;
    const nodes = [];
    for (let at = 0; at < total; at += MAX_NODE_CHARS) {
      nodes.push({ nodeType: 3, data: "\uac00".repeat(MAX_NODE_CHARS) });
    }
    const index = buildTextIndex(el("DIV", [el("P", nodes)]));
    const found = findMatches(index, "\uac00", MAX_MATCHES, index.text.length);
    if (found.length !== MAX_MATCHES) throw new Error("expected a capped search, got " + found.length);
    // One pass over the boundaries replaces the seeks, so what is left is what the budget bought
    // before it. The number is not fixed: the cap is time, and a seek into Hangul is cheap, so
    // this shape gets tens of thousands where a page of flags would get tens. 1.6M without it.
    if (seeks > 200000) throw new Error("seeks per candidate: " + seeks);
  `;
  const run = spawnSync(
    process.execPath,
    ["--experimental-strip-types", "--input-type=module", "--eval", probe],
    { encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
});

test("a cut reads back past itself for the anchor and the parity of what it split", () => {
  // The context a cut leaves was built from the dropped side alone, so a chain whose anchor was
  // retained looked anchorless and a run of indicators looked shorter than it is. Both are read
  // over the whole node now, since the cut is a place in the text and not a place on the page.
  const cons = "\u0915";
  const linker = "\u094d";
  const joined = "\u0937";
  const pictograph = String.fromCodePoint(0x1f469);
  const flag = (at: number) => String.fromCodePoint(0x1f1e6 + at);
  // The anchor is on the retained side and the linker is dropped: the three make one grapheme.
  const conjunct = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS - 1) + cons + linker),
        text(joined),
      ]),
    ]),
  );
  assert.deepEqual(findMatches(conjunct, joined, 10), []);
  // Same shape for GB11: a retained pictograph, a dropped ZWJ, a pictograph after the seam.
  const zwj = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS - 2) + pictograph + "\u200d"),
        text(pictograph),
      ]),
    ]),
  );
  assert.deepEqual(findMatches(zwj, pictograph, 10), []);
  // Parity counts the retained indicators too: the page pairs across the cut, so the two after
  // the seam are not a flag however they read in the index.
  const parity = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS - 2) + flag(0) + flag(1) + flag(2)),
        text(flag(3) + flag(4)),
      ]),
    ]),
  );
  assert.deepEqual(findMatches(parity, flag(3) + flag(4), 10), []);
});

test("the chain a cut leaves is followed past the seam, not only to it", () => {
  // The doubt a cut leaves does not stop at the first character of the next node: a linker there
  // carries it to the letter it joins. Found by putting chain characters on both sides of the cut,
  // which the junction fuzz had never done, and it cut a grapheme rather than merely losing one.
  const index = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS - 1) + "\u0937" + "\u094d"),
        text("\u17d2\u1000a"),
      ]),
    ]),
  );
  // The consonant after the seam is inside the grapheme the cut split, so it starts nothing.
  assert.deepEqual(findMatches(index, "\u1000", 10), []);
  // What lies past the end of the chain is still findable, or the fence has eaten the feature.
  assert.equal(findMatches(index, "a", MAX_MATCHES).length > 0, true);
});

test("a CRLF is one grapheme, which the fast path has to know about", () => {
  // The fast path rests on nothing below U+0300 joining, and CR before LF is the one thing there
  // that does (GB3). The sweep that proved the rest excused the line breaks, a collapsed newline
  // being its own grapheme. In a `<pre>` nothing is collapsed, the pair arrives intact, and every
  // character the shortcut looks at is ASCII, so it answered without asking anyone.
  const fence = el("PRE", [text("line\r\nnext")]);
  withStyles(new Map([[fence, { whiteSpace: "pre" }]]), () => {
    const index = buildTextIndex(fence);
    assert.equal(index.segments[0].preserved, true);
    assert.equal(index.text.includes("\r\n"), true);
    // The feed alone is the back half of a grapheme, so it is not a match on its own.
    assert.deepEqual(findMatches(index, "\n", 10), []);
    assert.deepEqual(findMatches(index, "\nnext", 10), []);
    // Asked for whole it is, and so is everything that was findable before.
    assert.equal(findMatches(index, "\r\n", 10).length, 1);
    assert.equal(findMatches(index, "\r\nnext", 10).length, 1);
    assert.equal(findMatches(index, "next", 10).length, 1);
  });
  // A feed with no return in front of it is its own grapheme and still findable.
  const alone = el("PRE", [text("line\nnext")]);
  withStyles(new Map([[alone, { whiteSpace: "pre" }]]), () => {
    assert.equal(findMatches(buildTextIndex(alone), "\n", 10).length, 1);
  });
});

test("a node left out entirely still says whether the end was a boundary", () => {
  // One code unit of room and a pair next means the whole node goes. That is not a reason to call
  // the end unknown: the node is still there to be read and its first code point answers it, as in
  // the ceiling check above. Assuming otherwise threw away the last match in the index.
  const nodes = [];
  for (
    let at = 0;
    at < MAX_INDEX_CHARS - 1 - MAX_NODE_CHARS;
    at += MAX_NODE_CHARS
  ) {
    nodes.push(text("z".repeat(MAX_NODE_CHARS)));
  }
  const used = nodes.length * MAX_NODE_CHARS;
  nodes.push(text(`${"z".repeat(MAX_INDEX_CHARS - 2 - used)}Q`));
  nodes.push(text(`${String.fromCodePoint(0x1f600)}tail`));
  const index = buildTextIndex(el("DIV", [el("P", nodes)]));
  // One unit short of the ceiling, so the pair could not fit and the node was dropped whole.
  assert.equal(index.text.length, MAX_INDEX_CHARS - 1);
  assert.equal(index.truncated, true);
  // The emoji begins its own grapheme, so what the index ends on is a boundary and stays findable.
  assert.equal(findMatches(index, "Q", 10).length, 1);
  // And when the next node does join, the end is still unknown.
  const joining = [...nodes.slice(0, -1), text("́tail")];
  const joined = buildTextIndex(el("DIV", [el("P", joining)]));
  assert.deepEqual(findMatches(joined, "Q", 10), []);
});

test("a cut never falls between the halves of a pair", () => {
  // Keeping the leading half leaves a code unit that is not a character. It reads as a grapheme of
  // its own, so a match could end against it, inside what the page draws as one thing: a node
  // ending in a base and a skin tone, cut between the tone's halves, matched the base alone.
  const tone = String.fromCodePoint(0x1f3fb);
  const index = buildTextIndex(
    el("DIV", [el("P", [text("z".repeat(MAX_NODE_CHARS - 2) + `a${tone}`)])]),
  );
  // One short of the ceiling, because the pair went whole rather than half.
  assert.equal(index.text.length, MAX_NODE_CHARS - 1);
  assert.equal(/[\ud800-\udbff]$/.test(index.text), false);
  // And the base is not findable on its own, since the page joins it to the tone that was dropped.
  assert.deepEqual(findMatches(index, "a", 10), []);
  // A cut that does not land inside a pair still takes its whole share.
  const clean = buildTextIndex(
    el("DIV", [el("P", [text("z".repeat(MAX_NODE_CHARS + 10))])]),
  );
  assert.equal(clean.text.length, MAX_NODE_CHARS);
});

test("asking whether anything is in doubt does not count what", () => {
  // Every candidate asks, and a cut through an odd indicator run leaves one run per pair, so
  // counting them made the search quadratic. Compared against the same text uncut rather than
  // against a clock: only the number of runs differs, so the ratio says whether it is walked.
  const flag = (at: number) => String.fromCodePoint(0x1f1e6 + (at % 26));
  let run = "";
  for (let at = 0; at < MAX_NODE_CHARS / 2; at += 1) run += flag(at + 3);
  const build = (cut: boolean) => {
    const head = "a".repeat(MAX_NODE_CHARS - 2);
    const nodes = [text(cut ? head + flag(0) + flag(1) + flag(2) : head)];
    for (let at = 0; at < 12; at += 1) nodes.push(text(run));
    return buildTextIndex(el("DIV", [el("P", nodes)]));
  };
  const measure = (index: ReturnType<typeof buildTextIndex>) => {
    const started = performance.now();
    findMatches(index, flag(4) + flag(5), MAX_MATCHES);
    return performance.now() - started;
  };
  const cut = build(true);
  const uncut = build(false);
  // The shape really does make a run per pair, or the comparison is measuring nothing.
  assert.equal(cut.unsafe.runs > 100_000, true);
  assert.equal(uncut.unsafe.runs, 0);
  // Warm both, so the first one through does not carry the compile for the other.
  measure(cut);
  measure(uncut);
  const ratio = measure(cut) / Math.max(measure(uncut), 1);
  // A hundredfold when the runs are counted, and about one when they are not.
  assert.equal(
    ratio < 10,
    true,
    `cut costs ${ratio.toFixed(1)}x the same text uncut`,
  );
});

test("doubt over a whole node is one run, not an entry per character", () => {
  // The per-node bound stops one sibling filling memory; it does not stop forty of them filling it
  // to the index ceiling. Held as offsets that was millions of entries for a document whose length
  // nothing on our side chose, so contiguous doubt is one run and the count is what is asserted.
  const nodes = [];
  for (let at = 0; at < MAX_INDEX_CHARS; at += MAX_NODE_CHARS) {
    nodes.push(text("́".repeat(MAX_NODE_CHARS + 1)));
  }
  const index = buildTextIndex(el("DIV", [el("P", nodes)]));
  assert.equal(index.text.length, MAX_INDEX_CHARS);
  // Millions of offsets in doubt, and it takes one run per node to say so.
  assert.equal(index.unsafe.size > MAX_INDEX_CHARS / 2, true);
  assert.equal(index.unsafe.runs < 100, true);
});

test("a chain a cut leaves ends where the block does", () => {
  // Carrying the chain between siblings has to stop where the next content becomes independent,
  // and a block closing does that. Left running, it marked a consonant past the block that nothing
  // joins, losing an occurrence. Text after the block, not inside another one: a following block
  // clears the chain as it opens, so only this shape reaches the closing boundary still live.
  const index = buildTextIndex(
    el("DIV", [
      el("P", [text("a".repeat(MAX_NODE_CHARS - 1) + "क्"), text("́")]),
      text("त rest"),
    ]),
  );
  assert.equal(findMatches(index, "त", 10).length, 1);
  // Inside the one block it is still carried, or the clear has taken the fix with it.
  const inline = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS - 1) + "क्"),
        text("́"),
        text("त"),
      ]),
    ]),
  );
  assert.deepEqual(findMatches(inline, "त", 10), []);
});

test("a chain a cut leaves crosses as many siblings as it needs", () => {
  // The doubt was dropped at the edge of the first sibling, so a chain that reached its target in
  // the sibling after that lost it. One extender is enough when it is a node of its own, which is
  // what markup made of nested inline spans produces all the time.
  const index = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS - 1) + "क्"),
        text("́"),
        text("त"),
      ]),
    ]),
  );
  assert.deepEqual(findMatches(index, "त", 10), []);
  // And the chain still ends: what follows the letter it joins is ordinary text again.
  const beyond = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS - 1) + "क्"),
        text("́"),
        text("त tail"),
      ]),
    ]),
  );
  assert.equal(findMatches(beyond, "tail", 10).length, 1);
});

test("a mark on a space in the query survives the space flexing", () => {
  // Whitespace in a query matches any run of it, a soft-wrapped paragraph rendering as one line
  // while its node holds the newline. A mark on that space is part of the same grapheme and went
  // with it, so the match ended mid-grapheme and the fence discarded text that is on the page.
  const index = buildTextIndex(el("DIV", [el("P", [text(" ́")])]));
  assert.equal(findMatches(index, " ́", 10).length, 1);
  // The flexing itself is untouched: a plain space still matches a run of whitespace.
  const wrapped = buildTextIndex(el("DIV", [el("P", [text("one \n two")])]));
  assert.equal(findMatches(wrapped, "one two", 10).length, 1);
});

test("the work a cut costs is bounded by the index, not by the node", () => {
  // A node is as long as whatever produced it chose, and only MAX_NODE_CHARS of one is indexed.
  // Following the chain over the whole of it bought an entry per code point of text that is not
  // indexed at all, so megabytes of one mark from a model cost seconds and hundreds of megabytes.
  const build = (run: number) =>
    buildTextIndex(
      el("DIV", [
        el("P", [
          text("a".repeat(MAX_NODE_CHARS - 1) + "क्"),
          text("́".repeat(run)),
        ]),
      ]),
    );
  const small = build(MAX_NODE_CHARS * 2);
  const large = build(MAX_NODE_CHARS * 20);
  // Same index either way, so the same amount recorded about it.
  assert.equal(small.text.length, large.text.length);
  assert.equal(small.unsafe.size, large.unsafe.size);
  assert.equal(large.unsafe.size <= MAX_NODE_CHARS + 2, true);
});

test("the chain a cut leaves is followed however long it runs", () => {
  // The backward window is the wrong bound going forward: the text ahead is in hand, so the run
  // may be as long as it likes. A sibling of 32 marks then the consonant put that consonant one
  // past the window and left it findable, inside the grapheme the cut split.
  for (const marks of [8, 31, 32, 64]) {
    const index = buildTextIndex(
      el("DIV", [
        el("P", [
          text("a".repeat(MAX_NODE_CHARS - 1) + "क्"),
          text("́".repeat(marks) + "ष"),
        ]),
      ]),
    );
    assert.deepEqual(findMatches(index, "ष", 10), [], `${marks} marks`);
  }
  // Past the end of the chain the text is ordinary again, or the fence has run away with itself.
  const beyond = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS - 1) + "क्"),
        text("́".repeat(64) + "ष tail"),
      ]),
    ]),
  );
  assert.equal(findMatches(beyond, "tail", 10).length, 1);
  // And what ends the chain is only in doubt where a rule could still take it. Past the window the
  // anchor is unknown, so this is the same question, and answering it yes for everything put plain
  // Latin text after a long run of marks out of reach.
  const latin = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS - 1) + "क्"),
        text("́".repeat(64) + "bcd"),
      ]),
    ]),
  );
  assert.equal(findMatches(latin, "b", 10).length, 1);
});

test("an unknown anchor reaches only what a rule could actually take", () => {
  // A context that outran its window leaves the anchor unknown, and that matters only where a rule
  // could still reach: GB9c wants a consonant on its right and GB11 a pictograph. Every letter was
  // treated as reachable, so a plain one after an overlong run of marks could not be found.
  const index = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS) + "\u0301".repeat(40)),
        text("bcd"),
      ]),
    ]),
  );
  assert.equal(findMatches(index, "b", 10).length, 1);
  // A consonant there is still in doubt, because that is the case the window ran out on.
  const indic = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS) + "\u0301".repeat(40)),
        text("\u0915"),
      ]),
    ]),
  );
  assert.deepEqual(findMatches(indic, "\u0915", 10), []);
});

test("a run resuming after an odd cut keeps the flags it really shows", () => {
  // A cut that drops an odd number of indicators leaves the run behind it pairing off one early,
  // so the flags on the page sit between the ones the index text would find. Calling the whole
  // resumed run unknown is safe and loses every flag in it, which on a page of them is the lot.
  const flag = (at: number) => String.fromCodePoint(0x1f1e6 + at);
  const dropped = flag(0);
  const rest = flag(1) + flag(2) + flag(3) + flag(4) + flag(5);
  const index = buildTextIndex(
    el("DIV", [
      el("P", [text("a".repeat(MAX_NODE_CHARS) + dropped), text(rest)]),
    ]),
  );
  assert.equal(index.truncated, true);
  // The page pairs the dropped indicator with the first of the run, so these two are whole flags.
  assert.equal(findMatches(index, flag(2) + flag(3), 10).length, 1);
  assert.equal(findMatches(index, flag(4) + flag(5), 10).length, 1);
  // And these are not: each straddles two flags on the page, however it reads in the index.
  assert.equal(findMatches(index, flag(1) + flag(2), 10).length, 0);
  assert.equal(findMatches(index, flag(3) + flag(4), 10).length, 0);
  // An even drop shifts nothing, so the run reads as it looks and no offset is in doubt.
  const even = buildTextIndex(
    el("DIV", [
      el("P", [
        text("a".repeat(MAX_NODE_CHARS) + flag(0) + flag(1)),
        text(rest),
      ]),
    ]),
  );
  assert.equal(findMatches(even, flag(1) + flag(2), 10).length, 1);
});

test("a run of regional indicators is measured once, not once per offset", () => {
  // Every offset in a run used to walk the whole run behind it to count parity, which is quadratic
  // and, on a log of flags, seconds of frozen tab. In its own process because the parity walk only
  // runs where there is no segmenter, which is the whole point of it being slow there.
  const probe = `
    delete Intl.Segmenter;
    const { buildTextIndex, findMatches, MAX_MATCHES } = await import(${JSON.stringify(
      new URL(
        "../src/features/find-in-page/lib/find-text-index.ts",
        import.meta.url,
      ).href,
    )});
    const flag = String.fromCodePoint(0x1f1e6);
    const index = buildTextIndex({
      nodeType: 1, tagName: "DIV", getAttribute: () => null,
      childNodes: [{ nodeType: 3, data: flag.repeat(8000) }],
    });
    const started = Date.now();
    const found = findMatches(index, flag.repeat(2), MAX_MATCHES).length;
    const cost = Date.now() - started;
    if (found !== 4000) throw new Error("found " + found + ", expected 4000");
    if (cost > 250) throw new Error("searching 8,000 indicators took " + cost + "ms");
  `;
  const run = spawnSync(
    process.execPath,
    ["--experimental-strip-types", "--input-type=module", "--eval", probe],
    { encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
});

test("a carriage return keeps the line feed after it", () => {
  // GB3. Split across a cut the two are in different nodes, so the generic control break was the
  // only rule that ran and it put a boundary inside the pair.
  const index = buildTextIndex(
    el("DIV", [
      el("P", [text(`${"x".repeat(MAX_NODE_CHARS)}\r`), text("\nerror")]),
    ]),
  );
  assert.deepEqual(findMatches(index, "\nerror", 5), []);
  assert.equal(findMatches(index, "error", 5).length, 1);
});

test("a real block boundary after a cut is still a boundary", () => {
  // The separator standing in for dropped text and the one a block writes are the same character,
  // but only the first is uncertain: a block break is one wherever the dropped text ended. The
  // dropped tail is a leading Jamo, which would reach on into the next syllable on the same line.
  const index = buildTextIndex(
    el("DIV", [
      el("SPAN", [text(`${"x".repeat(MAX_NODE_CHARS)}\u1100`)]),
      el("P", [text("가")]),
    ]),
  );
  assert.deepEqual([...index.unsafe], []);
  assert.equal(findMatches(index, "가", 10).length, 1);
});

test("a node cut exactly at the ceiling stays unsafe once past it", () => {
  const nodes = Array.from(
    { length: MAX_INDEX_CHARS / MAX_NODE_CHARS - 1 },
    () => text("x".repeat(MAX_NODE_CHARS)),
  );
  nodes.push(
    text(`${"a".repeat(MAX_NODE_CHARS)}\u0301`),
    text("never reached"),
  );
  const index = buildTextIndex(el("DIV", [el("P", nodes)]));
  assert.equal(index.text.length, MAX_INDEX_CHARS);
  assert.deepEqual([...index.unsafe], [MAX_INDEX_CHARS]);
  for (const match of findMatches(index, "aaa", MAX_MATCHES)) {
    assert.notEqual(match.end, MAX_INDEX_CHARS);
  }
});

test("an engine with no segmenter still fences a grapheme", () => {
  // Firefox shipped `Intl.Segmenter` in 125 and Vite's default target reaches back to 114, so this
  // is a supported build, not a hypothetical one. In its own process: the module remembers whether
  // the platform has a segmenter the first time it asks.
  const probe = `
    delete Intl.Segmenter;
    const { buildTextIndex, findMatches } = await import(${JSON.stringify(
      new URL(
        "../src/features/find-in-page/lib/find-text-index.ts",
        import.meta.url,
      ).href,
    )});
    const el = (tagName, childNodes) => ({
      nodeType: 1, tagName, childNodes, getAttribute: () => null,
    });
    const index = (body) =>
      buildTextIndex(el("DIV", [el("P", [{ nodeType: 3, data: body }])]));
    const fenced = [
      ["가́", "가"],
      ["가ा", "가"],
      ["각ᆨ", "각"],
      ["ᄀ가", "가"],
      ["؀가", "가"],
      ["\u{1f469}\u200d\u{1f469}", "\u{1f469}"],
      ["\u{1f1e6}\u{1f1e7}", "\u{1f1e6}"],
      // A skin tone is Extend by Emoji_Modifier, not by Grapheme_Extend: its category is Sk,
      // so the combining marks alone left it showing as a grapheme of its own.
      ["\u{1f44d}\u{1f3fb}", "\u{1f44d}"],
      ["\u{1f44d}\u{1f3fb}", "\u{1f3fb}"],
      // GB9c: a virama joins the consonant after it to the one before.
      ["क्त", "त"],
      // A ZWJ is an extender inside a conjunct, though not for the pictographic rule.
      ["\u0915\u094d\u200d\u0924", "\u0924"],
      // SpacingMark without being category Mc, so the category alone missed them.
      ["กำ", "ก"],
      ["ກຳ", "ກ"],
      ["\u{11380}\u{113d0}\u{11381}", "\u{11381}"],
      // And a real pair still holds, or the fix has taken the rule it guards with it.
      ["a\\u{1f469}", "\\udc69"],
    ];
    for (const [body, query] of fenced) {
      if (findMatches(index(body), query, 10).length !== 0) {
        throw new Error("not fenced: " + escape(body));
      }
    }
    // And still finds what is whole, or the fence has eaten the feature it protects.
    const found = [
      ["가나다", "나"],
      ["\u{1f1e6}\u{1f1e7}\u{1f1e8}\u{1f1e9}", "\u{1f1e8}\u{1f1e9}"],
      ["가 ᆨ", "가 "],
      ["hello", "ell"],
      // A ZWJ joins only to a pictograph (GB11). Used as an Indic joiner it ends its cluster,
      // and treating every ZWJ as joining left this unfindable.
      ["a\\u200db", "a\\u200d"],
      // GB11 wants a pictograph on both sides of the ZWJ, so this one ends its cluster.
      ["a\\u200d\\u{1f600}", "\\u{1f600}"],
      // Category Mc that is NOT SpacingMark, so nothing here joins: taking the category for the
      // class fenced off a cluster the platform never makes, and lost both halves of it.
      ["\\u1000\\u102c", "\\u1000"],
      ["\\u1000\\u102c", "\\u102c"],
      // GB9c needs a consonant on both sides of the linker, not just a linker somewhere behind.
      // A virama before a full stop, before a Latin letter, or with nothing anchoring it.
      ["\\u0915\\u094d!", "!"],
      ["\\u0915\\u094da", "a"],
      ["!\\u094d\\u0915", "\\u0915"],
      // Control is not just C0 and C1. A soft hyphen and a line separator break on both sides,
      // so the mark after each is its own grapheme and both halves are findable.
      ["\\u00ad\\u0301", "\\u00ad"],
      ["\\u00ad\\u0301", "\\u0301"],
      ["\\u2028\\u0903", "\\u2028"],
      ["\\u2028\\u0903", "\\u0903"],
      // A ZWNJ is Grapheme_Extend and still ends the conjunct, which is what it is for, so the
      // consonant after one starts a grapheme of its own.
      ["\\u0915\\u094d\\u200c\\u0915", "\\u0915"],
      // An unpaired low surrogate is a character, not half of one, and reaches a page through
      // JSON and through pasted model output. Taken for half a pair it joined what came before,
      // so neither it nor its neighbour could be found.
      ["a\\udc00b", "a"],
      ["a\\udc00b", "\\udc00"],
      ["a\\udc00b", "b"],
    ];
    for (const [body, query] of found) {
      if (findMatches(index(body), query, 10).length !== 1) {
        throw new Error("not found: " + escape(body));
      }
    }
  `;
  const run = spawnSync(
    process.execPath,
    ["--experimental-strip-types", "--input-type=module", "--eval", probe],
    { encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
});

test("nothing below U+0300 can join a grapheme, which is what the fast path rests on", () => {
  // Latin prose skips the segmenter on four comparisons, and every one of them assumes no character
  // below U+0300 can extend a grapheme or be extended into one. The far side of a cut leans on the
  // same fact. Asserted over every code point rather than argued from the blocks they sit in.
  const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
  const joiners: string[] = [];
  for (let code = 1; code < 0x300; code += 1) {
    // CR, LF and the other line breaks are their own cluster and never in an index unsplit.
    if (code >= 0x0a && code <= 0x0d) continue;
    const point = String.fromCodePoint(code);
    if (
      [...segmenter.segment(`${point}a`)].length === 1 ||
      [...segmenter.segment(`a${point}`)].length === 1
    ) {
      joiners.push(code.toString(16));
    }
  }
  assert.deepEqual(joiners, []);
});

test("the end of a full index is settled by what follows, not assumed to be a cut", () => {
  // Filling MAX_INDEX_CHARS exactly and stopping is not a cut: the next node is still there to be
  // read, and its first character answers the junction the same way a clip's does. Calling the end
  // unknown regardless threw away a match that ended on it, with a space sitting right after.
  const per = MAX_NODE_CHARS;
  const nodes = MAX_INDEX_CHARS / per;
  const pictograph = String.fromCodePoint(0x1f469);
  const build = (tail: string, next: string) => {
    const children = [];
    for (let at = 0; at < nodes; at += 1) {
      children.push(
        text(
          at === nodes - 1
            ? "x".repeat(per - tail.length) + tail
            : "x".repeat(per),
        ),
      );
    }
    children.push(text(next));
    return buildTextIndex(el("DIV", [el("P", children)]));
  };
  for (const [tail, next, query, want] of [
    ["needle", " rest", "needle", 1],
    ["needle", "rest", "needle", 1],
    ["needle", "\u0301rest", "needle", 0],
    [`a${pictograph}`, `\u200d${pictograph}`, pictograph, 0],
    ["x\u0915\u094d", "\u0937z", "\u0915\u094d", 0],
  ] as [string, string, string, number][]) {
    const index = build(tail, next);
    assert.equal(index.truncated, true);
    assert.equal(
      findMatches(index, query, 10).length,
      want,
      `${escape(tail)} then ${escape(next)}`,
    );
  }
});

test("the fallback finds neither more nor less than the platform, over a mixed corpus", () => {
  // The list above is the cases once wrong; this is the standing property, and the one that catches
  // a fallback which is safe but useless. Fencing too much never cuts a grapheme, so the
  // misalignment oracle cannot see it, yet it still loses matches: taking `Mc` for SpacingMark and
  // applying GB9c on the linker alone both did. Same corpus both ways, counts compared per body.
  const alphabet = [
    0x915, 0x937, 0x93e, 0x94d, 0x9cd, 0x995, 0x1000, 0x102c, 0x102b, 0x1038,
    0x1039, 0x1780, 0x17d2, 0x11133, 0x11103, 0x200d, 0x300, 0x903, 0xe33, 0x21,
    0x61, 0x20, 0x1f600, 0x1f1e6, 0x1100, 0x1161, 0x11a8, 0x600, 0x1f3fb,
  ];
  const probe = `
    const alphabet = ${JSON.stringify(alphabet)}.map((c) => String.fromCodePoint(c));
    if (process.env.NO_SEGMENTER === "1") delete Intl.Segmenter;
    const { buildTextIndex, findMatches } = await import(${JSON.stringify(
      new URL(
        "../src/features/find-in-page/lib/find-text-index.ts",
        import.meta.url,
      ).href,
    )});
    const el = (tagName, childNodes) => ({
      nodeType: 1, tagName, childNodes, getAttribute: () => null,
    });
    const bodies = [];
    let seed = 12345;
    const next = () => (seed = (seed * 1103515245 + 12345) >>> 0);
    for (let i = 0; i < 1500; i += 1) {
      let body = "";
      for (let k = 0, n = 2 + (next() % 6); k < n; k += 1) {
        body += alphabet[next() % alphabet.length];
      }
      bodies.push(body);
    }
    // Every ordered pair as well, since the rules meet two characters at a time.
    for (const a of alphabet) for (const b of alphabet) bodies.push(a + b);
    const counts = bodies.map((body) => {
      const index = buildTextIndex(el("DIV", [el("P", [{ nodeType: 3, data: body }])]));
      let found = 0;
      for (const query of alphabet) found += findMatches(index, query, 5000).length;
      return found;
    });
    console.log(JSON.stringify(counts));
  `;
  const run = (noSegmenter: boolean) => {
    const out = spawnSync(
      process.execPath,
      ["--experimental-strip-types", "--input-type=module", "--eval", probe],
      {
        encoding: "utf8",
        env: { ...process.env, NO_SEGMENTER: noSegmenter ? "1" : "0" },
      },
    );
    assert.equal(out.status, 0, out.stderr);
    return JSON.parse(out.stdout) as number[];
  };
  const platform = run(false);
  const fallback = run(true);
  assert.ok(platform.length > 800);
  assert.ok(platform.reduce((a, b) => a + b, 0) > 0);
  // Per body, so a shortfall on one shape cannot be paid for by a surplus on another.
  assert.deepEqual(
    fallback.flatMap((n, i) =>
      n === platform[i] ? [] : [`${i}: ${n} vs ${platform[i]}`],
    ),
    [],
  );
});

test("the segmenter fallback never misaligns, checked against the platform", () => {
  // The property behind the list above, held against `Intl.Segmenter` over every shape that has
  // caused trouble: no match the fallback returns may begin or end where the platform would not
  // break. It may find less than the platform, never cut a grapheme. This caught the Prepend set.
  const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
  const at = (code: number) => String.fromCodePoint(code);
  const pieces = [
    "a",
    "z",
    "가",
    "각",
    "ᄀ",
    "ᅡ",
    "ᆨ",
    "é",
    "i̇",
    "क्ष",
    "กั",
    "ൎക",
    "؀",
    at(0x1193f),
    at(0x1f469),
    at(0x1f44d),
    at(0x1f3fb),
    "‍",
    "‌",
    "️",
    at(0x1f1e6),
    at(0x1f1e7),
    " ",
    "́",
    "ा",
  ];
  // A fixed sequence, so a failure is the same failure tomorrow.
  let seed = 20200;
  let body = "";
  for (let i = 0; i < 2500; i += 1) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    body += pieces[seed % pieces.length];
  }
  const marks = new Uint8Array(body.length + 1);
  for (const { index } of segmenter.segment(body)) marks[index] = 1;
  marks[body.length] = 1;
  const clusters = [...segmenter.segment(body)].map((piece) => piece.segment);
  const queries = [...new Set(clusters)].slice(0, 80);

  const probe = `
    delete Intl.Segmenter;
    const { readFileSync } = await import("node:fs");
    const { buildTextIndex, findMatches } = await import(${JSON.stringify(
      new URL(
        "../src/features/find-in-page/lib/find-text-index.ts",
        import.meta.url,
      ).href,
    )});
    const { body, marks, queries } = JSON.parse(readFileSync(0, "utf8"));
    const el = (tagName, childNodes) => ({
      nodeType: 1, tagName, childNodes, getAttribute: () => null,
    });
    const index = buildTextIndex(el("DIV", [el("P", [{ nodeType: 3, data: body }])]));
    let matches = 0;
    for (const query of queries) {
      for (const hit of findMatches(index, query, 5000)) {
        matches += 1;
        if (marks[hit.start] !== 1 || marks[hit.end] !== 1) {
          throw new Error(
            "misaligned " + escape(query) + " at " + hit.start + ".." + hit.end,
          );
        }
      }
    }
    if (matches < 1000) throw new Error("only " + matches + " matches, corpus is not exercising it");
  `;
  const run = spawnSync(
    process.execPath,
    ["--experimental-strip-types", "--input-type=module", "--eval", probe],
    {
      encoding: "utf8",
      input: JSON.stringify({ body, marks: Array.from(marks), queries }),
    },
  );
  assert.equal(run.status, 0, run.stderr);
});

test("a match with no geometry is aimed at through its nearest laid-out ancestor", async () => {
  const dom = await readFile(
    new URL("../src/features/find-in-page/lib/find-dom.ts", import.meta.url),
    "utf8",
  );
  // Such text has a collapsed rect while the subtree's own box keeps its placeholder geometry.
  assert.match(
    dom,
    /export function revealRect\(range: Range\): DOMRect \| null/,
  );
  assert.match(dom, /export function rangeTop\(range: Range\): number \| null/);
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
  // The viewport starts below the navbar, so a match clipped off its top still has positive `top`.
  assert.match(engine, /top >= scrollViewportTop\(range\)/);
  assert.equal(/top >= 0/.test(engine), false);
});

test("a pending query clears the previous highlight before the next paint", async () => {
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // The input value is committed during the event. A passive effect may run only after the browser
  // has painted that new value beside the old query's ranges, which is the visible `sta`/`stan`
  // mismatch this guards. A layout effect clears those ranges in the same commit, before paint.
  assert.match(engine, /import \{[^}]*useLayoutEffect[^}]*\} from "react";/);
  assert.match(
    engine,
    /useLayoutEffect\(\(\) => \{\s*if \(queryPending\) \{\s*cancelRevealPasses\(\);\s*clearHighlights\(\);/,
  );
});

test("re-indexing while the document changes is a throttle, and says so", async () => {
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // A debounce would freeze the count for as long as a reply takes to write.
  assert.match(engine, /REINDEX_INTERVAL_MS/);
  assert.equal(/REINDEX_DEBOUNCE_MS/.test(engine), false);
  assert.match(engine, /A throttle rather than a debounce/);
});

test("the bar has no border, and its buttons have a hover that shows", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-bar.tsx",
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
  // The ghost variant's own `--muted/50` hover lands within a shade of this surface.
  assert.match(bar, /hover:bg-black\/\[0\.06\] dark:hover:bg-white\/10/);
  assert.equal((bar.match(/className=\{FIND_BUTTON_CLASS\}/g) ?? []).length, 3);
});

test("a long query rewinds to its first character when focus leaves", async () => {
  // Typing past the width of the field scrolls it, and a bar left showing the tail of a word says
  // nothing about what was searched for.
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(bar, /onBlur=\{rewindToStart\}/);
  assert.match(bar, /input\.setSelectionRange\(0, 0\);/);
  assert.match(bar, /input\.scrollLeft = 0;/);
  assert.match(bar, /onMouseDown=\{keepFocusInField\}/);
});

test("the observer watches the attributes a workspace switch flips", async () => {
  // Switching between kept-alive workspaces flips `inert` rather than mutating children.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(engine, /attributeFilter: \[[^\]]*"inert"/);
  // `open` too: it is all a `<details>` changes, while its body goes from visible to not.
  assert.match(engine, /attributeFilter: \[[^\]]*"open"/);
  // Not the whole stream: `class` changes on every hover. Scanned, not matched, since the
  // comments in between make a regex backtrack badly.
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
    else view.document = undefined;
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

test("explicit monitor portals are searched but are not dismissible surfaces", async () => {
  const monitor = surface(null);
  const popover = surface("open");
  const scope = { contains: () => false } as unknown as Element;
  const view = globalThis as { document?: unknown };
  const had = "document" in view;
  const saved = view.document;
  view.document = {
    querySelectorAll: (selector: string) =>
      selector.includes("data-find-portal") ? [monitor, popover] : [popover],
  };
  try {
    assert.deepEqual(resolvePortalSurfaces(scope), [monitor, popover]);
  } finally {
    if (had) view.document = saved;
    else view.document = undefined;
  }

  const dom = await readFile(
    new URL("../src/features/find-in-page/lib/find-dom.ts", import.meta.url),
    "utf8",
  );
  assert.match(dom, /export function resolveDismissiblePortalSurfaces/);

  for (const path of [
    "../src/features/api-monitor/api-monitor-overlay.tsx",
    "../src/components/floating-monitor.tsx",
  ]) {
    const component = await readFile(new URL(path, import.meta.url), "utf8");
    assert.match(component, /\[FIND_PORTAL_ATTRIBUTE\]: ""/);
  }
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
  // `data-state` is all a dismissed one changes, and it keeps its box until the animation ends.
  assert.match(engine, /attributeFilter: \[[^\]]*"data-state"/);
});

test("the rows progressive completion adds are re-anchored, not renumbered", async () => {
  // Progressive completion PREPENDS, and match 3 of the tail is not match 3 of the thread.
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
  // `reindex` answers false when nothing came in, so a settled thread's probe takes back no Enter.
  assert.equal(engine.includes("rebuild("), false);
});

test("Escape closes the bar from the walk buttons, not just the field", async () => {
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-bar.tsx",
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
  assert.match(effect, /window\.addEventListener\("keydown", onEscape, true\)/);
  assert.match(
    effect,
    /window\.removeEventListener\("keydown", onEscape, true\)/,
  );
  // A modal above the bar owns Escape, and an open popover is dismissed by its own first.
  assert.match(body, /isSurfaceBackgrounded\(/);
  assert.match(body, /resolveDismissiblePortalSurfaces\(/);
  // And nothing left on the landmark, which would take the inside presses before the window does.
  const landmark = bar.slice(bar.indexOf('role="search"'));
  assert.equal(
    landmark.slice(0, landmark.indexOf(">")).includes("onKeyDown"),
    false,
  );
});

test("only threads this search can read are forced to finish mounting", async () => {
  // Completing globally would make a retained conversation mount every row it withheld.
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
  // The exit has to read the filtered set, or a declined completer holds it open forever.
  assert.match(
    progressive,
    /if \(wanted\(\)\.length === 0 && \(observed \|\| Date\.now\(\) >= deadline\)\)/,
  );
});

test("the chord is left to the browser when the scope is behind a modal", async () => {
  // `useShortcut` prevents the event BEFORE the handler, so declining inside it kills the chord.
  const controller = await readComponentSource();
  assert.match(controller, /claims: \(\) => !isSurfaceBackgrounded\(/);
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
      "../src/features/find-in-page/components/find-bar.tsx",
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
      "../src/features/find-in-page/components/find-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Captured above the effect that focuses the field, or it reads the field it is about to fill.
  const capture = bar.indexOf("const active = document.activeElement");
  const takeFocus = bar.indexOf("input.select();");
  assert.ok(capture > 0 && capture < takeFocus);
  assert.match(bar, /origin\.focus\(\);/);
  // First answer only: StrictMode replays the effect, and by the second run the field has focus.
  assert.match(bar, /originRef\.current === null &&/);
  // Against the bar's element, not `data-find-skip`, which the composer carries.
  assert.match(bar, /barRef\.current\?\.contains\(active\) !== true/);
  assert.equal(bar.includes("closest(`[${FIND_SKIP_ATTRIBUTE}]`)"), false);
  assert.match(
    bar,
    /if \(focused !== null && focused !== document\.body\) return;/,
  );
});

test("the chat composer is out of the searchable scope", async () => {
  // Its draft lives in a textarea the index cannot read, leaving find only the pill labels.
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
  assert.match(
    engine,
    /activeStartRef\.current = active >= 0 \? matches\[active\]\.start : null;/,
  );
  // Read BEFORE the new list is installed, or it is the new list's answer being read back.
  const read = engine.indexOf("const wasAt = activeStartRef.current;");
  const install = engine.indexOf("matchesRef.current = matches;", read - 400);
  assert.ok(read > 0 && read < install);
  assert.match(engine, /ordinalOfStart\(matches, wasAt\)/);
  assert.match(
    engine,
    /at === -1 \? firstMatchFromViewport\(index, matches\) : at/,
  );
});

test("the ordinal survives an append and nothing else", async () => {
  // A streaming reply only adds at the tail; history above, `inert` flipping or a breakpoint
  // revealing a column each renumber the list.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // One rule, and it lives in the pure module so the tests below can exercise it directly.
  assert.match(
    engine,
    /return renumbersMatches\(before, indexRef\.current, activeStartRef\.current\);/,
  );
  assert.equal(engine.includes("search(false, false)"), false);
  assert.equal((engine.match(/search\(false, reindex\(\)\)/g) ?? []).length, 2);
  assert.match(
    engine,
    /reindex\(\);\n\s*\/\/[^\n]*\n\s*search\(false, true\);/,
  );
});

test("every navigation waits for the query to settle, buttons included", async () => {
  // The buttons stay enabled on the previous query's count while an edit is pending, and `apply`
  // will not paint until it settles, so calling `next`/`previous` from a click dropped it silently.
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(bar, /onClick=\{\(\) => stepWhenSettled\(-1\)\}/);
  assert.match(bar, /onClick=\{\(\) => stepWhenSettled\(1\)\}/);
  // No control reaches the raw pair, so a new one cannot copy the losing spelling.
  assert.equal(/onClick=\{(next|previous)\}/.test(bar), false);
  // Reaching the handler at all: a settled zero must not disable the walk while the next query is
  // still pending, or editing "no matches" into a match loses the press to a disabled button.
  assert.match(
    bar,
    /const canStep = searching && \(count > 0 \|\| queryPending\);/,
  );
  assert.equal((bar.match(/disabled=\{!canStep\}/g) ?? []).length, 2);
  assert.equal(bar.includes("disabled={count === 0}"), false);
  // The helper queues the step and forces the settle rather than dropping it.
  assert.match(
    bar,
    /if \(queryPending\) \{\s*queuedStepsRef\.current\.push\(\{ query, delta \}\);[\s\S]*?settleQuery\(\);/,
  );

  assert.match(bar, /filter\([\s\S]*?\(step\) => step\.query === query/);

  assert.match(bar, /queuedStepsRef\.current = \[\.\.\.pendingSteps\]/);
  assert.match(bar, /for \(const step of steps\)/);

  assert.match(bar, /if \(count > 0\) \{\s*for \(const step of steps\)/);
});

test("the seam between the workspace and the surfaces in front of it is recorded", () => {
  const workspace = el("DIV", [el("P", [text("unsloth studio")])]);
  const monitor = el("DIV", [el("P", [text("cpu 12%")])]);
  const alone = buildTextIndex(workspace);
  assert.equal(alone.rootLength, alone.text.length);
  const withMonitor = buildTextIndex(workspace, [monitor]);
  // The joining separator belongs to the surface, so the workspace slice is only the workspace.
  assert.equal(withMonitor.rootLength, alone.text.length);
  assert.equal(withMonitor.text.slice(0, withMonitor.rootLength), alone.text);
  assert.match(withMonitor.text.slice(withMonitor.rootLength), /cpu 12%$/);
});

test("inserting an earlier surface renumbers a reader in a later one", () => {
  const workspace = el("DIV", [el("P", [text("workspace")])]);
  const laterSurface = el("DIV", [el("P", [text("target")])]);
  const before = buildTextIndex(workspace, [laterSurface]);
  const reader = findMatches(before, "target")[0].start;

  const earlierSurface = el("DIV", [el("P", [text("target")])]);
  const after = buildTextIndex(workspace, [earlierSurface, laterSurface]);
  assert.equal(findMatches(after, "target")[0].start, reader);
  assert.equal(renumbersMatches(before, after, reader), true);
});

test("a monitor polling behind a streaming reply does not move the reader", () => {
  // The monitor is a permanent tail that rewrites itself on a timer. Judged as one string, every
  // poll and every streamed character reads as renumbered and re-anchors the reader.
  const monitorAt = (load: string) => el("DIV", [el("P", [text(load)])]);
  const reply = (body: string) => el("DIV", [el("P", [text(body)])]);

  const before = buildTextIndex(reply("unsloth one"), [monitorAt("cpu 12%")]);
  // Where the reader is: on the occurrence in the workspace.
  const readerInWorkspace = findMatches(before, "unsloth")[0].start;

  // The monitor polls; the workspace has not moved.
  const polled = buildTextIndex(reply("unsloth one"), [monitorAt("cpu 34%")]);
  assert.equal(renumbersMatches(before, polled, readerInWorkspace), false);

  // The reply streams on; the monitor has not moved.
  const streamed = buildTextIndex(reply("unsloth one unsloth two"), [
    monitorAt("cpu 12%"),
  ]);
  assert.equal(renumbersMatches(before, streamed, readerInWorkspace), false);
  // And the occurrence really is still at the offset the search looks it up by.
  assert.equal(findMatches(streamed, "unsloth")[0].start, readerInWorkspace);

  // Both at once, which is the ordinary case while a reply lands.
  const both = buildTextIndex(reply("unsloth one unsloth two"), [
    monitorAt("cpu 34%"),
  ]);
  assert.equal(renumbersMatches(before, both, readerInWorkspace), false);
});

test("a reader inside a surface gives up its place when the workspace grows under it", () => {
  // Growing the workspace moves the tail along, so an occurrence inside a surface is no longer at
  // the offset it is looked up by. That reader, and only that reader, re-anchors.
  const monitorText = { nodeType: 3 as const, data: "unsloth monitor" };
  const monitor = el("DIV", [el("P", [monitorText])]);
  const before = buildTextIndex(el("DIV", [el("P", [text("reply one")])]), [
    monitor,
  ]);
  const readerInMonitor = findMatches(before, "unsloth")[0].start;
  assert.ok(readerInMonitor >= before.rootLength);

  const grown = buildTextIndex(
    el("DIV", [el("P", [text("reply one reply two")])]),
    [monitor],
  );
  assert.equal(renumbersMatches(before, grown, readerInMonitor), true);

  // The same growth leaves a reader in the workspace exactly where they were.
  const workspaceReader = 0;
  assert.equal(renumbersMatches(before, grown, workspaceReader), false);

  // A poll that does not move the seam keeps even the surface's own reader in place.
  monitorText.data = "unsloth monitor idle";
  const polled = buildTextIndex(el("DIV", [el("P", [text("reply one")])]), [
    monitor,
  ]);
  assert.equal(renumbersMatches(before, polled, readerInMonitor), false);
});

test("a surface reader only minds the text ahead of their occurrence", () => {
  // A monitor rewrites a reading every second. Only what sits before the occurrence can move it, so
  // an occurrence in the static label survives the number below it changing.
  const reply = el("DIV", [el("P", [text("reply one")])]);
  const labelledText = {
    nodeType: 3 as const,
    data: "unsloth monitor cpu 12%",
  };
  const labelled = el("DIV", [el("P", [labelledText])]);
  const before = buildTextIndex(reply, [labelled]);
  const reader = findMatches(before, "unsloth")[0].start;
  assert.ok(reader >= before.rootLength);
  labelledText.data = "unsloth monitor cpu 345%";
  const polled = buildTextIndex(reply, [labelled]);
  assert.equal(renumbersMatches(before, polled, reader), false);
  assert.equal(findMatches(polled, "unsloth")[0].start, reader);

  // A reading AHEAD of the occurrence changing width does move it, and there the offset stops
  // naming anything: `search` finds no match starting there and re-anchors itself.
  const leadingText = { nodeType: 3 as const, data: "cpu 12% unsloth monitor" };
  const leading = el("DIV", [el("P", [leadingText])]);
  const wasLeading = buildTextIndex(reply, [leading]);
  const readerAfter = findMatches(wasLeading, "unsloth")[0].start;
  leadingText.data = "cpu 345% unsloth monitor";
  const grewLeading = buildTextIndex(reply, [leading]);
  assert.equal(
    findMatches(grewLeading, "unsloth").some((m) => m.start === readerAfter),
    false,
  );

  // Changing it without changing its width leaves the occurrence exactly where it was, which is
  // the ordinary poll and must not move the reader.
  leadingText.data = "cpu 99% unsloth monitor";
  const sameWidth = buildTextIndex(reply, [leading]);
  assert.equal(renumbersMatches(wasLeading, sameWidth, readerAfter), false);
  assert.equal(
    findMatches(sameWidth, "unsloth").some((m) => m.start === readerAfter),
    true,
  );
});

test("history arriving above the reader still renumbers the list", () => {
  // The rule this all rests on has not been widened: a prepend is not an append on either side.
  const monitorText = { nodeType: 3 as const, data: "cpu 12%" };
  const monitor = el("DIV", [el("P", [monitorText])]);
  const before = buildTextIndex(el("DIV", [el("P", [text("unsloth one")])]), [
    monitor,
  ]);
  const prepended = buildTextIndex(
    el("DIV", [el("P", [text("older unsloth one")])]),
    [monitor],
  );
  assert.equal(renumbersMatches(before, prepended, 0), true);

  // A surface whose reading is replaced rather than extended is the ordinary poll, and it must not
  // renumber anything for a reader in the conversation behind it.
  monitorText.data = "gpu 12%";
  const replaced = buildTextIndex(el("DIV", [el("P", [text("unsloth one")])]), [
    monitor,
  ]);
  assert.equal(renumbersMatches(before, replaced, 0), false);
  // Nor for a reader inside it: the rewrite kept its width, so nothing moved for them either.
  assert.equal(
    renumbersMatches(before, replaced, before.text.length - 1),
    false,
  );
  // Growing the workspace under them is what moves a surface reader, and that still renumbers.
  const grown = buildTextIndex(
    el("DIV", [el("P", [text("unsloth one unsloth two")])]),
    [monitor],
  );
  assert.equal(renumbersMatches(before, grown, before.text.length - 1), true);
});

test("a breakpoint that changes what is rendered invalidates the index", async () => {
  // Crossing one reveals whole columns with nothing in the DOM to observe.
  const engine = await readFile(
    new URL(
      "../src/features/find-in-page/hooks/use-find-in-page.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(engine, /window\.addEventListener\("resize", invalidate\);/);
  assert.match(engine, /window\.removeEventListener\("resize", invalidate\);/);
  assert.match(
    engine,
    /const invalidate = \(\) => \{[\s\S]*?REINDEX_INTERVAL_MS\);/,
  );
});

test("closing preserves the query while leaving the shell forgets the session", async () => {
  const controller = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(controller, /const \[query, setQuery\] = useState\(""\);/);
  assert.match(
    controller,
    /const close = useCallback\(\(\) => \{[\s\S]*?setOpen\(false\);[\s\S]*?\}, \[\]\);/,
  );
  assert.equal(controller.includes('setQuery("")'), false);
  assert.equal(controller.includes("useFindInPageStore"), false);
});

test("the capped window follows the reader, not the top of the document", () => {
  const body = `${"q".repeat(20_000)} needle ${"q".repeat(20_000)}`;
  const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const anchor = index.text.indexOf("needle");
  const limit = 100;

  const fromTheTop = findMatches(index, "q", limit);
  assert.equal(fromTheTop.length, limit);
  assert.equal(fromTheTop[fromTheTop.length - 1].end <= anchor, true);

  const aroundTheReader = findMatches(index, "q", limit, anchor);
  assert.equal(aroundTheReader.length, limit);
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
  // The window recentres, so the same number is a different occurrence.
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
  assert.notEqual(first[limit - 1].start, second[limit - 1].start);
  assert.equal(
    second.some((match) => match.start === first[limit - 1].start),
    true,
  );
});

test("the window is only computed when the cap bites", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text("q q q q q")])]));
  assert.deepEqual(
    findMatches(index, "q", 100, 8),
    findMatches(index, "q", 100, 0),
  );
});

test("stopping the count early does not move the window", () => {
  // A stop that came too soon would drag the window back toward the top.
  const body = "q".repeat(500);
  const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const at = 200;
  const window = findMatches(index, "q", 50, at);
  assert.equal(window.length, 50);
  assert.equal(window[0].start, at - 25);
  assert.equal(window[window.length - 1].end, at + 25);
  assert.equal(findMatches(index, "q", 500, 0).length, 500);
});

test("the window stops at the ends of the list", () => {
  const body = `needle ${"q".repeat(500)}`;
  const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const atTheTop = findMatches(index, "q", 50, 1);
  assert.equal(atTheTop[0].start, index.text.indexOf("q"));
  const atTheEnd = findMatches(index, "q", 50, index.text.length);
  assert.equal(atTheEnd.length, 50);
  assert.equal(atTheEnd[atTheEnd.length - 1].end, index.text.length);
});

test("clipped accessibility text is not searchable", () => {
  // `sr-only` keeps a real box at full opacity, which `checkVisibility` calls visible.
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
  const legacy = el("SPAN", [text("Data input")]);
  withStyles(new Map([[legacy, { clip: "rect(0px, 0px, 0px, 0px)" }]]), () => {
    assert.equal(buildTextIndex(el("DIV", [legacy])).text, "");
  });
});

test("the counter says '+' only when the cap actually cut something off", () => {
  // A count equal to the cap cannot say whether it is the total or a floor.
  const occurrences = (n: number) =>
    findMatches(
      buildTextIndex(el("DIV", [el("P", [text("a".repeat(n))])])),
      "a",
      MAX_MATCHES + 1,
    ).length;
  assert.equal(occurrences(MAX_MATCHES - 1) > MAX_MATCHES, false);
  assert.equal(occurrences(MAX_MATCHES) > MAX_MATCHES, false);
  assert.equal(occurrences(MAX_MATCHES + 1) > MAX_MATCHES, true);
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
  // Trimmed from the end the reader is further from: a window anchored near the bottom ends at
  // the document's last match.
  assert.match(
    engine,
    /if \(cappedRef\.current\) dropProbeFurthestFrom\(matches, anchoredAt\);/,
  );
  assert.match(engine, /let anchoredAt: number \| null = null;/);
  assert.match(engine, /anchoredAt = viewportOffset\(index\);/);
  const bar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-bar.tsx",
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
      "../src/features/find-in-page/components/find-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  const escapeHandler = bar.slice(bar.indexOf("const onEscape ="));
  const guard = escapeHandler.indexOf("isImeComposing(event)");
  const consume = escapeHandler.indexOf("event.preventDefault()");
  assert.ok(guard > 0 && guard < consume);
  // Safe: the global listener stands aside for a composing event before looking for a binding.
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
  // The bar paints once on opening, before any query, so clearing unconditionally there throws
  // away what the reader had selected. Boundary points, since engines differ on `getRangeAt`.
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
  // Persistently transparent text has to say so, since the index cannot tell it from a fade-in.
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
  // An affordance, not an entrance animation, so it is marked at the call site rather than by
  // turning the opacity check on.
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
  // Images is an `@container`, so pinning the sidebar crosses a breakpoint with no resize and
  // no mutation inside the scope.
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
  // The first delivery is the size the scope already had, and would cost a flatten on every open.
  assert.match(
    engine,
    /if \(!measured\) \{\s*\n\s*measured = true;\s*\n\s*return;/,
  );
});

test("nothing of the engine is mounted while the bar is closed", async () => {
  const controller = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-in-page.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(controller, /if \(!enabled \|\| !open\) return null;/);
  assert.match(
    controller,
    /lazy\(\(\) => import\("\.\/find-bar-loader\.tsx"\)\)/,
  );
  assert.equal(controller.includes("useFindInPage("), false);

  const loadedBar = await readFile(
    new URL(
      "../src/features/find-in-page/components/find-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.equal((loadedBar.match(/useFindInPage\(/g) ?? []).length, 1);
});
