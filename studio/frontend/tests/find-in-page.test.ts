// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
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

import { readSrc, readSrcAsync } from "./helpers/kit.ts";

const FIND_BAR = readSrc("features/find-in-page/components/find-bar.tsx");
const FIND_IN_PAGE = readSrc(
  "features/find-in-page/components/find-in-page.tsx",
);
const USE_FIND_IN_PAGE = readSrc(
  "features/find-in-page/hooks/use-find-in-page.ts",
);
const FIND_DOM = readSrc("features/find-in-page/lib/find-dom.ts");
const FIND_TEXT_INDEX = readSrc("features/find-in-page/lib/find-text-index.ts");
const USE_SHORTCUT = readSrc("features/settings/hooks/use-shortcut.ts");
const INDEX = readSrc("index.css");

async function readComponentSource(): Promise<string> {
  return await readSrcAsync(
    "features/find-in-page/components/find-in-page.tsx",
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
  const dottedI = String.fromCharCode(0x0130); // Turkish dotted I, whose fold is two units
  const greek = `\u039f\u0394\u039f\u03a3 ${dottedI} \u039f\u03a3`;
  const index = buildTextIndex(el("DIV", [el("P", [text(greek)])]));
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
  // `toLowerCase` picks the Greek final sigma form by position.
  const index = buildTextIndex(
    el("DIV", [el("P", [text("\u039f\u0394\u039f\u03a3 \u039f\u03a3")])]),
  );
  for (const query of [
    "\u03bf\u03c3",
    "\u03bf\u03c2",
    "\u039f\u03a3",
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
  // A literal U+00A0 in source looks like a space to every reader.
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
  const index = buildTextIndex(
    el("DIV", [el("P", [text("a")]), el("P", [text("b")])]),
  );
  assert.equal(normalizeQuery(`a${BLOCK_SEPARATOR}b`), null);
  assert.deepEqual(findMatches(index, `a${BLOCK_SEPARATOR}b`), []);
});

test("content-visibility skipping is not treated as invisibility", () => {
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
  // Chrome 105-120 / Firefox 106-121: `checkVisibility` ignores the modern option names.
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

function skipNode(options: {
  skipped?: boolean;
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
  // On WebKit and Blink the caret goes with the selection, so the field stays active
  // but swallows every keystroke.
  const fallback = FIND_DOM.slice(
    FIND_DOM.indexOf("export function selectRangeFallback"),
  );
  const body = fallback.slice(0, fallback.indexOf("\n}"));
  assert.match(body, /holdCaret\(\)/);
  assert.match(body, /releaseCaret\(/);
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
  // Extended and Old Hangul jamo have no precomposed form, so NFC and NFD spell them alike.
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
    assert.deepEqual(findMatches(index, "가"), []);
  }
});

test("the index itself is left in the form the document wrote", () => {
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
  // `checkVisibility` is undefined on WebKitGTK.
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

function documentOfMatches(count: number): FindElementLike {
  return el("DIV", [text("x-".repeat(count))]);
}

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

function cssRule(css: string, selector: string): string {
  const at = css.indexOf(`${selector} {`);
  assert.notEqual(at, -1, `${selector} is gone`);
  return css.slice(at, css.indexOf("}", at));
}

test("the stylesheet paints the two highlights the code registers", async () => {
  for (const name of [FIND_HIGHLIGHT, FIND_HIGHLIGHT_ACTIVE]) {
    assert.ok(
      INDEX.includes(`::highlight(${name})`),
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
  assert.match(FIND_BAR, new RegExp(`${FIND_SKIP_ATTRIBUTE}=`));
  assert.match(USE_FIND_IN_PAGE, /FIND_SKIP_ATTRIBUTE/);
});

test("light mode is the chatbox's background, under a slightly heavier shadow", async () => {
  const composer = cssRule(INDEX, ".unsloth-composer-surface");
  const bar = cssRule(INDEX, ".find-bar-surface");

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
  const resolved = (rule: string) => {
    const ref = /box-shadow:\s*var\((--[\w-]+)\);/.exec(rule);
    if (!ref) return rule;
    const value = new RegExp(`\\s${ref[1]}:\\s*([^;]+);`).exec(cssRule(INDEX, ":root"));
    assert.ok(value, `${ref[1]} has no light value in :root`);
    return `box-shadow: ${value[1]};`;
  };
  const from = shape(resolved(composer));
  const to = shape(resolved(bar));
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
  const value = (selector: string, property: string) => {
    const hit = new RegExp(`${property}:\\s*([^;]+);`).exec(
      cssRule(INDEX, selector),
    );
    assert.ok(hit, `${selector} has no ${property}`);
    return hit[1].trim();
  };
  const grey = (declaration: string) => {
    const hex = /#[0-9a-f]{6}/i.exec(declaration);
    assert.ok(hex, `no colour in ${declaration}`);
    return Number.parseInt(hex[0].slice(1, 3), 16);
  };
  const whiteMix = (declaration: string) => {
    const hit =
      /color-mix\(in srgb, var\(--card\), white (\d+(?:\.\d+)?)%\)/.exec(
        declaration,
      );
    assert.ok(hit, `not --card mixed toward white: ${declaration}`);
    return Number(hit[1]) / 100;
  };
  const barDeclaration = value(".dark .find-bar-surface", "background-color");
  const menuSurface =
    /\.dark :is\(\.unsloth-plus-menu, \.app-user-menu\)\.sidebar-menu\[data-slot\] \{\s*background-color:\s*([^;]+);/.exec(
      INDEX,
    );
  assert.ok(menuSurface, "the dark sidebar menu has no surface colour");
  const menuDeclaration = menuSurface[1];
  const card = grey(value(".dark", "--card-base"));
  const border = grey(value(".dark", "--border-base"));
  const lift = (mix: number) => card + (255 - card) * mix;
  const bar = lift(whiteMix(barDeclaration));
  const menu = lift(whiteMix(menuDeclaration));
  assert.ok(bar > card, `bar ${bar} is not lighter than --card ${card}`);
  assert.ok(bar < border, `bar ${bar} is not darker than --border ${border}`);
  assert.ok(
    bar > menu,
    `bar ${bar} is not lighter than the sidebar menu ${menu}`,
  );
  assert.ok(
    bar - menu <= 4,
    `bar ${bar} is more than slightly lighter than ${menu}`,
  );
  assert.match(
    value(".dark .find-bar-surface", "box-shadow"),
    /var\(--background\)/,
  );
  assert.match(
    value("html[data-contrast-adjust]", "--card"),
    /var\(--contrast-surface-mix\)/,
  );
});

test("the bar stays out of a backgrounded scope, and off the document origin", async () => {
  // Read from the backdrop: Radix never sets aria-hidden on the searched region, which holds
  // live regions (see find-backgrounded.ts).
  assert.match(FIND_BAR, /if \(isFindScopeBackgrounded\(\)\) return;/);
  const backgrounded = await readSrcAsync(
    "features/find-in-page/lib/find-backgrounded.ts",
  );
  for (const slot of ["dialog-overlay", "alert-dialog-overlay", "sheet-overlay"]) {
    assert.ok(backgrounded.includes(`"${slot}"`), `${slot} is not read as a modal`);
  }
  assert.match(backgrounded, /:not\(\[data-state="closed"\]\)/);
  assert.ok(backgrounded.includes(`'[aria-modal="true"]'`));
  assert.ok(backgrounded.includes('"[data-blocking-screen]"'));
  assert.match(
    await readSrcAsync("features/tour/components/guided-tour.tsx"),
    /<DialogPrimitive\.Overlay asChild>\s*<motion\.div[\s\S]*?data-slot="dialog-overlay"/,
  );
  assert.match(
    backgrounded,
    /isSurfaceBackgrounded\(`\[\$\{FIND_SCOPE_ATTRIBUTE\}\]`\)/,
  );
  const surface = /className="(find-bar-surface[^"]*)"/.exec(FIND_BAR);
  assert.ok(surface);
  assert.match(surface[1], /\bfixed\b/);
  assert.equal(/\babsolute\b/.test(surface[1]), false);
  assert.match(surface[1], /max-w-\[calc\(100vw-2rem\)\]/);

  // 22.25/28.25rem is the previous short-counter width: fixed input + 12rem chrome.
  assert.match(surface[1], /(?:^|\s)w-\[calc\(22\.25rem\*var\(--ui-space-scale,1\)\)\](?:\s|$)/);
  assert.match(surface[1], /(?:^|\s)sm:w-\[calc\(28\.25rem\*var\(--ui-space-scale,1\)\)\](?:\s|$)/);
  assert.match(surface[1], /(?:^|\s)data-scoped:w-\[calc\(27\.75rem\*var\(--ui-space-scale,1\)\)\](?:\s|$)/);
  assert.match(surface[1], /(?:^|\s)sm:data-scoped:w-\[calc\(33\.75rem\*var\(--ui-space-scale,1\)\)\](?:\s|$)/);
  const input = /<input[\s\S]*?className=\{?(?:cn\()?\s*"([^"]*)"/.exec(
    FIND_BAR,
  );
  assert.ok(input);
  assert.match(input[1], /\bflex-1\b/);
  assert.equal(/\bw-(?:40|64)\b/.test(input[1]), false);
});

const FIND_SURFACE_GEOMETRY =
  /className="find-bar-surface [^"]*top-\[calc\(var\(--studio-content-top-inset,0px\)\+3\.5rem\)\] [^"]*\bh-13\b/;
const TOAST_UNDER_FIND_BAR =
  /offset-top: calc\(var\(--studio-content-top-inset, 0px\) \+ 3\.5rem \+ 3\.25rem \* var\(--ui-space-scale, 1\) \+ 0\.5rem\) !important;\s*--mobile-offset-top: calc\(var\(--studio-content-top-inset, 0px\) \+ 3\.5rem \+ 3\.25rem \* var\(--ui-space-scale, 1\) \+ 0\.5rem\) !important;/;

test("toasts clear the bar while it is open", () => {
  assert.match(FIND_BAR, FIND_SURFACE_GEOMETRY);
  assert.match(FIND_IN_PAGE, FIND_SURFACE_GEOMETRY);
  const toaster = '[data-sonner-toaster][data-y-position="top"]';
  const rule = cssRule(
    INDEX,
    `:root:has([data-find-bar-layer]:not([hidden]) .find-bar-surface) ${toaster}`,
  );
  assert.match(rule, TOAST_UNDER_FIND_BAR);
  assert.match(
    readSrc("app/provider.tsx"),
    /set\("--studio-content-top-inset", usesCustomTitlebar \? "34px" : null\);/,
  );
});

test("the reveal looks again while the scroll is still moving", async () => {
  // Such a subtree has placeholder height until rendered, clamping the first scroll short.
  assert.match(
    FIND_DOM,
    /export function scrollRangeIntoView\(range: Range\): boolean/,
  );
  const reveal = FIND_DOM.slice(FIND_DOM.indexOf("function revealPass("));
  const body = reveal.slice(0, reveal.indexOf("\n}\n"));
  assert.match(
    body,
    /if \(!scrollRangeIntoView\(range\) \|\| tries <= 1\) return;/,
  );
  assert.match(body, /tries - 1/);
  assert.match(FIND_DOM, /revealRangeWhenPainted\(range: Range, tries = \d\)/);
  assert.match(body, /requestAnimationFrame\(/);
  assert.match(body, /range\.startContainer\.isConnected/);
  assert.match(USE_FIND_IN_PAGE, /revealRangeWhenPainted\(activeRange\)/);
  assert.equal(
    /scrollRangeIntoView\(activeRange\)/.test(USE_FIND_IN_PAGE),
    false,
  );
});

test("a dismissed or superseded search abandons its queued reveal passes", async () => {
  const entry = FIND_DOM.slice(
    FIND_DOM.indexOf("export function revealRangeWhenPainted"),
  );
  assert.match(
    entry.slice(0, entry.indexOf("\n}\n")),
    /cancelRevealPasses\(\)/,
  );
  const pass = FIND_DOM.slice(FIND_DOM.indexOf("function revealPass("));
  assert.match(
    pass.slice(0, pass.indexOf("\n}\n")),
    /if \(generation !== revealGeneration\) return;/,
  );
  assert.match(USE_FIND_IN_PAGE, /cancelRevealPasses\(\);/);
});

test("a query too large to compile falls back instead of throwing on the first scan", () => {
  // V8 compiles lazily, so the throw can come from the first scan, not the constructor.
  const index = buildTextIndex(el("DIV", [el("P", [text("b".repeat(60000))])]));
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
  const ga = "\uac00";
  const gag = "\uac01";
  const gagNfd = "\u1100\u1161\u11a8";
  const gagHalf = "\uac00\u11a8";
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));

  for (const body of [gag, gagNfd, gagHalf]) {
    for (const query of [gag, gagNfd, gagHalf]) {
      const hits = findMatches(index(body), query, 10);
      assert.equal(
        hits.length,
        1,
        `${escape(body)} searched for ${escape(query)}`,
      );
      assert.deepEqual(hits[0], { start: 0, end: body.length });
    }
  }

  for (const body of [gag, gagNfd, gagHalf]) {
    assert.deepEqual(findMatches(index(body), ga, 10), [], escape(body));
  }
  assert.deepEqual(findMatches(index(ga), ga, 10), [{ start: 0, end: 1 }]);
});

test("Hangul clusters keep every Jamo, and only a real syllable is fenced", () => {
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));

  const twoTrailing = "\uac00\u11a8\u11a8";
  assert.deepEqual(findMatches(index(twoTrailing), twoTrailing, 10), [
    { start: 0, end: 3 },
  ]);
  assert.deepEqual(
    findMatches(index("\uac01\u11a8"), "\u1100\u1161\u11a8\u11a8", 10),
    [{ start: 0, end: 2 }],
    "the same grapheme, spelt half-composed in the text",
  );
  assert.deepEqual(
    findMatches(index("\uac00\u11a8"), "\uac00\u11a8\u11a8", 10),
    [],
  );

  assert.deepEqual(
    findMatches(index("\uac00\u1100\u11a8"), "\uac00\u1100", 10),
    [{ start: 0, end: 2 }],
  );
});

test("a Hangul match starts and stops on grapheme boundaries", () => {
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const gag = "\uac01";
  const ga = "\uac00";
  const lead = "\u1100";

  for (const body of [`${gag}\u11a8`, `${lead}\u1161\u11a8\u11a8`]) {
    assert.deepEqual(findMatches(index(body), gag, 10), [], escape(body));
  }
  for (const body of [`${lead}${ga}`, `${lead}${lead}\u1161`]) {
    assert.deepEqual(findMatches(index(body), ga, 10), [], escape(body));
  }
  assert.deepEqual(findMatches(index(`${lead}${ga}`), `${lead}${ga}`, 10), [
    { start: 0, end: 2 },
  ]);
  assert.deepEqual(findMatches(index(`${ga} hello`), ga, 10), [
    { start: 0, end: 1 },
  ]);
});

test("an engine without lookbehind falls back rather than throwing", () => {
  // Lookbehind shipped in JavaScriptCore only in Safari 16.4; older engines throw at construction.
  const real = globalThis.RegExp;
  const refuseLookbehind = (source: string, flags?: string) => {
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
    assert.deepEqual(findMatches(index, "\uac00\ub098\ub2e4", 10), [
      { start: 0, end: 3 },
    ]);
    assert.deepEqual(findMatches(index, "hello", 10), [{ start: 4, end: 9 }]);
  } finally {
    globalThis.RegExp = real;
  }
});

test("no Hangul match ever begins or ends inside a grapheme", () => {
  // Asserts the property against `Intl.Segmenter` as the grapheme authority, not hand-picked cases.
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

  const leads = ["\u1100", "\u1101", "\ua960"];
  const vowels = ["\u1161", "\u1162", "\ud7b0"];
  const trails = ["\u11a8", "\u11a9", "\ud7cb"];
  const corpus = new Set([
    "hello \uac00",
    "\uac00 hello",
    "\uac00\ub098\ub2e4",
    "caf\u00e9",
    "cafe\u0301",
    "\uac00\u0301",
    "\uac00\u200d\ub098",
    "\u0600\uac00",
    "\u1100\uac00\ub098",
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
    assert.ok(
      findMatches(index, body, 10).length >= 1,
      `${escape(body)} cannot find itself`,
    );
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
  assert.equal(
    FIND_TEXT_INDEX.includes("(?<"),
    false,
    "no lookbehind in the pattern",
  );

  const real = globalThis.RegExp;
  const refuse = (pattern: string, flags?: string) => {
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
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  for (const [body, query] of [
    ["\uac00\u093e", "\uac00"],
    ["\uac00\u{1f3fb}", "\uac00"],
    ["\u{1193f}\uac00", "\uac00"],
    ["\uac00\u0301", "\uac00"],
    ["\u0600\uac00", "\uac00"],
    ["\uac01\u11a8", "\uac01"],
    ["\u1100\uac00", "\uac00"],
  ] as const) {
    assert.deepEqual(findMatches(index(body), query, 10), [], escape(body));
  }
  assert.deepEqual(findMatches(index("\uac00 \u11a8"), "\uac00 ", 10), [
    { start: 0, end: 2 },
  ]);
  assert.deepEqual(
    findMatches(
      index("\u{1f469}\u200d\u{1f469}\u200d\u{1f466}"),
      "\u{1f469}",
      10,
    ),
    [],
  );
});

test("an engine whose `containing` is off by one is not trusted with it", () => {
  // WebKit reads 0, 0, 1 here where Chromium and Firefox read 0, 1, 1.
  const body = `x${"\u{1f44d}"} and caf\u00e9`;
  const withReal = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const correct = findMatches(withReal, "x", 10);
  assert.deepEqual(correct, [{ start: 0, end: 1 }]);

  // A fresh process, since the segmenter and the probe are held for the module's lifetime.
  {
    const probe = spawnSync(
      process.execPath,
      [
        "--experimental-strip-types",
        "--no-warnings",
        "--input-type=module",
        "-e",
        `
        const real = Intl.Segmenter;
        Intl.Segmenter = class {
          constructor(...args) { this.inner = new real(...args); }
          segment(input) {
            const segments = this.inner.segment(input);
            const starts = [...segments].map(({ index }) => index);
            return {
              containing: (at) => {
                if (at >= input.length) return undefined;
                const previous = starts.filter((start) => start <= at);
                const own = previous[previous.length - 1];
                const earlier = previous[previous.length - 2];
                return { index: own === at && earlier !== undefined ? earlier : own };
              },
              [Symbol.iterator]: () => segments[Symbol.iterator](),
            };
          }
        };
        const { buildTextIndex, findMatches } = await import(${JSON.stringify(
          new URL(
            "../src/features/find-in-page/lib/find-text-index.ts",
            import.meta.url,
          ).href,
        )});
        const el = (tagName, childNodes) => ({
          nodeType: 1, tagName, childNodes, getAttribute: () => null,
        });
        const text = (data) => ({ nodeType: 3, data });
        const index = buildTextIndex(el("DIV", [el("P", [text(${JSON.stringify(
          body,
        )})])]));
        console.log(JSON.stringify(findMatches(index, "x", 10)));
        `,
      ],
      { encoding: "utf8" },
    );
    assert.equal(probe.status, 0, probe.stderr);
    assert.deepEqual(
      JSON.parse(probe.stdout.trim()),
      correct,
      "the same answer, by the table rather than by seeking",
    );
  }
});

test("the seek probe is asked once, on a fixture the spec settles", () => {
  assert.match(FIND_TEXT_INDEX, /let seeksBoundaries: boolean \| undefined;/);
  assert.match(
    FIND_TEXT_INDEX,
    /if \(seeksBoundaries !== undefined\) return seeksBoundaries;/,
  );
  assert.match(FIND_TEXT_INDEX, /probe\.containing\(1\)\?\.index === 1/);
  assert.equal(
    (FIND_TEXT_INDEX.match(/segmenterSeeksBoundaries\(/g) ?? []).length,
    3,
  );
});

test("a query that needs no pattern is not given one", () => {
  const body = "\u11a8".repeat(50_000);
  const index = buildTextIndex(el("DIV", [el("P", [text(body)])]));
  assert.deepEqual(findMatches(index, body, 10), [{ start: 0, end: 50_000 }]);
});

test("plain text does not pay for the boundary check", () => {
  assert.match(
    FIND_TEXT_INDEX,
    /const JOINS_GRAPHEME = \/\[\^\\u0000-\\u02ff\]\//,
  );
  const guard = FIND_TEXT_INDEX.slice(
    FIND_TEXT_INDEX.indexOf("function alignsToGraphemes"),
  );
  const before = guard.indexOf("JOINS_GRAPHEME");
  const asks = guard.indexOf("graphemeSegmenter()");
  assert.ok(before > 0 && before < asks, "the cheap test comes first");
});

test("the cheap boundary test looks at both sides of each edge", () => {
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  assert.deepEqual(findMatches(index("\u0600a"), "\u0600", 10), []);
  assert.deepEqual(findMatches(index("a\u0301"), "\u0301", 10), []);
});

test("text with no whitespace for hundreds of characters is still fenced", () => {
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  const run = "a".repeat(257);
  assert.deepEqual(findMatches(index(`${run}\uac01\u11a8`), "\uac01", 10), []);
  assert.deepEqual(findMatches(index(`${run}\uac01 x`), "\uac01", 10), [
    { start: 257, end: 258 },
  ]);
});

test("whitespace is not treated as a grapheme boundary", () => {
  const index = (body: string) =>
    buildTextIndex(el("DIV", [el("P", [text(body)])]));
  assert.deepEqual(findMatches(index("a \u0301"), "\u0301", 10), []);
  assert.deepEqual(findMatches(index("\u0600 "), "\u0600", 10), []);
});

test("a long run of regional indicators keeps its parity", () => {
  // Flags pair off by run parity and runs are unbounded, so no fixed context suffices.
  const flags = "\u{1f1e6}".repeat(129) + "\u{1f1e8}\u{1f1e9}";
  const index = buildTextIndex(el("DIV", [el("P", [text(flags)])]));
  assert.deepEqual(findMatches(index, "\u{1f1e8}\u{1f1e9}", 10), []);
});

test("a capped search does not pay twice for the same segmentation", () => {
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
  assert.ok(
    Date.now() - second <= cost + 50,
    `second search took ${Date.now() - second}ms against ${cost}ms`,
  );
});

test("a cluster is offered its longest spelling first", () => {
  const dotted = buildTextIndex(el("DIV", [el("P", [text("İstanbul")])]));
  assert.deepEqual(findMatches(dotted, "i", 10), [{ start: 0, end: 2 }]);
  assert.deepEqual(findMatches(dotted, "istanbul", 10), [{ start: 0, end: 9 }]);
});

test("a portal is its own surface, whatever the workspace ended on", () => {
  const index = buildTextIndex(
    el("SPAN", [text(`${"x".repeat(MAX_NODE_CHARS)}\u1100`)]),
    [el("SPAN", [text("\uac00 hello")])],
  );
  assert.equal(findMatches(index, "\uac00", 10).length, 1);
});

test("the time between two searches is not charged to either", () => {
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
  const probe = `
    let seeks = 0;
    let scans = 0;
    let clock = 1;
    Object.defineProperty(performance, "now", { value: () => clock });
    const Real = Intl.Segmenter;
    Intl.Segmenter = class {
      constructor(...args) { this.inner = new Real(...args); }
      segment(input) {
        const segments = this.inner.segment(input);
        return {
          containing: (at) => { seeks += 1; clock += 0.01; return segments.containing(at); },
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
    const total = 400000;
    const nodes = [];
    for (let at = 0; at < total; at += MAX_NODE_CHARS) {
      nodes.push({ nodeType: 3, data: "\uac00".repeat(MAX_NODE_CHARS) });
    }
    const index = buildTextIndex(el("DIV", [el("P", nodes)]));
    const found = findMatches(index, "\uac00", MAX_MATCHES, index.text.length);
    if (found.length !== MAX_MATCHES) throw new Error("expected a capped search, got " + found.length);
    if (scans !== 1) throw new Error("expected one boundary scan, got " + scans);
    if (seeks > 10000) throw new Error("seeks per candidate: " + seeks);
  `;
  const run = spawnSync(
    process.execPath,
    ["--experimental-strip-types", "--input-type=module", "--eval", probe],
    { encoding: "utf8" },
  );
  assert.equal(run.status, 0, run.stderr);
});

test("a CRLF is one grapheme, which the fast path has to know about", () => {
  // CR before LF is the one joining pair below U+0300 (GB3), and `<pre>` keeps it intact.
  const fence = el("PRE", [text("line\r\nnext")]);
  withStyles(new Map([[fence, { whiteSpace: "pre" }]]), () => {
    const index = buildTextIndex(fence);
    assert.equal(index.segments[0].preserved, true);
    assert.equal(index.text.includes("\r\n"), true);
    assert.deepEqual(findMatches(index, "\n", 10), []);
    assert.deepEqual(findMatches(index, "\nnext", 10), []);
    assert.equal(findMatches(index, "\r\n", 10).length, 1);
    assert.equal(findMatches(index, "\r\nnext", 10).length, 1);
    assert.equal(findMatches(index, "next", 10).length, 1);
  });
  const alone = el("PRE", [text("line\nnext")]);
  withStyles(new Map([[alone, { whiteSpace: "pre" }]]), () => {
    assert.equal(findMatches(buildTextIndex(alone), "\n", 10).length, 1);
  });
});

test("a per-node cut does not hand back half a grapheme", () => {
  const node = text(`${"z".repeat(MAX_NODE_CHARS - 1)}Q\u0301tail`);
  const index = buildTextIndex(el("DIV", [el("P", [node, text(" after")])]));
  assert.equal(index.truncated, true);
  assert.equal(index.text[MAX_NODE_CHARS - 1], "q");
  assert.deepEqual(findMatches(index, "Q", 10), []);
  assert.equal(findMatches(index, "after", 10).length, 1);
});

test("what a clip dropped cannot reach past the end of its block", () => {
  const clipped = `${"z".repeat(MAX_NODE_CHARS)}dropped\u0600`;
  const index = buildTextIndex(
    el("SPAN", [el("DIV", [text(clipped)]), text("a")]),
  );
  assert.equal(index.truncated, true);
  assert.equal(index.text.endsWith(`${BLOCK_SEPARATOR}a`), true);
  assert.equal(findMatches(index, "a", 5).length, 1);
});

test("a cut inside a run reads back to an anchor, not to a fixed window", () => {
  const ri = (n: number) => String.fromCodePoint(0x1f1e6 + n);
  const [a, b, c] = [ri(0), ri(1), ri(2)];
  const node = `x${a.repeat(49_998)}${b}${c}tail`;
  const index = buildTextIndex(
    el("DIV", [el("P", [text(node), text(" next")])]),
  );
  assert.equal(index.truncated, true);
  assert.equal(index.text.codePointAt(MAX_NODE_CHARS - 3), 0x1f1e7);
  assert.deepEqual(findMatches(index, b, 5), []);
  const plain = buildTextIndex(
    el("DIV", [
      el("P", [text(`${"z".repeat(MAX_NODE_CHARS - 1)}Q tail`), text(" next")]),
    ]),
  );
  assert.equal(findMatches(plain, "Q", 5).length, 1);
});

test("the end of a truncated index is not a boundary by default", () => {
  const kids = [];
  for (let i = 0; i < 39; i += 1) kids.push(text("z".repeat(MAX_NODE_CHARS)));
  kids.push(text(`${"z".repeat(MAX_NODE_CHARS - 1)}Q`));
  kids.push(text("\u0301rest"));
  const index = buildTextIndex(el("DIV", [el("P", kids)]));
  assert.equal(index.text.length, MAX_INDEX_CHARS);
  assert.equal(index.truncated, true);
  assert.deepEqual(findMatches(index, "Q", 10), []);
});

test("a node left out entirely still says whether the end was a boundary", () => {
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
  assert.equal(index.text.length, MAX_INDEX_CHARS - 1);
  assert.equal(index.truncated, true);
  assert.equal(findMatches(index, "Q", 10).length, 1);
  const joining = [...nodes.slice(0, -1), text("́tail")];
  const joined = buildTextIndex(el("DIV", [el("P", joining)]));
  assert.deepEqual(findMatches(joined, "Q", 10), []);
});

test("a mark on a space in the query survives the space flexing", () => {
  const index = buildTextIndex(el("DIV", [el("P", [text(" ́")])]));
  assert.equal(findMatches(index, " ́", 10).length, 1);
  const wrapped = buildTextIndex(el("DIV", [el("P", [text("one \n two")])]));
  assert.equal(findMatches(wrapped, "one two", 10).length, 1);
});

test("nothing below U+0300 can join a grapheme, which is what the fast path rests on", () => {
  // The fast path assumes nothing below U+0300 extends a grapheme; checked over every code point.
  const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
  const joiners: string[] = [];
  for (let code = 1; code < 0x300; code += 1) {
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

test("a match with no geometry is aimed at through its nearest laid-out ancestor", async () => {
  assert.match(
    FIND_DOM,
    /export function revealRect\(range: Range\): DOMRect \| null/,
  );
  assert.match(
    FIND_DOM,
    /export function rangeTop\(range: Range\): number \| null/,
  );
  assert.match(USE_FIND_IN_PAGE, /const top = rangeTop\(range\);/);
  assert.equal(
    /range\.getBoundingClientRect\(\)/.test(USE_FIND_IN_PAGE),
    false,
  );
});

test("a fresh query starts from the scroll container's top, not the window's", async () => {
  assert.match(USE_FIND_IN_PAGE, /top >= scrollViewportTop\(range\)/);
  assert.equal(/top >= 0/.test(USE_FIND_IN_PAGE), false);
});

test("a pending query clears the previous highlight before the next paint", async () => {
  // A layout effect clears stale ranges before paint; a passive effect could paint them.
  assert.match(
    USE_FIND_IN_PAGE,
    /import \{[^}]*useLayoutEffect[^}]*\} from "react";/,
  );
  assert.match(
    USE_FIND_IN_PAGE,
    /useLayoutEffect\(\(\) => \{\s*if \(queryPending\) \{\s*cancelRevealPasses\(\);\s*clearHighlights\(\);/,
  );
});

test("re-indexing while the document changes is a throttle, and says so", async () => {
  assert.match(USE_FIND_IN_PAGE, /REINDEX_INTERVAL_MS/);
  assert.equal(/REINDEX_DEBOUNCE_MS/.test(USE_FIND_IN_PAGE), false);
  assert.match(USE_FIND_IN_PAGE, /A throttle rather than a debounce/);
});

test("the bar has no border, and its buttons have a hover that shows", async () => {
  const surface = /className="(find-bar-surface[^"]*)"/.exec(FIND_BAR);
  assert.ok(surface, "the bar no longer wears the shared surface class");
  assert.equal(
    /\bborder\b/.test(surface[1]),
    false,
    "the bar took a border back",
  );
  assert.match(
    FIND_BAR,
    /hover:bg-\[rgb\(0_0_0_\/_calc\(0\.06\*var\(--contrast-wash-gain,1\)\)\)\] dark:hover:bg-\[rgb\(255_255_255_\/_calc\(0\.1\*var\(--contrast-wash-gain,1\)\)\)\]/,
  );
  assert.equal(
    (FIND_BAR.match(/className=\{FIND_BUTTON_CLASS\}/g) ?? []).length,
    3,
  );
});

test("a long query rewinds to its first character when focus leaves", async () => {
  assert.match(FIND_BAR, /onBlur=\{rewindToStart\}/);
  assert.match(FIND_BAR, /input\.setSelectionRange\(0, 0\);/);
  assert.match(FIND_BAR, /input\.scrollLeft = 0;/);
  assert.match(FIND_BAR, /onMouseDown=\{keepFocusInField\}/);
});

test("the observer watches the attributes a workspace switch flips", async () => {
  assert.match(USE_FIND_IN_PAGE, /attributeFilter: \[[^\]]*"inert"/);
  assert.match(USE_FIND_IN_PAGE, /attributeFilter: \[[^\]]*"open"/);
  // Scanned, not matched: the comments in between make a regex backtrack badly.
  const opensAttributes = USE_FIND_IN_PAGE.indexOf("attributes: true,");
  const opensFilter = USE_FIND_IN_PAGE.indexOf(
    "attributeFilter:",
    opensAttributes,
  );
  assert.ok(opensAttributes !== -1 && opensFilter > opensAttributes);
  const between = USE_FIND_IN_PAGE.slice(
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
  assert.equal(USE_FIND_IN_PAGE.includes('attributeFilter: ["class"'), false);
});

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

  assert.match(FIND_DOM, /export function resolveDismissiblePortalSurfaces/);

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
  assert.match(USE_FIND_IN_PAGE, /scope\?\.ownerDocument\?\.body \?\? scope/);
  assert.match(USE_FIND_IN_PAGE, /attributeFilter: \[[^\]]*"data-state"/);
});

test("the rows progressive completion adds are re-anchored, not renumbered", async () => {
  assert.match(
    USE_FIND_IN_PAGE,
    /completeProgressiveMounts\([\s\S]*?\.then\(\(\) => \{[\s\S]*?search\(false, reindex\(\)\);/,
  );
  assert.equal(USE_FIND_IN_PAGE.includes("rebuild("), false);
});

test("Escape closes the bar from the walk buttons, not just the field", async () => {
  // Window capture phase: a bar-local handler misses presses that started elsewhere, and an
  // unprevented Escape would reach `declineToolRequest`.
  const effect = FIND_BAR.slice(FIND_BAR.indexOf("const onEscape ="));
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
  assert.match(body, /isFindScopeBackgrounded\(\)/);
  assert.match(body, /resolveDismissiblePortalSurfaces\(/);
  const landmark = FIND_BAR.slice(FIND_BAR.indexOf('role="search"'));
  assert.equal(
    landmark.slice(0, landmark.indexOf(">")).includes("onKeyDown"),
    false,
  );
});

test("only threads this search can read are forced to finish mounting", async () => {
  assert.match(
    USE_FIND_IN_PAGE,
    /completeProgressiveMounts\(\(viewport\) =>\s*\n?\s*indexReaches\(scope, viewport\)/,
  );
  const progressive = await readSrcAsync(
    "components/assistant-ui/progressive-messages.tsx",
  );
  assert.match(
    progressive,
    /if \(wanted\(\)\.length === 0 && \(observed \|\| Date\.now\(\) >= deadline\)\)/,
  );
});

test("the chord is left to the browser when the scope is behind a modal", async () => {
  // `useShortcut` prevents the event BEFORE the handler, so declining inside it kills the chord.
  const controller = await readComponentSource();
  assert.match(controller, /claims: \(\) => !isFindScopeBackgrounded\(\)/);
  const consume = USE_SHORTCUT.indexOf("event.preventDefault();");
  assert.ok(consume > 0);
  assert.ok(
    USE_SHORTCUT.lastIndexOf("latestRef.current.claims?.()", consume) > 0,
  );
});

test("the Enter that commits an IME candidate is left alone", async () => {
  const enter = FIND_BAR.slice(FIND_BAR.indexOf('event.key === "Enter"'));
  const guard = enter.indexOf("isImeComposing(event.nativeEvent)");
  const prevent = enter.indexOf("event.preventDefault()");
  assert.ok(guard > 0 && guard < prevent);
});

test("closing the bar hands focus back to where it came from", async () => {
  const capture = FIND_BAR.indexOf("const active = document.activeElement");
  const takeFocus = FIND_BAR.indexOf("input.select();");
  assert.ok(capture > 0 && capture < takeFocus);
  assert.match(FIND_BAR, /origin\.focus\(\);/);
  // StrictMode replays the effect, and by the second run the field has focus.
  assert.match(FIND_BAR, /originRef\.current === null &&/);
  assert.match(FIND_BAR, /barRef\.current\?\.contains\(active\) !== true/);
  assert.equal(FIND_BAR.includes("closest(`[${FIND_SKIP_ATTRIBUTE}]`)"), false);
  assert.match(
    FIND_BAR,
    /if \(focused !== null && focused !== document\.body\) return;/,
  );
});

test("the chat composer is out of the searchable scope", async () => {
  const thread = await readSrcAsync("components/assistant-ui/thread.tsx");
  const root = thread.slice(thread.indexOf("<ComposerPrimitive.Root"));
  assert.match(
    root.slice(0, root.indexOf(">")),
    /\{\.\.\.\{ \[FIND_SKIP_ATTRIBUTE\]: "" \}\}/,
  );
});

test("the reader is kept on the occurrence, not on the number", async () => {
  assert.match(
    USE_FIND_IN_PAGE,
    /activeStartRef\.current = active >= 0 \? matches\[active\]\.start : null;/,
  );
  const read = USE_FIND_IN_PAGE.indexOf(
    "const wasAt = activeStartRef.current;",
  );
  const install = USE_FIND_IN_PAGE.indexOf(
    "matchesRef.current = matches;",
    read - 400,
  );
  assert.ok(read > 0 && read < install);
  assert.match(USE_FIND_IN_PAGE, /ordinalOfStart\(matches, wasAt\)/);
  assert.match(
    USE_FIND_IN_PAGE,
    /at === -1 \? firstMatchFromViewport\(index, matches\) : at/,
  );
});

test("the ordinal survives an append and nothing else", async () => {
  assert.match(
    USE_FIND_IN_PAGE,
    /return renumbersMatches\(before, indexRef\.current, activeStartRef\.current\);/,
  );
  assert.equal(USE_FIND_IN_PAGE.includes("search(false, false)"), false);
  assert.equal(
    (USE_FIND_IN_PAGE.match(/search\(false, reindex\(\)\)/g) ?? []).length,
    2,
  );
  assert.match(
    USE_FIND_IN_PAGE,
    /reindex\(\);\n\s*\/\/[^\n]*\n\s*search\(false, true\);/,
  );
});

test("every navigation waits for the query to settle, buttons included", async () => {
  assert.match(FIND_BAR, /onClick=\{\(\) => stepWhenSettled\(-1\)\}/);
  assert.match(FIND_BAR, /onClick=\{\(\) => stepWhenSettled\(1\)\}/);
  assert.equal(/onClick=\{(next|previous)\}/.test(FIND_BAR), false);
  assert.match(
    FIND_BAR,
    /const canStep = searching && \(count > 0 \|\| queryPending\);/,
  );
  assert.equal((FIND_BAR.match(/disabled=\{!canStep\}/g) ?? []).length, 2);
  assert.equal(FIND_BAR.includes("disabled={count === 0}"), false);
  assert.match(
    FIND_BAR,
    /if \(queryPending\) \{\s*queuedStepsRef\.current\.push\(\{ query, delta \}\);[\s\S]*?settleQuery\(\);/,
  );

  assert.match(FIND_BAR, /filter\([\s\S]*?\(step\) => step\.query === query/);

  assert.match(FIND_BAR, /queuedStepsRef\.current = \[\.\.\.pendingSteps\]/);
  assert.match(FIND_BAR, /for \(const step of steps\)/);

  assert.match(FIND_BAR, /if \(count > 0\) \{\s*for \(const step of steps\)/);
});

test("the seam between the workspace and the surfaces in front of it is recorded", () => {
  const workspace = el("DIV", [el("P", [text("unsloth studio")])]);
  const monitor = el("DIV", [el("P", [text("cpu 12%")])]);
  const alone = buildTextIndex(workspace);
  assert.equal(alone.rootLength, alone.text.length);
  const withMonitor = buildTextIndex(workspace, [monitor]);
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
  const monitorAt = (load: string) => el("DIV", [el("P", [text(load)])]);
  const reply = (body: string) => el("DIV", [el("P", [text(body)])]);

  const before = buildTextIndex(reply("unsloth one"), [monitorAt("cpu 12%")]);
  const readerInWorkspace = findMatches(before, "unsloth")[0].start;

  const polled = buildTextIndex(reply("unsloth one"), [monitorAt("cpu 34%")]);
  assert.equal(renumbersMatches(before, polled, readerInWorkspace), false);

  const streamed = buildTextIndex(reply("unsloth one unsloth two"), [
    monitorAt("cpu 12%"),
  ]);
  assert.equal(renumbersMatches(before, streamed, readerInWorkspace), false);
  assert.equal(findMatches(streamed, "unsloth")[0].start, readerInWorkspace);

  const both = buildTextIndex(reply("unsloth one unsloth two"), [
    monitorAt("cpu 34%"),
  ]);
  assert.equal(renumbersMatches(before, both, readerInWorkspace), false);
});

test("a reader inside a surface gives up its place when the workspace grows under it", () => {
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

  const workspaceReader = 0;
  assert.equal(renumbersMatches(before, grown, workspaceReader), false);

  monitorText.data = "unsloth monitor idle";
  const polled = buildTextIndex(el("DIV", [el("P", [text("reply one")])]), [
    monitor,
  ]);
  assert.equal(renumbersMatches(before, polled, readerInMonitor), false);
});

test("a surface reader only minds the text ahead of their occurrence", () => {
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

  leadingText.data = "cpu 99% unsloth monitor";
  const sameWidth = buildTextIndex(reply, [leading]);
  assert.equal(renumbersMatches(wasLeading, sameWidth, readerAfter), false);
  assert.equal(
    findMatches(sameWidth, "unsloth").some((m) => m.start === readerAfter),
    true,
  );
});

test("history arriving above the reader still renumbers the list", () => {
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

  monitorText.data = "gpu 12%";
  const replaced = buildTextIndex(el("DIV", [el("P", [text("unsloth one")])]), [
    monitor,
  ]);
  assert.equal(renumbersMatches(before, replaced, 0), false);
  assert.equal(
    renumbersMatches(before, replaced, before.text.length - 1),
    false,
  );
  const grown = buildTextIndex(
    el("DIV", [el("P", [text("unsloth one unsloth two")])]),
    [monitor],
  );
  assert.equal(renumbersMatches(before, grown, before.text.length - 1), true);
});

test("a breakpoint that changes what is rendered invalidates the index", async () => {
  assert.match(
    USE_FIND_IN_PAGE,
    /window\.addEventListener\("resize", invalidate\);/,
  );
  assert.match(
    USE_FIND_IN_PAGE,
    /window\.removeEventListener\("resize", invalidate\);/,
  );
  assert.match(
    USE_FIND_IN_PAGE,
    /const invalidate = \(\) => \{[\s\S]*?REINDEX_INTERVAL_MS\);/,
  );
});

test("closing preserves the query while leaving the shell forgets the session", async () => {
  assert.match(FIND_IN_PAGE, /const \[query, setQuery\] = useState\(""\);/);
  assert.match(
    FIND_IN_PAGE,
    /const close = useCallback\(\(\) => \{[\s\S]*?setOpen\(false\);[\s\S]*?\}, \[\]\);/,
  );
  assert.equal(FIND_IN_PAGE.includes('setQuery("")'), false);
  assert.equal(FIND_IN_PAGE.includes("useFindInPageStore"), false);
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
  assert.match(
    USE_FIND_IN_PAGE,
    /findMatches\(\s*\n\s*index,\s*\n\s*queryRef\.current,\s*\n\s*MAX_MATCHES \+ 1,/,
  );
  assert.match(
    USE_FIND_IN_PAGE,
    /cappedRef\.current = matches\.length > MAX_MATCHES;/,
  );
  assert.match(
    USE_FIND_IN_PAGE,
    /if \(cappedRef\.current\) dropProbeFurthestFrom\(matches, anchoredAt\);/,
  );
  assert.match(USE_FIND_IN_PAGE, /let anchoredAt: number \| null = null;/);
  assert.match(USE_FIND_IN_PAGE, /anchoredAt = viewportOffset\(index\);/);
  assert.match(FIND_BAR, /\$\{capped \? "\+" : ""\}/);
  assert.equal(FIND_BAR.includes("count >= MAX_MATCHES"), false);
});

test("Escape is left to the IME while it is composing", async () => {
  const escapeHandler = FIND_BAR.slice(FIND_BAR.indexOf("const onEscape ="));
  const guard = escapeHandler.indexOf("isImeComposing(event)");
  const consume = escapeHandler.indexOf("event.preventDefault()");
  assert.ok(guard > 0 && guard < consume);
  assert.match(
    USE_SHORTCUT,
    /if \(isImeComposing\(event\)\) return;\n\s*const hit = bindings\.find/,
  );
});

test("the selection fallback only clears what it put there", () => {
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
    // Annotated, or `deepEqual`'s assertion signature narrows `ranges` to `never[]`.
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
  const tool = await readSrcAsync(
    "components/assistant-ui/tool-ui-image-generation.tsx",
  );
  const at = tool.indexOf("sm:group-hover/generated-image:opacity-100");
  assert.notEqual(at, -1);
  assert.match(
    tool.slice(Math.max(at - 700, 0), at),
    /\{\.\.\.\{ \[FIND_SKIP_ATTRIBUTE\]: "" \}\}/,
  );
});

test("a hover-only badge is out of the index", async () => {
  const sheet = await readSrcAsync(
    "components/assistant-ui/message-response-details-sheet.tsx",
  );
  const badge = sheet.slice(sheet.indexOf("aui-response-model-badge") - 400);
  assert.match(
    badge.slice(0, badge.indexOf("aui-response-model-badge")),
    /\{\.\.\.\{ \[FIND_SKIP_ATTRIBUTE\]: "" \}\}/,
  );
  assert.match(FIND_TEXT_INDEX, /opacityProperty: false/);
});

test("a container query resizing the scope invalidates the index", async () => {
  assert.match(USE_FIND_IN_PAGE, /new ResizeObserver\(\(\) => \{/);
  assert.match(USE_FIND_IN_PAGE, /sized\.observe\(scope\);/);
  assert.match(USE_FIND_IN_PAGE, /sized\?\.disconnect\(\);/);
  assert.match(
    USE_FIND_IN_PAGE,
    /if \(!measured\) \{\s*\n\s*measured = true;\s*\n\s*return;/,
  );
});

test("nothing of the engine is mounted while the bar is closed", async () => {
  assert.match(FIND_IN_PAGE, /if \(!enabled \|\| !open\) return null;/);
  assert.match(
    FIND_IN_PAGE,
    /lazy\(\(\) => import\("\.\/find-bar-loader\.tsx"\)\)/,
  );
  assert.equal(FIND_IN_PAGE.includes("useFindInPage("), false);

  assert.equal((FIND_BAR.match(/useFindInPage\(/g) ?? []).length, 1);
});
