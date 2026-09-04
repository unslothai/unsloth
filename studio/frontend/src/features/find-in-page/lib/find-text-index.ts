// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The searchable text of a subtree, flattened into one string plus the map back to its text nodes.
// Flattening once and running `indexOf` keeps a keystroke off the DOM: a tree walk per character
// typed would re-read a 300K conversation six times to type "unsloth". The walk is paid only when
// the document changes (see use-find-in-page).
//
// Pure, and written against structural types, so it runs under `node --test` with the hand-rolled
// DOM in tests/find-in-page.test.ts. There is no DOM library in this project.

import { FIND_SKIP_ATTRIBUTE } from "./find-attributes.ts";

/** `Node.ELEMENT_NODE` / `Node.TEXT_NODE`, spelled out so this module needs no DOM globals. */
export const ELEMENT_NODE = 1;
export const TEXT_NODE = 3;

/** Written at block boundaries so a match cannot run from one paragraph into the next. NUL because
 *  no query can contain it, and `findMatches` rejects the one route that could (a paste). */
export const BLOCK_SEPARATOR = "\u0000";

/** Ceiling on the flattened text. A thread is bounded by what a person scrolled through, a tool
 *  result is not: a Bash step can paste a megabyte of build log in. Scanning 4M costs about 2ms. */
export const MAX_INDEX_CHARS = 4_000_000;

/** Ceiling on matches for one query. Every match is a `Range` the highlight registry has to paint,
 *  and a single letter in a long thread has thousands. */
export const MAX_MATCHES = 5_000;

/** Ceiling on what ONE text node may contribute. A Bash step's log arrives as a single text node,
 *  and left to itself it spends the whole budget on the top of the document, so the messages the
 *  reader is looking at are not indexed at all. A share each, and the walk goes on. */
export const MAX_NODE_CHARS = 100_000;

/**
 * Budget held back from the workspace for the surfaces portaled in front of it.
 *
 * The workspace is walked first, because that is where it sits in the document, and a conversation
 * long enough to reach the ceiling would otherwise spend the whole budget before the popover the
 * reader is actually looking at was reached. A popover, a menu or a listbox is small; this is 2.5%
 * of the ceiling and only held back while one of them is open.
 */
export const PORTAL_RESERVE_CHARS = 100_000;

/** Elements whose subtree holds no findable text. Form controls carry theirs in `value`; a `Range`
 *  over `SVG` or `CANVAS` content is not paintable on every engine. */
const SKIP_TAGS: ReadonlySet<string> = new Set([
  "SCRIPT",
  "STYLE",
  "NOSCRIPT",
  "TEMPLATE",
  "INPUT",
  "TEXTAREA",
  "SELECT",
  "OPTION",
  "SVG",
  "CANVAS",
  "VIDEO",
  "AUDIO",
  "IFRAME",
  "OBJECT",
  "EMBED",
]);

/** Elements that end the line they sit on, so a separator goes in either side of them. A tag set,
 *  not `getComputedStyle`: this runs on every element in the thread. Being wrong about an exotic
 *  one inserts a break that was not needed, never invents a match. */
const BLOCK_TAGS: ReadonlySet<string> = new Set([
  "ADDRESS",
  "ARTICLE",
  "ASIDE",
  "BLOCKQUOTE",
  "BR",
  "BUTTON",
  "DD",
  "DETAILS",
  "DIALOG",
  "DIV",
  "DL",
  "DT",
  "FIELDSET",
  "FIGCAPTION",
  "FIGURE",
  "FOOTER",
  "FORM",
  "H1",
  "H2",
  "H3",
  "H4",
  "H5",
  "H6",
  "HEADER",
  "HR",
  "LI",
  "MAIN",
  "NAV",
  "OL",
  "P",
  "PRE",
  "SECTION",
  "SUMMARY",
  "TABLE",
  "TBODY",
  "TD",
  "TFOOT",
  "TH",
  "THEAD",
  "TR",
  "UL",
]);

// Re-exported so this module stays the one import for everything about the index.
export {
  FIND_SCOPE_ATTRIBUTE,
  FIND_SKIP_ATTRIBUTE,
} from "./find-attributes.ts";

/** Spaces that render as a space but are not one. Each is one UTF-16 unit, which keeps the map valid. */
const HARD_SPACE_PATTERN = /[\u00A0\u2002\u2003\u2007\u2009\u202F]/g;

/** Only the shape this module reads, so a test can hand it plain objects. */
export interface FindTextNodeLike {
  readonly nodeType: number;
  readonly data: string;
}

export interface FindElementLike {
  readonly nodeType: number;
  readonly tagName: string;
  readonly childNodes: ArrayLike<FindTextNodeLike | FindElementLike>;
  getAttribute(name: string): string | null;
  /** Optional so a test can hand this plain objects; every browser we ship on has it. */
  checkVisibility?(options?: {
    contentVisibilityAuto?: boolean;
    opacityProperty?: boolean;
    visibilityProperty?: boolean;
    /** Historic spellings of the two above, still the only ones older engines read. */
    checkOpacity?: boolean;
    checkVisibilityCSS?: boolean;
  }): boolean;
}

export type FindNodeLike = FindTextNodeLike | FindElementLike;

/** One text node's contribution, at the offset its first character took in `text`. */
export interface TextSegment {
  node: FindTextNodeLike;
  start: number;
  /** Always the node's own `data.length`, so an offset inside the run maps straight through. */
  length: number;
  /** True inside a `<pre>` or anything else that keeps its whitespace. See `findMatches`. */
  preserved: boolean;
}

export interface FindTextIndex {
  /** Case-folded haystack, block boundaries written as `BLOCK_SEPARATOR`. */
  text: string;
  /** Sorted by `start`, gapped wherever a separator was written. */
  segments: TextSegment[];
  /** True when `MAX_INDEX_CHARS` stopped the walk early. */
  truncated: boolean;
}

export const EMPTY_TEXT_INDEX: FindTextIndex = {
  text: "",
  segments: [],
  truncated: false,
};

/** Dotted I, the only code point whose `toLowerCase` grows (scanned all of them). Mapped to bare
 *  `i`, the Turkic fold: the default adds a combining dot, and one index char must stand for one
 *  document char. */
const DOTTED_I_PATTERN = /\u0130/g;

/** Final sigma, mapped to medial as `CaseFolding.txt` does. `toLowerCase` picks between them by
 *  what follows, so only one of the two spellings a reader can type would match. One unit each. */
const FINAL_SIGMA_PATTERN = /\u03c2/g;

/**
 * Case-fold the flattened document without changing its length.
 *
 * The two mappings above are what the default fold misses. With dotted I gone first `toLowerCase`
 * cannot change a length, so this is one pass: 11ms at the 4M ceiling, against 146ms walking code
 * points.
 */
export function foldText(raw: string): string {
  const spaced = raw
    .replace(HARD_SPACE_PATTERN, " ")
    .replace(DOTTED_I_PATTERN, "i");
  const folded = spaced.toLowerCase();
  if (folded.length === spaced.length) {
    return folded.replace(FINAL_SIGMA_PATTERN, "\u03c3");
  }
  // Nothing reaches this: dotted I was the only expansion and it is gone by here. But a wrong
  // length would misplace every offset after it, so fall back to the fold that cannot drift.
  let plain = "";
  for (const point of spaced) {
    const lower = point.toLowerCase();
    plain += lower.length === point.length ? lower : point;
  }
  return plain.replace(FINAL_SIGMA_PATTERN, "\u03c3");
}

/**
 * The half of the skip rule that costs no layout: tag name and attributes.
 *
 * Asked first, and by the walk directly, so a subtree turned down on markup alone never resolves a
 * style at all. That is most of what gets skipped: SCRIPT and STYLE, the bar itself, the off-route
 * workspaces the shell parks under `inert`, and the whole page behind a modal.
 */
function skipsByMarkup(element: FindElementLike): boolean {
  // Uppercased first: only HTML elements report their tag that way. SVG and MathML keep their source
  // casing, so an inline `<svg>` answers "svg" and walked straight past this set, putting Mermaid
  // labels in the index as matches no engine can paint. Already uppercase for everything else.
  if (SKIP_TAGS.has(element.tagName.toUpperCase())) return true;
  if (element.getAttribute(FIND_SKIP_ATTRIBUTE) !== null) return true;
  // Boolean attributes, so presence is the whole signal. The shell parks an off-route workspace
  // under `inert`; Radix marks the page `aria-hidden` behind a modal.
  if (element.getAttribute("hidden") !== null) return true;
  if (element.getAttribute("inert") !== null) return true;
  return element.getAttribute("aria-hidden") === "true";
}

/**
 * True when the walk must not descend into this element.
 *
 * `style` is the element's resolved style, passed in by the walk so the read is made once per
 * element rather than once here and once again in `visit` -- which is what the measurement quoted
 * on `computedStyle` was taken to mean, and was not what the code did. Optional because the
 * exported signature is one the tests call directly; resolving it here is the same answer.
 */
export function skipsSubtree(
  element: FindElementLike,
  style: ResolvedStyle | null = computedStyle(element),
): boolean {
  if (skipsByMarkup(element)) return true;
  // Anything the engine is not painting. Attributes miss the common case: `hidden lg:flex` is a
  // CLASS, and text under it would be counted and walked to while nobody can see it.
  //
  // `contentVisibilityAuto` off: such a subtree is skipped, not hidden, and asking would drop the
  // far half of a Hub README and of a maths thread, with nothing to put it back (scrolling renders
  // without mutating, so the observer never fires). Opacity off, so a message fading in stays
  // findable.
  //
  // Both spellings of each option: `visibilityProperty`/`opacityProperty` are renames of
  // `checkVisibilityCSS`/`checkOpacity`, an engine reads only the name it knows, and Web IDL drops
  // an unknown member silently. The modern name alone is a no-op on Chrome 105-120 and Firefox
  // 106-121, which would then index and highlight `visibility: hidden` text.
  const painted = element.checkVisibility?.({
    contentVisibilityAuto: false,
    opacityProperty: false,
    checkOpacity: false,
    visibilityProperty: true,
    checkVisibilityCSS: true,
  });
  if (painted === false) {
    // `display: contents` has no box, and no box is the first thing `checkVisibility` calls
    // invisible, so a wrapper whose children are all on screen answers false. The shell and the
    // training page both use one, which is most of what there is to search. A real box is what
    // makes an element hidden rather than absent.
    return style?.display !== "contents";
  }
  // No `checkVisibility` to ask: it landed in Safari 17.4, and WebKitGTK is a supported engine
  // here, so this is a real path and the visible branch would index every `display: none` subtree.
  // `paintsNothing` mirrors what the call above asks for.
  if (painted === undefined && paintsNothing(style)) return true;
  return clippedAway(style);
}

interface ResolvedStyle {
  display?: string;
  visibility?: string;
  whiteSpace?: string;
  clip?: string;
  clipPath?: string;
}

/**
 * True when this element keeps its own text off screen while still being descended into.
 *
 * Only `display: contents` reaches this, and `skipsSubtree` lets it through on purpose. But
 * `visibility` inherits, and only ELEMENT children are re-checked, so a direct text child of a
 * hidden `contents` wrapper would be indexed and highlighted. Scoped to the element's own text:
 * a descendant restoring `visibility: visible` is painted again and still reached.
 */
function hidesOwnText(style: ResolvedStyle | null): boolean {
  return (
    style?.display === "contents" &&
    (style.visibility === "hidden" || style.visibility === "collapse")
  );
}

/** True when this element paints no box, for engines with no `checkVisibility`. `display: contents`
 *  is boxless rather than hidden, so it is excluded here and `hidesOwnText` covers its own text,
 *  which is what the branch above does when the API answers. */
function paintsNothing(style: ResolvedStyle | null): boolean {
  if (style?.display === "none") return true;
  if (style?.display === "contents") return false;
  return style?.visibility === "hidden" || style?.visibility === "collapse";
}

/** The two spellings of Tailwind's `sr-only`: a real box, full opacity, clipped to nothing.
 *  `checkVisibility` calls that visible, so without this the app's 46 screen-reader labels are
 *  counted and walked to under a highlight clipped away with them. */
function clippedAway(style: ResolvedStyle | null): boolean {
  return (
    style?.clipPath === "inset(50%)" ||
    style?.clip === "rect(0px, 0px, 0px, 0px)"
  );
}

/**
 * `display` and `white-space` as resolved, or null off the DOM where the tag sets stand in.
 *
 * Used for three things: telling a boxless wrapper from a hidden one, the block boundaries no tag
 * name carries, and where whitespace is preserved. At 8000 elements: 2.5ms for the visibility check
 * alone, 4.3ms with this, against a 300ms floor between rebuilds.
 */
function computedStyle(element: FindElementLike): ResolvedStyle | null {
  const view = globalThis as unknown as {
    getComputedStyle?: (element: FindElementLike) => ResolvedStyle;
  };
  return view.getComputedStyle?.(element) ?? null;
}

/** True when this element ends the line it sits on. `inline-block` and friends do not, `contents`
 *  has no box. Being wrong inserts a needless separator, which can only lose a match and never
 *  invent one, so anything unrecognised counts as a boundary. */
function isBlockDisplay(display: string | undefined): boolean {
  if (display === undefined) return false;
  return !(
    display.startsWith("inline") ||
    display === "contents" ||
    display === "none"
  );
}

/** True where whitespace is kept rather than collapsed. `pre-line` is excluded: it still collapses
 *  runs of spaces, which is the half the query flexibility below cares about. */
function preservesWhitespace(whiteSpace: string | undefined): boolean {
  return (
    whiteSpace === "pre" ||
    whiteSpace === "pre-wrap" ||
    whiteSpace === "break-spaces"
  );
}

/**
 * Flatten `root` into one case-folded string plus the map back to its text nodes.
 *
 * Recursive rather than a `TreeWalker`: the walker reports entering an element but not leaving one,
 * and the closing separator is what stops `<p>a</p>b` reading as "ab".
 */
export function buildTextIndex(
  root: FindElementLike,
  /** Subtrees to index after `root`, in the order they sit in the document. Portaled surfaces. */
  extraRoots: readonly FindElementLike[] = [],
): FindTextIndex {
  const parts: string[] = [];
  const segments: TextSegment[] = [];
  let length = 0;
  /** The index does not hold everything: a node was clipped, or the ceiling was reached. */
  let truncated = false;
  /** The ceiling was reached, which is the only thing that stops the walk. */
  let full = false;
  /** What `full` is measured against. Raised for the portaled surfaces below, which the workspace
   *  gives up a share of the budget for, and only when there is one of them to give it to. */
  let ceiling =
    MAX_INDEX_CHARS - (extraRoots.length > 0 ? PORTAL_RESERVE_CHARS : 0);
  // Written lazily, so a run of empty blocks costs nothing and no separator lands at either end.
  let pendingSeparator = false;

  const visit = (element: FindElementLike, inherited: boolean): void => {
    // Markup first, so a subtree turned down on a tag or an attribute costs no layout read at all.
    // Then one resolved style, shared with the rest of the skip rule: both halves need it, and
    // resolving it separately in each doubled the cost the comment on `computedStyle` quotes.
    if (skipsByMarkup(element)) return;
    const style = computedStyle(element);
    if (skipsSubtree(element, style)) return;
    // The tag set answers `<br>`, whose display is inline. Layout is the rest: two `span.block`
    // run together without a boundary, so "Open" above "AI" reads as one word.
    const block =
      BLOCK_TAGS.has(element.tagName) || isBlockDisplay(style?.display);
    if (block) pendingSeparator = true;
    const preserved =
      style?.whiteSpace === undefined
        ? inherited
        : preservesWhitespace(style.whiteSpace);
    // Asked once per element, not per text node: a boxless wrapper that is also invisible paints
    // none of its own text, and that text is the one thing `skipsSubtree` never gets to judge.
    const ownTextHidden = hidesOwnText(style);
    const children = element.childNodes;
    for (let i = 0; i < children.length; i += 1) {
      if (full) return;
      const child = children[i];
      if (child.nodeType === TEXT_NODE) {
        if (ownTextHidden) continue;
        const node = child as FindTextNodeLike;
        const data = node.data;
        if (data.length === 0) continue;
        // Before the separator, not after: one emitted past the ceiling makes `take` negative, and
        // `slice(0, negative)` takes all but the last character of the next node.
        if (length >= ceiling) {
          truncated = true;
          full = true;
          return;
        }
        if (pendingSeparator) {
          pendingSeparator = false;
          if (length > 0) {
            parts.push(BLOCK_SEPARATOR);
            length += 1;
          }
        }
        // A share, not all of it: the prefix keeps one huge node findable, while the whole
        // remaining budget for it would leave out everything after, messages on screen included.
        const take = Math.min(ceiling - length, MAX_NODE_CHARS);
        if (take <= 0) {
          truncated = true;
          full = true;
          return;
        }
        const raw = data.length > take ? data.slice(0, take) : data;
        parts.push(raw);
        segments.push({ node, start: length, length: raw.length, preserved });
        length += raw.length;
        // Siblings still get indexed, but what was dropped must leave a boundary or the prefix
        // runs into the next node and a match across that seam paints over the gap.
        if (raw.length < data.length) {
          truncated = true;
          pendingSeparator = true;
        }
      } else if (child.nodeType === ELEMENT_NODE) {
        visit(child as FindElementLike, preserved);
        if (full) return;
      }
    }
    if (block) pendingSeparator = true;
  };

  visit(root, false);
  // The reserve, handed over. Portaled surfaces come after the workspace, so without this the one
  // thing in front of the reader is the one left out. `truncated` stays as the workspace left it.
  ceiling = MAX_INDEX_CHARS;
  full = false;
  for (const extra of extraRoots) {
    if (full) break;
    // A boundary between roots: they are separate surfaces, and a match must not run across.
    pendingSeparator = true;
    visit(extra, false);
  }
  // Folded once, over the joined document: see foldText for why it cannot be done a node at a time.
  return { text: foldText(parts.join("")), segments, truncated };
}

/** Fold a query the way the haystack was folded. Null when it cannot match: empty, or carrying the
 *  separator, which only a paste could produce. */
export function normalizeQuery(query: string): string | null {
  if (query.length === 0) return null;
  const folded = foldText(query);
  if (folded.includes(BLOCK_SEPARATOR)) return null;
  return folded;
}

export interface FindMatch {
  start: number;
  /** Exclusive, as a `Range` end is. */
  end: number;
}

/** Regex metacharacters, so a query of "a.b" does not match "axb". */
const REGEX_META_PATTERN = /[.*+?^${}()|[\]\\]/g;

/** Combining dot above, the tail of a decomposed dotted I. */
const COMBINING_DOT = "̇";

/**
 * The canonically equivalent spellings of `needle`, longest first so the fuller one wins.
 *
 * Composed and decomposed spellings look identical on screen and both turn up in one thread.
 * Normalizing the index would change its length, and every offset stands for one document
 * character, so the variants go into the pattern instead and the document is left alone.
 *
 * `dotted` adds the spelling a decomposed dotted I leaves behind. U+0130 decomposes to `I` plus a
 * combining dot, which folds to `i` plus that dot, and `i` + dot has no precomposed form, so NFC
 * cannot put it back and the plain query misses a word plainly on screen. Only offered when the
 * index carries a combining dot, so an ordinary document keeps the single-variant `indexOf` path.
 */
function canonicalVariants(needle: string, dotted: boolean): string[] {
  const variants = [needle];
  for (const form of ["NFC", "NFD"] as const) {
    const variant = needle.normalize(form);
    if (!variants.includes(variant)) variants.push(variant);
  }
  if (dotted && needle.includes("i")) {
    for (const variant of [...variants]) {
      const dottedVariant = variant.replace(/i/g, `i${COMBINING_DOT}`);
      if (!variants.includes(dottedVariant)) variants.push(dottedVariant);
    }
  }
  if (variants.length > 1) variants.sort((a, b) => b.length - a.length);
  return variants;
}

/** A base character with the combining marks that belong to it, which compose or decompose as one. */
const CLUSTER_PATTERN = /[\s\S][̀-ͯ҃-҉᪰-᫿᷀-᷿⃐-⃰︠-︯]*/gu;

/**
 * The needle as a regex source in which each cluster may match either of its spellings.
 *
 * Alternating whole spellings of the WHOLE query only reaches text that is all-composed or
 * all-decomposed. A single occurrence can be neither: joining two text nodes joins two sources, so
 * `café` in one and `café` in the next make one visible word with a spelling the query
 * cannot be written in. Every engine's own find matches that (measured on chromium, firefox and
 * webkit), so it is below the baseline the rest of this file is held to.
 *
 * Per cluster rather than per query, which is also SMALLER: the old form repeated the query once
 * per spelling, this writes it once and pays about eight characters for each cluster that actually
 * has two spellings. Decomposed first, since it is the longer of the two.
 */
function canonicalSource(needle: string, dotted: boolean): string {
  let out = "";
  for (const [cluster] of needle.normalize("NFD").matchAll(CLUSTER_PATTERN)) {
    if (/^\s/.test(cluster)) {
      // A whitespace run in the query matches a run in the document, however it is spelt there.
      out += out.endsWith("\\s+") ? "" : "\\s+";
      continue;
    }
    const spellings = [cluster];
    const composed = cluster.normalize("NFC");
    if (composed !== cluster) spellings.push(composed);
    // A decomposed dotted I folds to `i` plus a combining dot, which has no precomposed form, so
    // NFC cannot put it back and the plain query would miss a word plainly on screen.
    if (dotted && cluster === "i") spellings.push(`i${COMBINING_DOT}`);
    out +=
      spellings.length === 1
        ? escapeForRegex(spellings[0])
        : `(?:${spellings.map(escapeForRegex).join("|")})`;
  }
  return out;
}

function escapeForRegex(text: string): string {
  return text.replace(REGEX_META_PATTERN, "\\$&");
}

/**
 * A pattern for a query spanning whitespace or spelt more than one way, or null for a plain scan.
 *
 * HTML collapses whitespace, so a soft-wrapped paragraph renders as one line while its node holds
 * the newline; each run in the query matches a run in the document. The separator is not
 * whitespace, so block boundaries stay closed. Single-word ASCII queries keep the `indexOf` path.
 */
function matchPattern(variants: string[], needle: string): RegExp | null {
  const dotted = variants.some((variant) => variant.includes(COMBINING_DOT));
  if (variants.length === 1 && !/\s/.test(needle)) return null;
  try {
    const pattern = new RegExp(canonicalSource(needle, dotted), "g");
    // V8 compiles lazily, so an oversized pattern is accepted here and throws on the first `exec`
    // instead, back outside this `try`. One run against nothing forces the compile while it can
    // still be caught, and leaves `lastIndex` at 0 for the real scan.
    pattern.exec("");
    return pattern;
  } catch {
    // Every engine caps how large a pattern it will compile, and the cap is its own business:
    // the spec sets none, so there is no length to test against that would be right everywhere.
    // A pasted log reaches it -- measured at 15,651 characters on V8 -- and the throw came out
    // through the keystroke that caused it and took the bar down with it. Falling back to the
    // literal scan below costs the flexed whitespace and keeps the search working.
    return null;
  }
}

/**
 * Walk every match for `needle` in document order, stopping when `visit` says so. Non-overlapping,
 * like every browser's own find, which is what terminates a self-overlapping query.
 *
 * One place, so the two ways of matching stay one behaviour. Flexed whitespace is for prose; inside
 * a `<pre>` the whitespace on screen IS the whitespace in the node, so "foo bar" must not land on
 * "foo   bar" there. The platform's own find draws the same line.
 */
function eachMatch(
  index: FindTextIndex,
  needle: string,
  visit: (start: number, end: number) => boolean,
): void {
  const variants = canonicalVariants(
    needle,
    index.text.includes(COMBINING_DOT),
  );
  // Nothing longer than the haystack can be inside it. Measured against the SHORTEST spelling,
  // since a decomposed query is longer than the precomposed text it is meant to find. This is
  // ahead of `matchPattern` because that is the expensive half: escaping a pasted log, repeating
  // it once per variant and handing the result to the regex compiler.
  if (
    Math.min(...variants.map((variant) => variant.length)) > index.text.length
  )
    return;
  // What counts as "spelt the way it was typed": any whole-query spelling, or any mixture of the
  // per-cluster ones, which compare equal once composed. Whitespace flexing survives NFC, so a hit
  // that only flexed a space is still told apart from one that merely spells a letter differently.
  const composedNeedle = needle.normalize("NFC");
  const asTyped = (hit: string): boolean =>
    variants.includes(hit) || hit.normalize("NFC") === composedNeedle;
  const pattern = matchPattern(variants, needle);
  if (pattern) {
    for (;;) {
      const hit = pattern.exec(index.text);
      if (hit === null) return;
      const end = hit.index + hit[0].length;
      // Any spelling of the query counts as typed exactly; only flexed whitespace does not.
      if (
        touchesPreserved(index.segments, hit.index, end) &&
        !asTyped(hit[0])
      ) {
        pattern.lastIndex = hit.index + 1;
        continue;
      }
      if (!visit(hit.index, end)) return;
      pattern.lastIndex = end;
    }
  }
  let from = 0;
  for (;;) {
    const at = index.text.indexOf(needle, from);
    if (at === -1) return;
    if (!visit(at, at + needle.length)) return;
    from = at + needle.length;
  }
}

/** `limit` matches starting from the `skip`-th one. */
function collectMatches(
  index: FindTextIndex,
  needle: string,
  limit: number,
  skip: number,
): FindMatch[] {
  const out: FindMatch[] = [];
  let seen = 0;
  eachMatch(index, needle, (start, end) => {
    seen += 1;
    if (seen <= skip) return true;
    out.push({ start, end });
    return out.length < limit;
  });
  return out;
}

/**
 * Matches for `query`, at most `limit` of them, as a window around `anchor`.
 *
 * A common letter in a long thread has tens of thousands of matches, and keeping the first `limit`
 * keeps only the top of the document, walking a reader at the bottom away from the occurrences
 * beside them. When the cap bites the kept matches are the ones nearest the reader; under the cap
 * this is the same single pass it always was.
 *
 * `anchor` may be a thunk, because working it out reads layout and a plain argument is evaluated
 * whether or not it is wanted. A number still means the same thing.
 */
export function findMatches(
  index: FindTextIndex,
  query: string,
  limit = MAX_MATCHES,
  anchor: number | (() => number) = 0,
): FindMatch[] {
  const needle = normalizeQuery(query);
  if (needle === null) return [];
  const head = collectMatches(index, needle, limit, 0);
  // Before resolving the anchor, so an under-cap query never pays for it.
  if (head.length < limit) return head;
  const at = typeof anchor === "function" ? anchor() : anchor;
  if (at <= 0) return head;

  // Capped, so where the window sits matters. Two more passes, no allocation in the first.
  //
  // The count stops early: `total` only keeps the window from running off the end, and once
  // `total - limit` reaches the left edge it can no longer pull it back, so later matches are
  // counted for nothing. Past the anchor `before` is final, which is what makes the edge knowable.
  // The clamp is safe because it only binds near the end, where the true total is under the
  // ceiling and the walk never stops early. At 4M chars, 444,444 matches, anchored halfway:
  // 6.4ms over all of them to 3.3ms over 224,722, same window either way.
  let total = 0;
  let before = 0;
  let enough = Number.POSITIVE_INFINITY;
  eachMatch(index, needle, (start) => {
    total += 1;
    if (start < at) {
      before += 1;
      return true;
    }
    if (enough === Number.POSITIVE_INFINITY) {
      enough = Math.max(before - (limit >> 1), 0) + limit;
    }
    return total < enough;
  });
  // Centred on the reader, then pushed back inside the list at either end.
  const start = Math.min(
    Math.max(before - (limit >> 1), 0),
    Math.max(total - limit, 0),
  );
  return start === 0 ? head : collectMatches(index, needle, limit, start);
}

/**
 * Throw away the one match asked for over the cap, from whichever end the reader is further from.
 *
 * The caller asks for `MAX_MATCHES + 1` so that a page holding exactly the cap does not have to
 * read as a floor, then throws the extra one away. Taking it off the tail is right only while the
 * window starts at the top of the document. Once the reader is far enough down the anchored window
 * IS the tail, and its last entry is the document's final occurrence -- the one nearest them, and
 * the one the walk was then unable to reach at all.
 *
 * Measured on 6000 matches with the reader at the bottom: the window covered the final occurrence
 * at offset 11998, and trimming the tail left the walk ending at 11996.
 *
 * Lives here rather than beside its caller for the same reason `mutatesSearchableText` does: the
 * hook imports React and cannot be loaded under `node --test`.
 */
export function dropProbeFurthestFrom(
  matches: FindMatch[],
  anchor: number | null,
  limit = MAX_MATCHES,
): void {
  // No anchor means `findMatches` never resolved one, which only happens under the cap. A window
  // that already starts at the document's first match has nothing above the reader to give up.
  if (
    anchor !== null &&
    matches.length > 0 &&
    anchor - matches[0].start > matches[matches.length - 1].start - anchor
  ) {
    matches.shift();
    return;
  }
  matches.length = limit;
}

/** True when `[start, end)` reaches into a run whose whitespace is kept rather than collapsed. */
function touchesPreserved(
  segments: TextSegment[],
  start: number,
  end: number,
): boolean {
  let at = segmentAt(segments, start);
  // A match can open on a separator, which belongs to no segment; take the next one along.
  if (at === -1) {
    at = segments.findIndex((segment) => segment.start >= start);
    if (at === -1) return false;
  }
  for (let i = at; i < segments.length; i += 1) {
    const segment = segments[i];
    if (segment.start >= end) return false;
    if (segment.preserved) return true;
  }
  return false;
}

/** The segment holding `offset`, or -1 when it lands on a separator or past the end. */
export function segmentAt(segments: TextSegment[], offset: number): number {
  let lo = 0;
  let hi = segments.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const segment = segments[mid];
    if (offset < segment.start) {
      hi = mid - 1;
    } else if (offset >= segment.start + segment.length) {
      lo = mid + 1;
    } else {
      return mid;
    }
  }
  return -1;
}

/** A text node and an offset inside it, which is what a `Range` boundary takes. */
export interface TextPosition {
  node: FindTextNodeLike;
  offset: number;
}

/** Where a match starts. */
export function startPositionAt(
  segments: TextSegment[],
  offset: number,
): TextPosition | null {
  const index = segmentAt(segments, offset);
  if (index === -1) return null;
  const segment = segments[index];
  return { node: segment.node, offset: offset - segment.start };
}

/**
 * Where a match ends. Located from its last character, since an exclusive end sits one past the run
 * whenever the match finishes a text node, and that is the boundary `setEnd` wants there.
 */
export function endPositionAt(
  segments: TextSegment[],
  end: number,
): TextPosition | null {
  const index = segmentAt(segments, end - 1);
  if (index === -1) return null;
  const segment = segments[index];
  return { node: segment.node, offset: end - segment.start };
}
