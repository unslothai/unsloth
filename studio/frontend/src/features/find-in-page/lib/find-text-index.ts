// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The searchable text of a subtree, flattened into one string plus the map back to its text nodes.
// Flattening once and running `indexOf` keeps a keystroke off the DOM: a tree walk per character
// typed would re-read a 300K conversation six times to type "unsloth". The walk is paid only when
// the document changes (see use-find-in-page).
//
// Pure, and written against structural types, so it runs under `node --test` with the hand-rolled
// DOM in tests/find-in-page.test.ts. There is no DOM library in this project.

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

/** Marks a subtree the bar must not read: the bar itself, first of all. */
export const FIND_SKIP_ATTRIBUTE = "data-find-skip";

/** Marks the scope the bar searches, set on the shell's content region in `__root.tsx`. */
export const FIND_SCOPE_ATTRIBUTE = "data-find-scope";

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

/**
 * Turkish capital dotted I, the only code point in Unicode whose `toLowerCase` is longer than
 * itself. Verified by scanning every assigned one. It is folded to a bare `i`, which is the Turkic
 * case fold of it and the only fold that fits: the default one adds a combining dot, and one index
 * character has to stand for one document character.
 */
const DOTTED_I_PATTERN = /\u0130/g;

/**
 * Greek final sigma, folded to the medial one, which is what `CaseFolding.txt` maps it to.
 *
 * `toLowerCase` picks between the two by what follows, so `\u039f\u03a3` lowercases to a final
 * sigma while the same word typed with a medial one does not, and only one of the two spellings a
 * reader can produce finds text plainly on screen. Every engine's own find treats them as one
 * letter. Both are single code points, so this cannot change a length.
 */
const FINAL_SIGMA_PATTERN = /\u03c2/g;

/**
 * Case-fold the whole flattened document without changing its length.
 *
 * Two mappings the default fold does not make, both of them one code point for one: dotted I,
 * whose own `toLowerCase` is the only one in Unicode that grows, and final sigma, which is a
 * position rather than a letter. Without them a word on screen matches only some of the ways it
 * can be typed.
 *
 * The whole string, not a node at a time. Sigma is the reason the two used to differ, and the
 * mapping above settles it, but the flatten already joins before it folds and there is nothing to
 * gain by unpicking that.
 *
 * With dotted I mapped first, `toLowerCase` cannot change a length, so this is one pass over the
 * string. On a document at the 4,000,000 character ceiling that is 11ms, of which the dotted I pass
 * is 0.7ms; walking code points to the same answer is 146ms.
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

/** True when the walk must not descend into this element. */
export function skipsSubtree(element: FindElementLike): boolean {
  // Uppercased first: only HTML elements report their tag that way. SVG and MathML keep their source
  // casing, so an inline `<svg>` answers "svg" and walked straight past this set, putting Mermaid
  // labels in the index as matches no engine can paint. Already uppercase for everything else.
  if (SKIP_TAGS.has(element.tagName.toUpperCase())) return true;
  if (element.getAttribute(FIND_SKIP_ATTRIBUTE) !== null) return true;
  // Boolean attributes, so presence is the whole signal. The shell parks an off-route workspace
  // under `inert`; Radix marks the page `aria-hidden` behind a modal.
  if (element.getAttribute("hidden") !== null) return true;
  if (element.getAttribute("inert") !== null) return true;
  if (element.getAttribute("aria-hidden") === "true") return true;
  // Anything the engine is not painting. Attributes alone miss the common case: a responsive
  // `hidden lg:flex` is a CLASS, and text under it would be counted, and walked to, while nobody
  // can see it.
  //
  // `contentVisibilityAuto` stays OFF. A `content-visibility: auto` subtree the reader has not
  // scrolled to yet is skipped, not hidden, and asking about it would drop the far half of a Hub
  // README (hub.css) and of a maths-bearing thread from the index. Nothing would put it back
  // either: scrolling renders the subtree without mutating the DOM, so the observer never fires.
  // Opacity is off too, so a message still fading in stays findable.
  //
  // Both spellings of each option go in. `visibilityProperty` and `opacityProperty` are renames;
  // the original names are `checkVisibilityCSS` and `checkOpacity`, and an engine reads only the
  // one it knows. Web IDL drops a dictionary member it does not recognise silently, so passing the
  // modern name alone is not a fallback, it is a no-op on Chrome 105-120 and Firefox 106-121,
  // which have the method but not the rename. Those builds would report `visibility: hidden` text
  // as visible and index, count and highlight it.
  const painted = element.checkVisibility?.({
    contentVisibilityAuto: false,
    opacityProperty: false,
    checkOpacity: false,
    visibilityProperty: true,
    checkVisibilityCSS: true,
  });
  const style = computedStyle(element);
  if (painted === false) {
    // `display: contents` generates no box, and no box is the first thing `checkVisibility` calls
    // invisible, so a wrapper whose children are all on screen answers false. The shell uses one
    // (sidebar.tsx) and so does the training page (studio-page.tsx), which between them is most of
    // what there is to search. A real box is what makes an element hidden rather than absent.
    return style?.display !== "contents";
  }
  // No `checkVisibility` to ask. It landed in Safari 17.4, and WebKitGTK is already a supported
  // engine here -- it is the one `selectRangeFallback` exists for -- so this is a real path, and
  // taking the visible branch on it would index every `display: none` subtree in the app. The two
  // properties below are what the call above is asked for: `visibilityProperty` on, opacity and
  // content-visibility off.
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
 * True when this element keeps its own text off the screen while still being descended into.
 *
 * Only `display: contents` reaches this. `skipsSubtree` lets that case through on purpose, because
 * a boxless wrapper is absent rather than hidden and its children are usually painted normally. But
 * `visibility` inherits, so a `contents` wrapper that is also `visibility: hidden` paints neither
 * itself nor its text, and its DIRECT TEXT children never pass through `skipsSubtree` at all -
 * only element children are re-checked. Without this they are indexed, counted and highlighted.
 *
 * Deliberately about the element's own text, not the subtree: a descendant that sets
 * `visibility: visible` is painted again, and is still reached because the walk does not turn back.
 */
function hidesOwnText(style: ResolvedStyle | null): boolean {
  return (
    style?.display === "contents" &&
    (style.visibility === "hidden" || style.visibility === "collapse")
  );
}

/**
 * True when this element paints no box at all, for engines with no `checkVisibility` to ask.
 *
 * `display: contents` is not one of them: it is boxless rather than hidden, `hidesOwnText` covers
 * the text it holds directly, and the walk keeps descending so a child that turns visibility back
 * on is still found. That is what the branch above does when the API answers, so the two agree.
 */
function paintsNothing(style: ResolvedStyle | null): boolean {
  if (style?.display === "none") return true;
  if (style?.display === "contents") return false;
  return style?.visibility === "hidden" || style?.visibility === "collapse";
}

/**
 * The two spellings of the visually-hidden idiom, which Tailwind's `sr-only` uses and nothing else
 * does: a real box, full opacity, clipped to nothing. `checkVisibility` calls that visible, so
 * without this the app's 46 screen-reader labels are counted and walked to under a highlight
 * clipped away with them.
 */
function clippedAway(style: ResolvedStyle | null): boolean {
  return (
    style?.clipPath === "inset(50%)" ||
    style?.clip === "rect(0px, 0px, 0px, 0px)"
  );
}

/**
 * `display` and `white-space` as resolved, or null off the DOM, where the tag sets stand in.
 *
 * One read per element, used for three things: telling a boxless wrapper from a hidden one, finding
 * the block boundaries no tag name carries, and knowing where whitespace is preserved. Measured at
 * 8000 elements, a thread-sized tree: 2.5ms for the visibility check alone, 4.3ms with this, against
 * a 300ms floor between rebuilds.
 */
function computedStyle(element: FindElementLike): ResolvedStyle | null {
  const view = globalThis as unknown as {
    getComputedStyle?: (element: FindElementLike) => ResolvedStyle;
  };
  return view.getComputedStyle?.(element) ?? null;
}

/**
 * True when this element ends the line it sits on.
 *
 * `inline-block` and friends do not, and `contents` has no box of its own. Being wrong about an
 * exotic display inserts a separator that was not needed, which can only lose a match, never invent
 * one, so anything unrecognised counts as a boundary.
 */
function isBlockDisplay(display: string | undefined): boolean {
  if (display === undefined) return false;
  return !(
    display.startsWith("inline") ||
    display === "contents" ||
    display === "none"
  );
}

/**
 * True where whitespace is kept rather than collapsed: `pre`, `pre-wrap`, `break-spaces`.
 *
 * `pre-line` is not in it. That mode still collapses runs of spaces, which is the half the query
 * flexibility below cares about; it only keeps newlines.
 */
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
 * and the closing separator is what stops `<p>a</p>b` reading as "ab". Depth is bounded by markup.
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
    if (skipsSubtree(element)) return;
    const style = computedStyle(element);
    // The tag set is the fallback and the fast answer for `<br>`, whose display is inline. Layout is
    // the rest: Tailwind renders `span.block` all over the app, and two of those run together in the
    // index without a boundary, so "Open" above "AI" reads as one word.
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
        // Checked before the separator is written, not after: a separator emitted with the ceiling
        // already reached pushes `length` past it, and the negative `room` that follows turns
        // `slice(0, room)` into "all but the last character" of the next node.
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
        // A share of the budget, not all of it. Taking the prefix that fits keeps a document made
        // of one huge node findable, but taking the WHOLE remaining budget for it left everything
        // after it out, the messages on screen included.
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
        // Clipping a node is not the end of the walk: its siblings still have to be indexed. But
        // what was dropped has to leave a boundary, or the retained prefix runs straight into the
        // next node and a match across that seam paints over everything thrown away in between.
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
  // The reserve, handed over. A workspace that filled its share stops the walk, and the portaled
  // surfaces come after it, so without this the one thing in front of the reader was the one thing
  // left out. `truncated` stays as the workspace left it: what it dropped is still dropped.
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

/**
 * Fold a query the way the haystack was folded. Null for a query that cannot match: empty, or one
 * carrying the separator, which only a paste could produce.
 */
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

/**
 * The canonically equivalent spellings of `needle`, longest first, `needle` itself always included.
 *
 * The same word can be composed or decomposed and look identical on screen: `café` is four code
 * points, `café` is five. macOS hands back decomposed filenames and a model writes composed
 * prose, so both forms turn up in one thread, and the platform's own find matches either from
 * either. Normalizing the index would change its length, and every offset in it stands for one
 * character of the document, so the variants go into the pattern and the document is left alone.
 *
 * Longest first, so where two could match at one position the fuller spelling wins.
 */
function canonicalVariants(needle: string): string[] {
  const variants = [needle];
  for (const form of ["NFC", "NFD"] as const) {
    const variant = needle.normalize(form);
    if (!variants.includes(variant)) variants.push(variant);
  }
  if (variants.length > 1) variants.sort((a, b) => b.length - a.length);
  return variants;
}

/**
 * A pattern for a query that spans whitespace or is spelt more than one way, or null for a plain
 * scan.
 *
 * HTML collapses runs of whitespace, so a markdown paragraph soft-wrapped mid-sentence renders as
 * one line while its text node still holds the newline. Searching the phrase a reader can see would
 * otherwise miss it. Each run of whitespace in the query matches a run in the document; the
 * separator is not whitespace, so block boundaries stay closed.
 *
 * Single-word ASCII queries, which are most of them, keep the `indexOf` path.
 */
function matchPattern(variants: string[], needle: string): RegExp | null {
  if (variants.length === 1 && !/\s/.test(needle)) return null;
  const escaped = variants.map((variant) =>
    variant.replace(REGEX_META_PATTERN, "\\$&").replace(/\s+/g, "\\s+"),
  );
  return new RegExp(
    escaped.length === 1 ? escaped[0] : `(?:${escaped.join("|")})`,
    "g",
  );
}

/**
 * Every occurrence of `query`, left to right, capped at `limit`. Non-overlapping, like every
 * browser's own find, which is what makes the walk terminate on a self-overlapping query.
 */
/**
 * Walk every match for `needle` in document order, stopping when `visit` says so.
 *
 * One place, so the two ways of matching stay one behaviour. The flexible run is for prose, where
 * the source newline of a soft wrap renders as a space; inside a `<pre>` the whitespace on screen
 * IS the whitespace in the node, so a query for "foo bar" must not land on "foo   bar" there.
 * Measured: the platform's own find draws the same line.
 */
function eachMatch(
  index: FindTextIndex,
  needle: string,
  visit: (start: number, end: number) => boolean,
): void {
  const variants = canonicalVariants(needle);
  const pattern = matchPattern(variants, needle);
  if (pattern) {
    for (;;) {
      const hit = pattern.exec(index.text);
      if (hit === null) return;
      const end = hit.index + hit[0].length;
      // Any spelling of the query counts as typed exactly; only flexed whitespace does not.
      if (
        touchesPreserved(index.segments, hit.index, end) &&
        !variants.includes(hit[0])
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
 * The window is what `anchor` is for. A single common letter in a long thread has tens of thousands
 * of matches, and keeping the first `limit` of them keeps only the top of the document: a reader at
 * the bottom is walked away from every occurrence beside them, to a match they were not looking for.
 * So when the cap bites, the kept matches are the ones nearest where the reader is.
 *
 * Costs nothing until it bites. Under the cap this is the same single pass it always was.
 *
 * `anchor` may be a thunk, and the caller that knows where the reader is passes one. Working out
 * the viewport offset means reading layout, and as a plain argument it was evaluated on every
 * keystroke however few matches there were, which is the one thing this comment promised it did
 * not do. A number still works and still means the same thing.
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

  // Capped, so where the window sits matters. Two more passes over a string, no allocation in the
  // first: cheap next to the tens of thousands of matches this only happens for.
  //
  // The count stops early. `total` is only wanted to keep the window from running off the end, and
  // once `total - limit` has reached the left edge below it can no longer pull that edge back, so
  // every match after that point is counted for nothing. Past the anchor `before` is final, which
  // is what makes the edge knowable there.
  //
  // It cannot cost the clamp: the clamp only binds when the reader is near the end, and there the
  // true total is below the ceiling, so the walk never stops early in the first place. Measured on
  // a 4,000,000 character index with a single-letter query, 444,444 matches, anchored halfway:
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
