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
 * Case-fold a run without changing its length.
 *
 * One character of `text` has to stand for one character of a text node, and Turkish dotted I
 * (U+0130) folds to two code units. The fast path is the whole run at once; when that grows, fold
 * per code point and keep only the folds that fit, so one such character no longer makes the rest
 * of its run case-sensitive.
 */
export function foldChunk(raw: string): string {
  const spaced = raw.replace(HARD_SPACE_PATTERN, " ");
  const folded = spaced.toLowerCase();
  if (folded.length === spaced.length) return folded;
  let out = "";
  for (const point of spaced) {
    const lower = point.toLowerCase();
    out += lower.length === point.length ? lower : point;
  }
  return out;
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
  if (
    element.checkVisibility?.({
      contentVisibilityAuto: false,
      opacityProperty: false,
      visibilityProperty: true,
    }) !== false
  ) {
    return false;
  }
  // `display: contents` generates no box, and no box is the first thing `checkVisibility` calls
  // invisible, so a wrapper whose children are all on screen answers false. The shell uses one
  // (sidebar.tsx) and so does the training page (studio-page.tsx), which between them is most of
  // what there is to search. A real box is what makes an element hidden rather than absent.
  return computedStyle(element)?.display !== "contents";
}

interface ResolvedStyle {
  display?: string;
  whiteSpace?: string;
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
export function buildTextIndex(root: FindElementLike): FindTextIndex {
  const parts: string[] = [];
  const segments: TextSegment[] = [];
  let length = 0;
  let truncated = false;
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
    const children = element.childNodes;
    for (let i = 0; i < children.length; i += 1) {
      if (truncated) return;
      const child = children[i];
      if (child.nodeType === TEXT_NODE) {
        const node = child as FindTextNodeLike;
        const full = node.data;
        if (full.length === 0) continue;
        // Checked before the separator is written, not after: a separator emitted with the ceiling
        // already reached pushes `length` past it, and the negative `room` that follows turns
        // `slice(0, room)` into "all but the last character" of the next node.
        if (length >= MAX_INDEX_CHARS) {
          truncated = true;
          return;
        }
        if (pendingSeparator) {
          pendingSeparator = false;
          if (length > 0) {
            parts.push(BLOCK_SEPARATOR);
            length += 1;
          }
        }
        const room = MAX_INDEX_CHARS - length;
        if (room <= 0) {
          truncated = true;
          return;
        }
        // A single node can be bigger than the whole ceiling: one Bash step's log arrives as one
        // text node. Take the prefix that fits rather than dropping the node, or a document made
        // of one such node would index to nothing at all.
        const raw = full.length > room ? full.slice(0, room) : full;
        if (raw.length < full.length) truncated = true;
        parts.push(foldChunk(raw));
        segments.push({ node, start: length, length: raw.length, preserved });
        length += raw.length;
        if (truncated) return;
      } else if (child.nodeType === ELEMENT_NODE) {
        visit(child as FindElementLike, preserved);
        if (truncated) return;
      }
    }
    if (block) pendingSeparator = true;
  };

  visit(root, false);
  return { text: parts.join(""), segments, truncated };
}

/**
 * Fold a query the way the haystack was folded. Null for a query that cannot match: empty, or one
 * carrying the separator, which only a paste could produce.
 */
export function normalizeQuery(query: string): string | null {
  if (query.length === 0) return null;
  const folded = foldChunk(query);
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
 * A pattern for a query that spans whitespace, or null when a plain scan will do.
 *
 * HTML collapses runs of whitespace, so a markdown paragraph soft-wrapped mid-sentence renders as
 * one line while its text node still holds the newline. Searching the phrase a reader can see would
 * otherwise miss it. Each run of whitespace in the query matches a run in the document; the
 * separator is not whitespace, so block boundaries stay closed.
 *
 * Single-word queries, which are most of them, keep the `indexOf` path.
 */
function whitespacePattern(needle: string): RegExp | null {
  if (!/\s/.test(needle)) return null;
  const escaped = needle
    .replace(REGEX_META_PATTERN, "\\$&")
    .replace(/\s+/g, "\\s+");
  return new RegExp(escaped, "g");
}

/**
 * Every occurrence of `query`, left to right, capped at `limit`. Non-overlapping, like every
 * browser's own find, which is what makes the walk terminate on a self-overlapping query.
 */
export function findMatches(
  index: FindTextIndex,
  query: string,
  limit = MAX_MATCHES,
): FindMatch[] {
  const needle = normalizeQuery(query);
  if (needle === null) return [];
  const out: FindMatch[] = [];
  const pattern = whitespacePattern(needle);
  if (pattern) {
    for (;;) {
      const hit = pattern.exec(index.text);
      if (hit === null) return out;
      const end = hit.index + hit[0].length;
      // The flexible run is for prose, where the source newline of a soft wrap renders as a space.
      // Inside a `<pre>` the whitespace on screen is the whitespace in the node, so a query for
      // "foo bar" must not land on "foo   bar" there. Measured: the platform's own find draws the
      // same line, matching across a wrap in a paragraph and not inside a code fence.
      if (
        touchesPreserved(index.segments, hit.index, end) &&
        hit[0] !== needle
      ) {
        pattern.lastIndex = hit.index + 1;
        continue;
      }
      out.push({ start: hit.index, end });
      if (out.length >= limit) return out;
      pattern.lastIndex = end;
    }
  }
  let from = 0;
  for (;;) {
    const at = index.text.indexOf(needle, from);
    if (at === -1) return out;
    out.push({ start: at, end: at + needle.length });
    if (out.length >= limit) return out;
    from = at + needle.length;
  }
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
