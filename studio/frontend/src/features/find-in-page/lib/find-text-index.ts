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
}

export type FindNodeLike = FindTextNodeLike | FindElementLike;

/** One text node's contribution, at the offset its first character took in `text`. */
export interface TextSegment {
  node: FindTextNodeLike;
  start: number;
  /** Always the node's own `data.length`, so an offset inside the run maps straight through. */
  length: number;
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
 * Case-fold a run, but only when folding leaves its length alone.
 *
 * One character of `text` has to stand for one character of a text node, and Turkish dotted I
 * (U+0130) folds to two code units. A run holding one would shift every offset after it, painting
 * highlights to the left of the word, so it keeps its case instead.
 */
export function foldChunk(raw: string): string {
  const spaced = raw.replace(HARD_SPACE_PATTERN, " ");
  const folded = spaced.toLowerCase();
  return folded.length === spaced.length ? folded : spaced;
}

/** True when the walk must not descend into this element. */
export function skipsSubtree(element: FindElementLike): boolean {
  if (SKIP_TAGS.has(element.tagName)) return true;
  if (element.getAttribute(FIND_SKIP_ATTRIBUTE) !== null) return true;
  // Boolean attributes, so presence is the whole signal. The shell parks an off-route workspace
  // under `inert`; Radix marks the page `aria-hidden` behind a modal.
  if (element.getAttribute("hidden") !== null) return true;
  if (element.getAttribute("inert") !== null) return true;
  return element.getAttribute("aria-hidden") === "true";
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

  const visit = (element: FindElementLike): void => {
    if (skipsSubtree(element)) return;
    const block = BLOCK_TAGS.has(element.tagName);
    if (block) pendingSeparator = true;
    const children = element.childNodes;
    for (let i = 0; i < children.length; i += 1) {
      if (truncated) return;
      const child = children[i];
      if (child.nodeType === TEXT_NODE) {
        const node = child as FindTextNodeLike;
        const raw = node.data;
        if (raw.length === 0) continue;
        if (length + raw.length > MAX_INDEX_CHARS) {
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
        parts.push(foldChunk(raw));
        segments.push({ node, start: length, length: raw.length });
        length += raw.length;
      } else if (child.nodeType === ELEMENT_NODE) {
        visit(child as FindElementLike);
        if (truncated) return;
      }
    }
    if (block) pendingSeparator = true;
  };

  visit(root);
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
  let from = 0;
  for (;;) {
    const at = index.text.indexOf(needle, from);
    if (at === -1) return out;
    out.push({ start: at, end: at + needle.length });
    if (out.length >= limit) return out;
    from = at + needle.length;
  }
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
