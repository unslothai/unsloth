// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The searchable text of a subtree, flattened into one string plus the map back to its text nodes.
// Structurally typed, so `node --test` can run it against a hand-rolled DOM.

import { FIND_SKIP_ATTRIBUTE } from "./find-attributes.ts";

export const ELEMENT_NODE = 1;
export const TEXT_NODE = 3;

/** Written at block boundaries. NUL because no query can contain it, bar a paste, which
 *  `normalizeQuery` rejects. */
export const BLOCK_SEPARATOR = "\u0000";

/** A thread is bounded by what a person scrolled through; a tool result is not. */
export const MAX_INDEX_CHARS = 4_000_000;

export const MAX_MATCHES = 5_000;

/** A log arrives as one text node and would otherwise spend the whole budget on its own. */
export const MAX_NODE_CHARS = 100_000;

/** Held back from the workspace, walked first, for the surfaces portaled in front of it. */
export const PORTAL_RESERVE_CHARS = 100_000;

/** Form controls carry their text in `value`; `SVG`/`CANVAS` content is not paintable everywhere. */
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

/** A tag set, not `getComputedStyle`: being wrong inserts a needless break, never a match. */
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

export {
  FIND_SCOPE_ATTRIBUTE,
  FIND_SKIP_ATTRIBUTE,
} from "./find-attributes.ts";

/** Spaces that render as a space but are not one. Each is one UTF-16 unit, keeping the map valid. */
const HARD_SPACE_PATTERN = /[\u00A0\u2002\u2003\u2007\u2009\u202F]/g;

export interface FindTextNodeLike {
  readonly nodeType: number;
  readonly data: string;
}

export interface FindElementLike {
  readonly nodeType: number;
  readonly tagName: string;
  readonly childNodes: ArrayLike<FindTextNodeLike | FindElementLike>;
  getAttribute(name: string): string | null;
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

export interface TextSegment {
  node: FindTextNodeLike;
  start: number;
  length: number;
  preserved: boolean;
}

export interface FindTextIndex {
  text: string;
  /** Sorted by `start`, gapped wherever a separator was written. */
  segments: TextSegment[];
  truncated: boolean;
}

export const EMPTY_TEXT_INDEX: FindTextIndex = {
  text: "",
  segments: [],
  truncated: false,
};

/** The only code point whose `toLowerCase` grows. Mapped to the Turkic fold, a bare `i`, since one
 *  index char must stand for one document char. */
const DOTTED_I_PATTERN = /\u0130/g;

/** Mapped to medial as `CaseFolding.txt` does: `toLowerCase` picks by what follows, so otherwise
 *  only one of the two spellings a reader can type would match. */
const FINAL_SIGMA_PATTERN = /\u03c2/g;

export function foldText(raw: string): string {
  const spaced = raw
    .replace(HARD_SPACE_PATTERN, " ")
    .replace(DOTTED_I_PATTERN, "i");
  const folded = spaced.toLowerCase();
  if (folded.length === spaced.length) {
    return folded.replace(FINAL_SIGMA_PATTERN, "\u03c3");
  }
  // Unreachable, but a wrong length would misplace every offset after it.
  let plain = "";
  for (const point of spaced) {
    const lower = point.toLowerCase();
    plain += lower.length === point.length ? lower : point;
  }
  return plain.replace(FINAL_SIGMA_PATTERN, "\u03c3");
}

function skipsByMarkup(element: FindElementLike): boolean {
  // Uppercased: SVG and MathML keep their source casing, so `<svg>` answers "svg" and walks past.
  if (SKIP_TAGS.has(element.tagName.toUpperCase())) return true;
  if (element.getAttribute(FIND_SKIP_ATTRIBUTE) !== null) return true;
  if (element.getAttribute("hidden") !== null) return true;
  if (element.getAttribute("inert") !== null) return true;
  return element.getAttribute("aria-hidden") === "true";
}

export function skipsSubtree(
  element: FindElementLike,
  style: ResolvedStyle | null = computedStyle(element),
): boolean {
  if (skipsByMarkup(element)) return true;
  // `contentVisibilityAuto` off, since such a subtree is skipped rather than hidden and nothing
  // would put it back (scrolling renders without mutating, so the observer never fires); opacity
  // off, so a message fading in stays findable. Both spellings of each option, since an engine
  // reads only the name it knows: the modern one alone is a no-op on Chrome 105-120, Firefox 106-121.
  const painted = element.checkVisibility?.({
    contentVisibilityAuto: false,
    opacityProperty: false,
    checkOpacity: false,
    visibilityProperty: true,
    checkVisibilityCSS: true,
  });
  if (painted === false) {
    // No box is the first thing `checkVisibility` calls invisible, and the shell is built out of
    // `display: contents` wrappers whose children are all on screen.
    return style?.display !== "contents";
  }
  // `checkVisibility` landed in Safari 17.4 and WebKitGTK is supported here, so this is a real path.
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

/** `skipsSubtree` lets a `display: contents` wrapper through, but `visibility` inherits and only
 *  ELEMENT children are re-checked, so a direct text child of a hidden one would be indexed. */
function hidesOwnText(style: ResolvedStyle | null): boolean {
  return (
    style?.display === "contents" &&
    (style.visibility === "hidden" || style.visibility === "collapse")
  );
}

/** For engines with no `checkVisibility`. `display: contents` is boxless, not hidden. */
function paintsNothing(style: ResolvedStyle | null): boolean {
  if (style?.display === "none") return true;
  if (style?.display === "contents") return false;
  return style?.visibility === "hidden" || style?.visibility === "collapse";
}

/** Tailwind's `sr-only`: a real box at full opacity, which `checkVisibility` calls visible. */
function clippedAway(style: ResolvedStyle | null): boolean {
  return (
    style?.clipPath === "inset(50%)" ||
    style?.clip === "rect(0px, 0px, 0px, 0px)"
  );
}

function computedStyle(element: FindElementLike): ResolvedStyle | null {
  const view = globalThis as unknown as {
    getComputedStyle?: (element: FindElementLike) => ResolvedStyle;
  };
  return view.getComputedStyle?.(element) ?? null;
}

/** Anything unrecognised is a boundary: a needless separator can lose a match, never invent one. */
function isBlockDisplay(display: string | undefined): boolean {
  if (display === undefined) return false;
  return !(
    display.startsWith("inline") ||
    display === "contents" ||
    display === "none"
  );
}

/** `pre-line` is excluded: it still collapses runs of spaces, which is the half that matters. */
function preservesWhitespace(whiteSpace: string | undefined): boolean {
  return (
    whiteSpace === "pre" ||
    whiteSpace === "pre-wrap" ||
    whiteSpace === "break-spaces"
  );
}

/** Recursive rather than a `TreeWalker`, which reports entering an element but not leaving one, and
 *  the closing separator is what stops `<p>a</p>b` reading as "ab". */
export function buildTextIndex(
  root: FindElementLike,
  extraRoots: readonly FindElementLike[] = [],
): FindTextIndex {
  const parts: string[] = [];
  const segments: TextSegment[] = [];
  let length = 0;
  let truncated = false;
  /** The ceiling, the only thing that stops the walk. */
  let full = false;
  let ceiling =
    MAX_INDEX_CHARS - (extraRoots.length > 0 ? PORTAL_RESERVE_CHARS : 0);
  // Written lazily, so a run of empty blocks costs nothing and no separator lands at either end.
  let pendingSeparator = false;

  const visit = (element: FindElementLike, inherited: boolean): void => {
    // Markup first, so a subtree turned down on a tag or attribute costs no layout read at all.
    if (skipsByMarkup(element)) return;
    const style = computedStyle(element);
    if (skipsSubtree(element, style)) return;
    // The tag set answers `<br>`, whose display is inline; layout catches two stacked `span.block`.
    const block =
      BLOCK_TAGS.has(element.tagName) || isBlockDisplay(style?.display);
    if (block) pendingSeparator = true;
    const preserved =
      style?.whiteSpace === undefined
        ? inherited
        : preservesWhitespace(style.whiteSpace);
    // Its own text is the one thing `skipsSubtree` never gets to judge.
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
        // A share, not all: one huge node given the rest leaves out everything after it.
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
        // What was dropped must leave a boundary, or a match across the seam paints over the gap.
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
  // The reserve, handed over: portals come last, so without it they are what gets left out.
  ceiling = MAX_INDEX_CHARS;
  full = false;
  for (const extra of extraRoots) {
    if (full) break;
    pendingSeparator = true;
    visit(extra, false);
  }
  // Folded once over the joined document: a fold is context-sensitive and cannot go node at a time.
  return { text: foldText(parts.join("")), segments, truncated };
}

/** Null when the query cannot match: empty, or carrying the separator (only a paste can). */
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

const REGEX_META_PATTERN = /[.*+?^${}()|[\]\\]/g;

const COMBINING_DOT = "̇";

/** Longest first. Normalizing the index instead would change its length, and every offset stands
 *  for one document character. */
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

/** Hangul first and by its own rule: NFD takes a syllable apart into Jamo, none of which are
 *  combining marks, so the general branch made three clusters of one syllable and composed none
 *  back, and every Korean query missed. */
const CLUSTER_PATTERN =
  /[\u1100-\u115f\ua960-\ua97c]+[\u1160-\u11a7\ud7b0-\ud7c6]*[\u11a8-\u11ff\ud7cb-\ud7fb]*|[\s\S][̀-ͯ҃-҉᪰-᫿᷀-᷿⃐-⃰︠-︯]*/gu;

/** A trailing Jamo closing a syllable, and a leading Jamo with the vowel that makes it a syllable.
 *  The vowel is required: a bare leading Jamo is its own grapheme, not a syllable waiting to be
 *  closed, so a trailing Jamo after one belongs to something else and must not be fenced off. */
const HANGUL_TRAILING_PATTERN = /[\u11a8-\u11ff\ud7cb-\ud7fb]/;
/** A leading Jamo, and a leading Jamo followed by the vowel that makes the two a syllable. */
const HANGUL_LEADING_PATTERN = /^[\u1100-\u115f\ua960-\ua97c]/;
/** Characters that join to whatever follows them (UAX 29 Prepend), plus the leading Jamo that
 *  joins a syllable. Checked in code rather than as a lookbehind: JavaScriptCore only shipped
 *  lookbehind in Safari 16.4, and an engine without it fell back to an unfenced scan. */
const PREPEND_PATTERN = /[\u0600-\u0605\u06dd\u070f\u0890\u0891\u08e2\u0d4e]/;

/** True when a match at `at` would begin part way through the grapheme before it. */
function beginsInsideGrapheme(text: string, at: number): boolean {
  if (at === 0) return false;
  const before = text[at - 1];
  return PREPEND_PATTERN.test(before) || HANGUL_LEADING_PATTERN.test(before);
}

/** Any Hangul Jamo or precomposed syllable. */
const HANGUL_ANY_PATTERN =
  /[\u1100-\u11ff\ua960-\ua97c\ud7b0-\ud7fb\uac00-\ud7a3]/;

/** What may still join a grapheme, by the last Jamo the query got to (UAX 29 GB6/GB7/GB8). */
const HANGUL_AFTER_L =
  "(?![\\u1100-\\u115f\\ua960-\\ua97c\\u1160-\\u11a7\\ud7b0-\\ud7c6\\uac00-\\ud7a3\\u0300-\\u036f\\u0483-\\u0489\\u0591-\\u05bd\\u0610-\\u061a\\u064b-\\u065f\\u0670\\u06d6-\\u06dc\\u0e31\\u0e34-\\u0e3a\\u1ab0-\\u1aff\\u1dc0-\\u1dff\\u20d0-\\u20f0\\ufe00-\\ufe0f\\ufe20-\\ufe2f\\u200d])";
const HANGUL_AFTER_V =
  "(?![\\u1160-\\u11ff\\ud7b0-\\ud7c6\\ud7cb-\\ud7fb\\u0300-\\u036f\\u0483-\\u0489\\u0591-\\u05bd\\u0610-\\u061a\\u064b-\\u065f\\u0670\\u06d6-\\u06dc\\u0e31\\u0e34-\\u0e3a\\u1ab0-\\u1aff\\u1dc0-\\u1dff\\u20d0-\\u20f0\\ufe00-\\ufe0f\\ufe20-\\ufe2f\\u200d])";
const HANGUL_AFTER_T =
  "(?![\\u11a8-\\u11ff\\ud7cb-\\ud7fb\\u0300-\\u036f\\u0483-\\u0489\\u0591-\\u05bd\\u0610-\\u061a\\u064b-\\u065f\\u0670\\u06d6-\\u06dc\\u0e31\\u0e34-\\u0e3a\\u1ab0-\\u1aff\\u1dc0-\\u1dff\\u20d0-\\u20f0\\ufe00-\\ufe0f\\ufe20-\\ufe2f\\u200d])";

const HANGUL_SYLLABLE_PATTERN =
  /^[\u1100-\u115f\ua960-\ua97c]+[\u1160-\u11a7\ud7b0-\ud7c6]/;

/**
 * Per cluster, because alternating whole spellings of the WHOLE query reaches only all-composed or
 * all-decomposed text, and one occurrence can be neither: two text nodes joined join two sources,
 * so
 * `café` in one and `café` in the next make one visible word with a spelling the query
 * cannot be written in. Every engine's own find matches it. */
function canonicalSource(needle: string, dotted: boolean): string {
  let out = "";
  for (const [cluster] of needle.normalize("NFD").matchAll(CLUSTER_PATTERN)) {
    if (/^\s/.test(cluster)) {
      out += out.endsWith("\\s+") ? "" : "\\s+";
      continue;
    }
    const spellings = [cluster];
    const composed = cluster.normalize("NFC");
    if (composed !== cluster) spellings.push(composed);
    // `i` plus a combining dot has no precomposed form, so NFC cannot put it back.
    if (dotted && cluster === "i") spellings.push(`i${COMBINING_DOT}`);
    // A Hangul syllable has a third spelling: the LV part precomposed with its trailing Jamo left
    // alone. Joining two text nodes produces exactly that, an LV syllable in one and the T in the
    // next, and it normalizes to the same word.
    const trailing = HANGUL_TRAILING_PATTERN.exec(cluster);
    if (trailing !== null) {
      const half =
        cluster.slice(0, trailing.index).normalize("NFC") +
        cluster.slice(trailing.index);
      if (!spellings.includes(half)) spellings.push(half);
    }
    out +=
      spellings.length === 1
        ? escapeForRegex(spellings[0])
        : `(?:${spellings.map(escapeForRegex).join("|")})`;
  }
  return out;
}

/** How far back to look for a position that is certainly a grapheme boundary. Whitespace and the
 *  block separator are both boundaries, and prose has one every few characters. */
const ANCHOR_SCAN = 256;
const BOUNDARY_ANCHOR = /[\s\u0000]/;

/** Anything that could extend or be extended into a grapheme. See `alignsToGraphemes`. */
const JOINS_GRAPHEME = /[^\u0000-\u02ff]/;

let segmenter:
  | { segment(input: string): Iterable<{ index: number }> }
  | null
  | undefined;

/** The platform's own grapheme segmenter, or null where there is none. */
function graphemeSegmenter() {
  if (segmenter !== undefined) return segmenter;
  const scope = globalThis as unknown as {
    Intl?: { Segmenter?: new (locale?: string, options?: object) => never };
  };
  segmenter =
    typeof scope.Intl?.Segmenter === "function"
      ? new scope.Intl.Segmenter(undefined, { granularity: "grapheme" })
      : null;
  return segmenter;
}

/**
 * True when `[start, end)` begins and ends where a grapheme does.
 *
 * Asked of the platform rather than answered here. Six rounds of review found six ways to land
 * inside a cluster, each a Unicode range that had not been enumerated, and there is no end to that
 * list: Hangul Jamo, then combining marks, then Prepend, then spacing marks and skin tones. The
 * segmenter already knows the whole of UAX 29 and is kept current with it.
 *
 * Only around the match, not over the whole index: a 4M character haystack is not worth segmenting
 * to place two offsets. The window starts at the nearest whitespace or block separator behind the
 * match, which are boundaries by definition, so the segmentation inside it is the real one. Where
 * no anchor is near, the match is allowed, which is what this did before any of this existed.
 */
function alignsToGraphemes(text: string, start: number, end: number): boolean {
  // Almost every match is in text that cannot join at either edge, and asking the segmenter costs
  // far more than looking. Nothing below U+0300 extends a grapheme: the lowest combining mark is
  // U+0300, the lowest spacing mark U+0903, Prepend starts at U+0600, Hangul Jamo at U+1100, and
  // everything astral arrives here as a surrogate. Latin prose therefore pays one comparison.
  // Both sides of each edge, not just the outside: a query can itself begin with a mark that joins
  // what precedes it, or end with one that joins what follows.
  if (
    !(start > 0 && JOINS_GRAPHEME.test(text[start - 1])) &&
    !JOINS_GRAPHEME.test(text[start]) &&
    !JOINS_GRAPHEME.test(text[end - 1]) &&
    !(end < text.length && JOINS_GRAPHEME.test(text[end]))
  ) {
    return true;
  }
  const segments = graphemeSegmenter();
  if (segments === null) return true;
  let from = start;
  while (from > 0 && start - from < ANCHOR_SCAN) {
    if (BOUNDARY_ANCHOR.test(text[from - 1])) break;
    from -= 1;
  }
  // Forward only as far as the next anchor, for the same reason: what decides the boundary at the
  // end of the match is the handful of characters after it, and in prose the next space is close.
  let to = end;
  while (to < text.length && to - end < ANCHOR_SCAN) {
    if (BOUNDARY_ANCHOR.test(text[to])) break;
    to += 1;
  }
  // A run with no anchor in reach is still segmented, from as far back as the scan went, rather
  // than waved through: a log line or a URL has no whitespace for hundreds of characters, and
  // accepting there put the hole straight back. Only the first cluster of such a window can be
  // misread, and by `start` the segmentation has long resynchronised.
  let sawStart = start === from;
  let sawEnd = false;
  for (const { index } of segments.segment(text.slice(from, to))) {
    const at = from + index;
    if (at === start) sawStart = true;
    if (at === end) sawEnd = true;
    if (at > end) break;
  }
  return sawStart && (sawEnd || end === to);
}

function escapeForRegex(text: string): string {
  return text.replace(REGEX_META_PATTERN, "\\$&");
}

/** Null for a plain scan. Whitespace flexes because a soft-wrapped paragraph renders as one line
 *  while its node holds the newline; the separator is not whitespace, so blocks stay shut. */
function matchPattern(variants: string[], needle: string): RegExp | null {
  const dotted = variants.some((variant) => variant.includes(COMBINING_DOT));
  if (variants.length === 1 && !/\s/.test(needle)) return null;
  try {
    const pattern = new RegExp(canonicalSource(needle, dotted), "g");
    // V8 compiles lazily, so an oversized pattern is accepted here and throws on the first `exec`,
    // outside this `try`. One run against nothing forces the compile while it is catchable.
    pattern.exec("");
    return pattern;
  } catch {
    // Every engine caps how large a pattern it compiles and the spec sets none, so there is no
    // length to test against. A pasted log reaches it, and the throw took the bar down with it.
    return null;
  }
}

/** Non-overlapping, like every browser's own find, which terminates a self-overlapping query.
 *  Inside a `<pre>` the whitespace on screen IS the whitespace in the node, so it cannot flex. */
function eachMatch(
  index: FindTextIndex,
  needle: string,
  visit: (start: number, end: number) => boolean,
): void {
  const variants = canonicalVariants(
    needle,
    index.text.includes(COMBINING_DOT),
  );
  // Against the SHORTEST spelling: a decomposed query is longer than the text it is meant to find.
  if (
    Math.min(...variants.map((variant) => variant.length)) > index.text.length
  )
    return;
  // "Spelt as typed" survives NFC, so a hit that only flexed a space is still told apart.
  const composedNeedle = needle.normalize("NFC");
  const asTyped = (hit: string): boolean =>
    variants.includes(hit) || hit.normalize("NFC") === composedNeedle;
  const pattern = matchPattern(variants, needle);
  if (pattern) {
    for (;;) {
      const hit = pattern.exec(index.text);
      if (hit === null) return;
      const end = hit.index + hit[0].length;
      // Never part way through a grapheme, whichever way the match was found.
      if (!alignsToGraphemes(index.text, hit.index, end)) {
        pattern.lastIndex = hit.index + 1;
        continue;
      }
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
    const end = at + needle.length;
    if (!alignsToGraphemes(index.text, at, end)) {
      from = at + 1;
      continue;
    }
    if (!visit(at, end)) return;
    from = end;
  }
}

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

/** At most `limit` matches, as a window around `anchor`: keeping the first `limit` instead keeps
 *  only the top of the document. `anchor` may be a thunk, because working it out reads layout. */
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

  // The count stops early: `total` only keeps the window off the end, and once `total - limit`
  // reaches the left edge it can no longer pull it back. Past the anchor `before` is final.
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
  const start = Math.min(
    Math.max(before - (limit >> 1), 0),
    Math.max(total - limit, 0),
  );
  return start === 0 ? head : collectMatches(index, needle, limit, start);
}

/** Drop the one match asked for over the cap, from whichever end the reader is further from. Taking
 *  it off the tail is right only while the window starts at the top of the document. */
export function dropProbeFurthestFrom(
  matches: FindMatch[],
  anchor: number | null,
  limit = MAX_MATCHES,
): void {
  // No anchor means the window starts at the document's first match, with nothing above to give up.
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

export interface TextPosition {
  node: FindTextNodeLike;
  offset: number;
}

export function startPositionAt(
  segments: TextSegment[],
  offset: number,
): TextPosition | null {
  const index = segmentAt(segments, offset);
  if (index === -1) return null;
  const segment = segments[index];
  return { node: segment.node, offset: offset - segment.start };
}

/** Located from the match's last character: an exclusive end sits one past the run whenever the
 *  match finishes a text node, which is the boundary `setEnd` wants there. */
export function endPositionAt(
  segments: TextSegment[],
  end: number,
): TextPosition | null {
  const index = segmentAt(segments, end - 1);
  if (index === -1) return null;
  const segment = segments[index];
  return { node: segment.node, offset: end - segment.start };
}
