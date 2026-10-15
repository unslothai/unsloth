// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure module, so the Node tests can use a hand-rolled DOM.

import { FIND_SKIP_ATTRIBUTE } from "./find-attributes.ts";

export const ELEMENT_NODE = 1;
export const TEXT_NODE = 3;

/** NUL: no typed query can contain it, and `normalizeQuery` rejects pasted ones. */
export const BLOCK_SEPARATOR = "\u0000";

export const MAX_INDEX_CHARS = 4_000_000;

export const MAX_MATCHES = 5_000;

/** A log arrives as one text node and would otherwise spend the whole budget on its own. */
export const MAX_NODE_CHARS = 100_000;

export const PORTAL_RESERVE_CHARS = 100_000;

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

/** Each is one UTF-16 unit, keeping the offset map valid. */
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

interface IndexedSurface {
  root: FindElementLike;
  start: number;
  end: number;
}

export interface FindTextIndex {
  text: string;
  segments: TextSegment[];
  truncated: boolean;
  /** Offsets where text was dropped; a match may not end on one. */
  seams: ReadonlySet<number>;
  rootLength: number;
  surfaces: IndexedSurface[];
}

export const EMPTY_TEXT_INDEX: FindTextIndex = {
  text: "",
  segments: [],
  truncated: false,
  seams: new Set<number>(),
  rootLength: 0,
  surfaces: [],
};

/** The only code point whose `toLowerCase` grows; folded to `i` to keep lengths equal. */
const DOTTED_I_PATTERN = /\u0130/g;

/** Mapped to medial sigma as CaseFolding.txt does, so both spellings match. */
const FINAL_SIGMA_PATTERN = /\u03c2/g;

export function foldText(raw: string): string {
  const spaced = raw
    .replace(HARD_SPACE_PATTERN, " ")
    .replace(DOTTED_I_PATTERN, "i");
  const folded = spaced.toLowerCase();
  if (folded.length === spaced.length) {
    return folded.replace(FINAL_SIGMA_PATTERN, "\u03c3");
  }
  let plain = "";
  for (const point of spaced) {
    const lower = point.toLowerCase();
    plain += lower.length === point.length ? lower : point;
  }
  return plain.replace(FINAL_SIGMA_PATTERN, "\u03c3");
}

function hasClassToken(element: FindElementLike, token: string): boolean {
  return (element.getAttribute("class") ?? "").split(/\s+/).includes(token);
}
function skipsByMarkup(element: FindElementLike): boolean {
  // Uppercased: SVG and MathML keep their source casing.
  if (SKIP_TAGS.has(element.tagName.toUpperCase())) return true;
  if (hasClassToken(element, "katex-mathml")) return true;
  if (element.getAttribute(FIND_SKIP_ATTRIBUTE) !== null) return true;
  // KaTeX's painted HTML is deliberately aria-hidden, so it is excepted.
  if (element.getAttribute("hidden") !== null) return true;
  if (element.getAttribute("inert") !== null) return true;
  return (
    element.getAttribute("aria-hidden") === "true" &&
    !hasClassToken(element, "katex-html")
  );
}

export function skipsSubtree(
  element: FindElementLike,
  style: ResolvedStyle | null = computedStyle(element),
): boolean {
  if (skipsByMarkup(element)) return true;
  // contentVisibilityAuto and opacity off; both spellings since older Chrome and Firefox
  // read only the old names.
  const painted = element.checkVisibility?.({
    contentVisibilityAuto: false,
    opacityProperty: false,
    checkOpacity: false,
    visibilityProperty: true,
    checkVisibilityCSS: true,
  });
  if (painted === false) {
    // Shell wrappers are `display: contents`, which checkVisibility calls invisible.
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

/** `visibility` inherits and only element children are re-checked, so check own text. */
function hidesOwnText(style: ResolvedStyle | null): boolean {
  return (
    style?.display === "contents" &&
    (style.visibility === "hidden" || style.visibility === "collapse")
  );
}

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

/** `pre-line` is excluded: it still collapses runs of spaces. */
function preservesWhitespace(whiteSpace: string | undefined): boolean {
  return (
    whiteSpace === "pre" ||
    whiteSpace === "pre-wrap" ||
    whiteSpace === "break-spaces"
  );
}

/** Recursive, not a TreeWalker, which cannot report leaving an element. */
export function buildTextIndex(
  root: FindElementLike,
  extraRoots: readonly FindElementLike[] = [],
): FindTextIndex {
  const parts: string[] = [];
  const segments: TextSegment[] = [];

  const surfaces: IndexedSurface[] = [];
  const seams = new Set<number>();
  let tail = "";
  let dropped: string | null = null;
  let length = 0;
  let truncated = false;
  let full = false;
  let ceiling =
    MAX_INDEX_CHARS - (extraRoots.length > 0 ? PORTAL_RESERVE_CHARS : 0);
  // Written lazily, so no separator lands at either end.
  let pendingSeparator = false;

  const visit = (element: FindElementLike, inherited: boolean): void => {
    if (skipsByMarkup(element)) return;
    const style = computedStyle(element);
    if (skipsSubtree(element, style)) return;
    const block =
      BLOCK_TAGS.has(element.tagName) || isBlockDisplay(style?.display);
    if (block) {
      pendingSeparator = true;
      dropped = null;
    }
    const preserved =
      style?.whiteSpace === undefined
        ? inherited
        : preservesWhitespace(style.whiteSpace);
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
        // Before the separator: past the ceiling `take` goes negative and slice misbehaves.
        if (length >= ceiling) {
          truncated = true;
          if (!pendingSeparator && joinsAcross(tail, firstPointOf(data))) {
            seams.add(length);
          }
          full = true;
          return;
        }
        let separated = false;
        if (pendingSeparator) {
          pendingSeparator = false;
          if (length > 0) {
            parts.push(BLOCK_SEPARATOR);
            length += 1;
            tail = BLOCK_SEPARATOR;
            separated = true;
          }
        }
        if (dropped !== null) {
          if (joinsAcross(dropped, firstPointOf(data))) seams.add(length);
          dropped = null;
        }
        // A share, not all: one huge node must not starve everything after it.
        let take = Math.min(ceiling - length, MAX_NODE_CHARS);
        // Never cut between surrogate halves; drop the lead one so the cut is a real boundary.
        if (take > 0 && take < data.length && isPairedHalf(data, take))
          take -= 1;
        if (take <= 0) {
          truncated = true;
          if (!separated && joinsAcross(tail, firstPointOf(data))) {
            seams.add(length);
          }
          full = true;
          return;
        }
        const raw = data.length > take ? data.slice(0, take) : data;
        parts.push(raw);
        segments.push({ node, start: length, length: raw.length, preserved });
        length += raw.length;
        tail = (tail + raw).slice(-JOIN_CONTEXT);
        // Dropped text must leave a boundary, or a match across the seam paints over the gap.
        if (raw.length < data.length) {
          truncated = true;
          if (joinsAcross(raw, firstPointOf(data.slice(take))))
            seams.add(length);
          dropped = data.slice(-JOIN_CONTEXT);
          pendingSeparator = true;
        }
      } else if (child.nodeType === ELEMENT_NODE) {
        visit(child as FindElementLike, preserved);
        if (full) return;
      }
    }
    if (block) {
      pendingSeparator = true;
      dropped = null;
    }
  };

  visit(root, false);
  // `foldText` cannot change a length, so this offset survives it.
  const rootLength = length;
  // Hand portals the reserve, or the surface in front of the reader is the one left out.
  ceiling = MAX_INDEX_CHARS;
  full = false;
  for (const extra of extraRoots) {
    if (full) break;
    pendingSeparator = true;
    dropped = null;
    const firstSegment = segments.length;
    visit(extra, false);
    if (segments.length > firstSegment) {
      surfaces.push({
        root: extra,
        start: segments[firstSegment].start,
        end: length,
      });
    }
  }
  // Folded once over the joined document; see foldText.
  return {
    text: foldText(parts.join("")),
    segments,
    truncated,
    seams,
    rootLength,
    surfaces,
  };
}

/** True when a rebuild moved text ahead of the reader's offset, so search re-anchors.
 *  Kept here, not in the hook, so it runs under `node --test`. */
export function renumbersMatches(
  before: FindTextIndex,
  after: FindTextIndex,
  activeStart: number | null,
): boolean {
  const workspaceGrewAtTail =
    before.rootLength <= after.rootLength &&
    after.text.startsWith(before.text.slice(0, before.rootLength));
  // The workspace is the prefix, so no surface can move an offset inside it.
  if (activeStart === null || activeStart < before.rootLength) {
    return !workspaceGrewAtTail;
  }
  // A stable surface root tells whether a portal was inserted or reordered ahead.
  const beforeSurface = before.surfaces.find(
    ({ start, end }) => start <= activeStart && activeStart < end,
  );
  if (beforeSurface !== undefined) {
    const afterSurface = after.surfaces.find(
      ({ root }) => root === beforeSurface.root,
    );
    return (
      !workspaceGrewAtTail ||
      afterSurface === undefined ||
      afterSurface.start !== beforeSurface.start
    );
  }
  return !workspaceGrewAtTail || after.rootLength !== before.rootLength;
}

export function normalizeQuery(query: string): string | null {
  if (query.length === 0) return null;
  const folded = foldText(query);
  if (folded.includes(BLOCK_SEPARATOR)) return null;
  return folded;
}

export interface FindMatch {
  start: number;
  end: number;
}

const REGEX_META_PATTERN = /[.*+?^${}()|[\]\\]/g;

const COMBINING_DOT = "̇";

/** Longest first. The index itself is never normalized, to keep offsets 1:1. */
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

/** Hangul L*V*T* runs must win before the generic character alternative. */
const CLUSTER_PATTERN =
  // biome-ignore lint/suspicious/noMisleadingCharacterClass: Jamo and combining marks intentionally form canonical clusters.
  /(?:[ᄀ-ᅟꥠ-꥿]+[ᅠ-ᆧힰ-ퟆ]+[ᆨ-ᇿퟋ-ퟻ]*|[\s\S])[̀-ͯ҃-҉᪰-᫿᷀-᷿⃐-⃰︠-︯]*/gu;

const OPEN_HANGUL_CLUSTER_PATTERN =
  /^[\u1100-\u115f\ua960-\ua97f]+[\u1160-\u11a7\ud7b0-\ud7c6]+$/u;
const CLOSED_HANGUL_CLUSTER_PATTERN =
  /^[\u1100-\u115f\ua960-\ua97f]+[\u1160-\u11a7\ud7b0-\ud7c6]+[\u11a8-\u11ff\ud7cb-\ud7fb]+$/u;
const VOWEL_OR_TRAILING_HANGUL_JAMO_SOURCE =
  "[\\u1160-\\u11a7\\ud7b0-\\ud7c6\\u11a8-\\u11ff\\ud7cb-\\ud7fb]";
const TRAILING_HANGUL_JAMO_SOURCE = "[\\u11a8-\\u11ff\\ud7cb-\\ud7fb]";

const HANGUL_HINT_PATTERN = /[ᄀ-ᇿꥠ-꥿가-ퟻ]/u;

/** Old and Extended Hangul jamo have one spelling, so force the pattern path to get the
 * trailing-jamo boundary; otherwise an open syllable prefix-matches a closed one. */
function needsHangulBoundary(needle: string): boolean {
  if (!HANGUL_HINT_PATTERN.test(needle)) return false;
  for (const [cluster] of needle.normalize("NFD").matchAll(CLUSTER_PATTERN)) {
    if (
      OPEN_HANGUL_CLUSTER_PATTERN.test(cluster) ||
      CLOSED_HANGUL_CLUSTER_PATTERN.test(cluster)
    )
      return true;
  }
  return false;
}

const HANGUL_LVT_PATTERN =
  /^([\u1100-\u115f\ua960-\ua97f][\u1160-\u11a7\ud7b0-\ud7c6])([\u11a8-\u11ff\ud7cb-\ud7fb][\s\S]*)$/u;

/** L+V precomposed with a loose trailing jamo; neither NFC nor NFD writes this form. */
function partiallyComposedHangul(cluster: string): string | null {
  const parts = HANGUL_LVT_PATTERN.exec(cluster);
  if (!parts) return null;
  const partial = parts[1].normalize("NFC") + parts[2];
  return partial === cluster ? null : partial;
}

/** A lone low surrogate is a character in its own right, so only a real pair counts. */
function isPairedHalf(text: string, at: number): boolean {
  const low = text.charCodeAt(at);
  if (!(low >= 0xdc00 && low <= 0xdfff) || at === 0) return false;
  const high = text.charCodeAt(at - 1);
  return high >= 0xd800 && high <= 0xdbff;
}

/** Per cluster: joined text nodes can mix composed and decomposed spellings in one word. */
function canonicalSource(needle: string, dotted: boolean): string {
  let out = "";
  for (const [cluster] of needle.normalize("NFD").matchAll(CLUSTER_PATTERN)) {
    if (/^\s/.test(cluster)) {
      // Only one `\s+` per run, so marks attached to the last space still follow it.
      out += out.endsWith("\\s+") ? "" : "\\s+";
      out += escapeForRegex(cluster.slice(1));
      continue;
    }
    const spellings = [cluster];
    const composed = cluster.normalize("NFC");
    if (composed !== cluster) spellings.push(composed);
    // Joined text nodes can split Hangul after its first composition step.
    const partial = partiallyComposedHangul(cluster);
    if (partial !== null && !spellings.includes(partial))
      spellings.push(partial);
    // A decomposed dotted I folds to `i` plus a combining dot, which NFC cannot recompose.
    if (dotted && cluster === "i") spellings.push(`i${COMBINING_DOT}`);
    // Longest first: alternation takes the first fit, so a prefix spelling would end mid-grapheme.
    if (spellings.length > 1) spellings.sort((a, b) => b.length - a.length);
    const spellingSource =
      spellings.length === 1
        ? escapeForRegex(spellings[0])
        : `(?:${spellings.map(escapeForRegex).join("|")})`;
    const boundary = OPEN_HANGUL_CLUSTER_PATTERN.test(cluster)
      ? `(?!${VOWEL_OR_TRAILING_HANGUL_JAMO_SOURCE})`
      : CLOSED_HANGUL_CLUSTER_PATTERN.test(cluster)
        ? `(?!${TRAILING_HANGUL_JAMO_SOURCE})`
        : "";
    out += spellingSource + boundary;
  }
  return out;
}

/** Built lazily per index; `containing` seeks, so no full walk is needed. */
const segmentsCache = new WeakMap<FindTextIndex, GraphemeSegments>();

/** All boundaries, tabulated once seek time exceeds what a full scan would cost. Budgeted
 *  in time since seek cost varies by script (1.3M chars scanned in 82ms). */
const boundaryCache = new WeakMap<FindTextIndex, Uint8Array>();
const seekCosts = new WeakMap<
  FindTextIndex,
  { spent: number; since: number; seen?: number }
>();
const SCAN_CHARS_PER_MS = 16_000;
const MIN_SEEK_BUDGET_MS = 8;
/** Timed per block so the clock is read twice per block, not per seek. */
const SEEK_BLOCK = 32;

/** Drop a block left open between searches so idle time is not billed to the next query. */
function endSeekWindow(index: FindTextIndex): void {
  const cost = seekCosts.get(index);
  if (cost === undefined) return;
  cost.seen = (cost.seen ?? 0) - ((cost.seen ?? 0) % SEEK_BLOCK);
  cost.since = 0;
}

/** CRLF is the only pair below U+0300 that joins (GB3); seen intact inside `<pre>`. */
function splitsCrlf(text: string, at: number): boolean {
  return at > 0 && text.charCodeAt(at - 1) === 13 && text.charCodeAt(at) === 10;
}

const JOINS_GRAPHEME = /[^\u0000-\u02ff]/;

interface GraphemeSegments {
  containing(at: number): { index: number } | undefined;
  [Symbol.iterator](): IterableIterator<{ index: number }>;
}

let segmenter: { segment(input: string): GraphemeSegments } | null | undefined;

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

/** WebKit's `containing` answers the segment ending at the offset; there, tabulate instead. */
let seeksBoundaries: boolean | undefined;
function segmenterSeeksBoundaries(platform: {
  segment(input: string): GraphemeSegments;
}): boolean {
  if (seeksBoundaries !== undefined) return seeksBoundaries;
  try {
    const probe = platform.segment("x\u{1f44d}");
    seeksBoundaries =
      probe.containing(1)?.index === 1 && probe.containing(0)?.index === 0;
  } catch {
    seeksBoundaries = false;
  }
  return seeksBoundaries;
}

const JOIN_CONTEXT = 32;

/** Back to a char below U+0300, which always begins a cluster; regional indicators pair from
 *  the start of a run, so a fixed window would not do. */
function exactTail(before: string): string | null {
  const from = Math.max(0, before.length - JOIN_CONTEXT);
  for (let at = before.length - 1; at >= from; at -= 1) {
    if (before.charCodeAt(at) < 0x300) return before.slice(at);
  }
  return null;
}

function firstPointOf(text: string): string {
  return text.length === 0
    ? ""
    : String.fromCodePoint(text.codePointAt(0) as number);
}

/** False without a segmenter, keeping the old behaviour. */
function joinsAcross(before: string, point: string): boolean {
  if (before.length === 0 || point.length === 0) return false;
  const platform = graphemeSegmenter();
  if (platform === null) return false;
  const window = exactTail(before);
  if (window === null) return true;
  const at = window.length;
  const body = window + point;
  if (segmenterSeeksBoundaries(platform)) {
    return platform.segment(body).containing(at)?.index !== at;
  }
  for (const { index: start } of platform.segment(body)) {
    if (start === at) return false;
    if (start > at) break;
  }
  return true;
}

/** Asks the platform segmenter one offset at a time; hand-rolled UAX 29 kept missing cases. */
function alignsToGraphemes(
  index: FindTextIndex,
  start: number,
  end: number,
): boolean {
  const text = index.text;
  // Before the cheap test: at the end of a truncated index there is no `text[end]` to read.
  if (
    index.seams.size > 0 &&
    (index.seams.has(start) || index.seams.has(end))
  ) {
    return false;
  }
  // Fast path: nothing below U+0300 joins a grapheme, so skip the segmenter there.
  if (
    !splitsCrlf(text, start) &&
    !splitsCrlf(text, end) &&
    !(start > 0 && JOINS_GRAPHEME.test(text[start - 1])) &&
    !JOINS_GRAPHEME.test(text[start]) &&
    !JOINS_GRAPHEME.test(text[end - 1]) &&
    !(end < text.length && JOINS_GRAPHEME.test(text[end]))
  ) {
    return true;
  }
  return startsGrapheme(index, start) && startsGrapheme(index, end);
}

/** True with no segmenter (Firefox before 125) rather than hand-rolling UAX 29. */
function startsGrapheme(index: FindTextIndex, at: number): boolean {
  const text = index.text;
  if (at === 0 || at === text.length) return true;
  const platform = graphemeSegmenter();
  if (platform === null) return true;
  const marked = boundaryCache.get(index);
  if (marked !== undefined) return marked[at] === 1;
  let segments = segmentsCache.get(index);
  if (segments === undefined) {
    segments = platform.segment(text);
    segmentsCache.set(index, segments);
  }
  let cost = seekCosts.get(index);
  if (cost === undefined) {
    cost = { spent: 0, since: 0 };
    seekCosts.set(index, cost);
  }
  const budget = Math.max(MIN_SEEK_BUDGET_MS, text.length / SCAN_CHARS_PER_MS);
  if (segmenterSeeksBoundaries(platform) && cost.spent <= budget) {
    if (cost.since === 0) cost.since = performance.now();
    const answer = segments.containing(at)?.index === at;
    cost.seen = (cost.seen ?? 0) + 1;
    if (cost.seen % SEEK_BLOCK === 0) {
      cost.spent += performance.now() - cost.since;
      cost.since = 0;
    }
    return answer;
  }
  // Past the budget, tabulate once: a capped search can walk candidates up to three times.
  const marks = new Uint8Array(text.length + 1);
  for (const { index: start } of segments) marks[start] = 1;
  marks[text.length] = 1;
  boundaryCache.set(index, marks);
  return marks[at] === 1;
}

function escapeForRegex(text: string): string {
  return text.replace(REGEX_META_PATTERN, "\\$&");
}

/** Null for a plain scan. Whitespace flexes for soft-wrapped text; the separator does not. */
function matchPattern(variants: string[], needle: string): RegExp | null {
  const dotted = variants.some((variant) => variant.includes(COMBINING_DOT));
  if (
    variants.length === 1 &&
    !/\s/.test(needle) &&
    !needsHangulBoundary(needle)
  )
    return null;
  try {
    const pattern = new RegExp(canonicalSource(needle, dotted), "g");
    // V8 compiles lazily, so an oversized pattern is accepted here and throws on the first `exec`,
    // outside this `try`. One run against nothing forces the compile while it is catchable.
    pattern.exec("");
    return pattern;
  } catch {
    // Engines cap pattern size with no spec limit; a pasted log can hit it and must not throw.
    return null;
  }
}

/** Non-overlapping like browser find. Whitespace in `<pre>` cannot flex. */
function eachMatch(
  index: FindTextIndex,
  needle: string,
  visit: (start: number, end: number) => boolean,
): void {
  const variants = canonicalVariants(
    needle,
    index.text.includes(COMBINING_DOT),
  );
  // Against the shortest spelling: a decomposed query is longer than the text it finds.
  if (
    Math.min(...variants.map((variant) => variant.length)) > index.text.length
  )
    return;
  const composedNeedle = needle.normalize("NFC");
  const asTyped = (hit: string): boolean =>
    variants.includes(hit) || hit.normalize("NFC") === composedNeedle;
  const pattern = matchPattern(variants, needle);
  if (pattern) {
    for (;;) {
      const hit = pattern.exec(index.text);
      if (hit === null) return;
      const end = hit.index + hit[0].length;
      if (!alignsToGraphemes(index, hit.index, end)) {
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
    if (!alignsToGraphemes(index, at, end)) {
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

/** A window of `limit` around `anchor`; `anchor` may be a thunk since it reads layout. */
export function findMatches(
  index: FindTextIndex,
  query: string,
  limit = MAX_MATCHES,
  anchor: number | (() => number) = 0,
): FindMatch[] {
  const needle = normalizeQuery(query);
  if (needle === null) return [];
  endSeekWindow(index);
  const head = collectMatches(index, needle, limit, 0);
  // Before resolving the anchor, so an under-cap query never pays for it.
  if (head.length < limit) return head;
  const at = typeof anchor === "function" ? anchor() : anchor;
  if (at <= 0) return head;

  // The count may stop early once the window can no longer move.
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

/** Drop the over-cap probe match from the end farther from the reader. */
export function dropProbeFurthestFrom(
  matches: FindMatch[],
  anchor: number | null,
  limit = MAX_MATCHES,
): void {
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
  // A match can open on a separator, which belongs to no segment; take the next one.
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

/** From the last char: an exclusive end at a node's end is the boundary `setEnd` wants. */
export function endPositionAt(
  segments: TextSegment[],
  end: number,
): TextPosition | null {
  const index = segmentAt(segments, end - 1);
  if (index === -1) return null;
  const segment = segments[index];
  return { node: segment.node, offset: end - segment.start };
}
