// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Flatten searchable text once and retain offsets back to its text nodes. The module is pure, so
// the Node tests can use a hand-rolled DOM.

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

/** Reserve space for visible portaled surfaces after walking the workspace. */
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

interface IndexedSurface {
  root: FindElementLike;
  start: number;
  end: number;
}

export interface FindTextIndex {
  text: string;
  /** Sorted by `start`, gapped wherever a separator was written. */
  segments: TextSegment[];
  truncated: boolean;
  /** Where the portaled surfaces begin. The whole length when none are open. */
  rootLength: number;
  /** Stable portal roots and the offsets occupied by their searchable text. */
  surfaces: IndexedSurface[];
}

export const EMPTY_TEXT_INDEX: FindTextIndex = {
  text: "",
  segments: [],
  truncated: false,
  rootLength: 0,
  surfaces: [],
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

/** Check cheap markup before resolving styles, which avoids layout work for skipped subtrees. */
function hasClassToken(element: FindElementLike, token: string): boolean {
  return (element.getAttribute("class") ?? "").split(/\s+/).includes(token);
}
function skipsByMarkup(element: FindElementLike): boolean {
  // Uppercased: SVG and MathML keep their source casing, so `<svg>` answers "svg" and walks past.
  if (SKIP_TAGS.has(element.tagName.toUpperCase())) return true;
  if (hasClassToken(element, "katex-mathml")) return true;
  if (element.getAttribute(FIND_SKIP_ATTRIBUTE) !== null) return true;
  // Boolean attributes, so presence is the whole signal. The shell parks an off-route workspace
  // under `inert`; Radix marks the page `aria-hidden` behind a modal. KaTeX is the narrow exception:
  // its painted HTML tree is deliberately aria-hidden because a clipped MathML mirror speaks it.
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

  const surfaces: IndexedSurface[] = [];
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
        let take = Math.min(ceiling - length, MAX_NODE_CHARS);
        // Never between the halves of a pair: keeping the leading one leaves a code unit that is
        // not a character, which reads as a grapheme of its own and lets a match end against it,
        // inside what the page draws as one. Dropping it instead makes the cut a real boundary.
        if (take > 0 && take < data.length && isPairedHalf(data, take))
          take -= 1;
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
  // Before the surfaces and the separator joining them, so this slice is exactly the workspace.
  // `foldText` cannot change a length, so the offset survives it.
  const rootLength = length;
  // The reserve, handed over. Portaled surfaces come after the workspace, so without this the one
  // thing in front of the reader is the one left out. `truncated` stays as the workspace left it.
  ceiling = MAX_INDEX_CHARS;
  full = false;
  for (const extra of extraRoots) {
    if (full) break;
    // Its own surface, so a boundary whatever the last root ended on, and whatever tags either of
    // them happen to use: nothing dropped back there can reach into what a portal paints.
    pendingSeparator = true;
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
  // Folded once, over the joined document: see foldText for why it cannot be done a node at a time.
  return {
    text: foldText(parts.join("")),
    segments,
    truncated,
    rootLength,
    surfaces,
  };
}

/**
 * True when a rebuild renumbers the match list, so the search has to re-anchor to the viewport.
 *
 * The index is the workspace followed by the surfaces portaled in front of it, joined at
 * `rootLength`. A monitor stays searchable while it is up and rewrites its reading on a timer, so
 * judged as one string every poll reads as a renumbered document and throws the reader out of the
 * conversation behind it. What decides is whether text moved AHEAD of the reader's offset.
 *
 * This is the cheap half of the answer. `search` asks the exact question afterwards, by looking for
 * a match still starting at that offset, so this only has to turn down the rebuilds where the
 * offset would be meaningless.
 *
 * Here rather than beside its caller: the hook imports React and cannot run under `node --test`.
 */
export function renumbersMatches(
  before: FindTextIndex,
  after: FindTextIndex,
  /** Where the reader's occurrence started in `before`, or null when there was none. */
  activeStart: number | null,
): boolean {
  // One slice, and none at all without a surface open, where the workspace is the whole text.
  const workspaceGrewAtTail =
    before.rootLength <= after.rootLength &&
    after.text.startsWith(before.text.slice(0, before.rootLength));
  // The workspace is the prefix, so no surface can move an offset inside it. `search` re-anchors on
  // its own when there was no occurrence to keep.
  if (activeStart === null || activeStart < before.rootLength) {
    return !workspaceGrewAtTail;
  }
  // A stable surface root tells whether another portal was inserted, removed, or reordered ahead of
  // the occurrence. Equal-width polling within that root keeps its start and therefore the reader.
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

/**
 * One canonical cluster. Hangul conjoining sequences can contain repeated leading, vowel, and
 * trailing Jamo, so the whole L*V*T* run has to win before the generic character alternative.
 */
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

/** Any Hangul at all, asked first so an ASCII query pays one test and stops. */
const HANGUL_HINT_PATTERN = /[ᄀ-ᇿꥠ-꥿가-ퟻ]/u;

/**
 * True when a cluster needs the trailing-jamo boundary, which only the pattern path writes.
 *
 * Extended and Old Hangul jamo have no precomposed form, so NFC and NFD spell them the same way and
 * the single-spelling query would take the literal scan, where an open syllable prefix-matches a
 * closed one. Modern Hangul always has two spellings and reaches the pattern anyway.
 */
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

/** A closed syllable's L+V pair and everything after it. */
const HANGUL_LVT_PATTERN =
  /^([\u1100-\u115f\ua960-\ua97f][\u1160-\u11a7\ud7b0-\ud7c6])([\u11a8-\u11ff\ud7cb-\ud7fb][\s\S]*)$/u;

/**
 * The half-composed spelling of a closed syllable: the L+V pair precomposed, the trailing jamo left
 * as it is. Neither NFC nor NFD writes it, so a document holding one is invisible to both.
 */
function partiallyComposedHangul(cluster: string): string | null {
  const parts = HANGUL_LVT_PATTERN.exec(cluster);
  if (!parts) return null;
  const partial = parts[1].normalize("NFC") + parts[2];
  return partial === cluster ? null : partial;
}

/** The second half of a surrogate pair, and only where there is a pair: a low surrogate on its own
 *  is a character in its own right, and one can reach a page through JSON or a pasted log. Taking
 *  it for half of something joined it to whatever preceded it, hiding both. */
function isPairedHalf(text: string, at: number): boolean {
  const low = text.charCodeAt(at);
  if (!(low >= 0xdc00 && low <= 0xdfff) || at === 0) return false;
  const high = text.charCodeAt(at - 1);
  return high >= 0xd800 && high <= 0xdbff;
}

/**
 * Per cluster, because alternating whole spellings of the WHOLE query reaches only all-composed or
 * all-decomposed text, and one occurrence can be neither: joining two text nodes joins two sources,
 * so `café` in one and `café` in the next make one visible word with a spelling the
 * query cannot be written in. Every engine's own find matches it.
 */
function canonicalSource(needle: string, dotted: boolean): string {
  let out = "";
  for (const [cluster] of needle.normalize("NFD").matchAll(CLUSTER_PATTERN)) {
    if (/^\s/.test(cluster)) {
      // The space flexes, what is attached to it does not: a mark on a space is part of that
      // grapheme, and dropping it left the match ending inside one, which the fence then threw
      // away. Only one `\s+` for a run of them, so the marks of the last still follow it.
      out += out.endsWith("\\s+") ? "" : "\\s+";
      out += escapeForRegex(cluster.slice(1));
      continue;
    }
    const spellings = [cluster];
    const composed = cluster.normalize("NFC");
    if (composed !== cluster) spellings.push(composed);
    // Hangul composes in two steps, and text can stop after the first. Joining two text nodes
    // produces exactly that, an LV syllable in one and its trailing Jamo in the next.
    const partial = partiallyComposedHangul(cluster);
    if (partial !== null && !spellings.includes(partial))
      spellings.push(partial);
    // A decomposed dotted I folds to `i` plus a combining dot, which has no precomposed form, so
    // NFC cannot put it back and the plain query would miss a word plainly on screen.
    if (dotted && cluster === "i") spellings.push(`i${COMBINING_DOT}`);
    // Longest first, as `canonicalVariants` is: alternation takes the first that fits, so a short
    // spelling that is a prefix of a long one wins and the rest of the cluster is left outside the
    // match: `i` before `i` plus its combining dot ended the match inside the grapheme, and the
    // boundary check threw the occurrence away rather than reaching for the longer spelling.
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

/** The index's segmentation, made at the first match that needs one and kept for as long as the
 *  index lives. Making it walks nothing: `containing` seeks to the offset asked about, so a 4M
 *  index costs 4ms once and a fraction of a microsecond a question. */
const segmentsCache = new WeakMap<FindTextIndex, GraphemeSegments>();

/** Every boundary in the index, once seeking for them has cost more than walking the lot would.
 *
 *  In time, not in seeks: a seek is 0.2us into a page of Hangul and 1236us into a page of flags,
 *  so any count is far too small for one and far too large for the other, and the large end cost
 *  a first search twenty seconds. The budget is what a scan of this index would itself cost, so
 *  the total stays within twice the better of the two and no constant is left to be wrong about.
 *  The rate is the slower of the two measured, 1.3M characters scanned in 82ms. */
const boundaryCache = new WeakMap<FindTextIndex, Uint8Array>();
const seekCosts = new WeakMap<
  FindTextIndex,
  { spent: number; since: number; seen?: number }
>();
const SCAN_CHARS_PER_MS = 16_000;
const MIN_SEEK_BUDGET_MS = 8;
/** Timed in blocks, so the clock is read twice per block rather than twice per seek. The whole
 *  block is measured, so what accumulates is the real cost and not a sample of it. */
const SEEK_BLOCK = 32;

/** Drop a block left open by a search that ended inside one: it is wall time, and between two
 *  searches that is the reader thinking, which was being billed to the next query. What is lost
 *  is under a block of seeks, which the budget will not miss. */
function endSeekWindow(index: FindTextIndex): void {
  const cost = seekCosts.get(index);
  if (cost === undefined) return;
  cost.seen = (cost.seen ?? 0) - ((cost.seen ?? 0) % SEEK_BLOCK);
  cost.since = 0;
}

/** The one thing below U+0300 that joins (GB3), and so the one exception to the fast path below.
 *  Everywhere whitespace is collapsed a newline is its own grapheme; in a `<pre>` the pair arrives
 *  intact and a search for the feed alone landed between them. */
function splitsCrlf(text: string, at: number): boolean {
  return at > 0 && text.charCodeAt(at - 1) === 13 && text.charCodeAt(at) === 10;
}

/** Anything that could extend or be extended into a grapheme. See `alignsToGraphemes`. */
const JOINS_GRAPHEME = /[^\u0000-\u02ff]/;

interface GraphemeSegments {
  containing(at: number): { index: number } | undefined;
  [Symbol.iterator](): IterableIterator<{ index: number }>;
}

let segmenter: { segment(input: string): GraphemeSegments } | null | undefined;

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
 * Asked of the platform, which knows the whole of UAX 29 and is kept current with it; enumerating
 * the ranges here kept missing one more way to land inside a cluster every round. Asked one offset
 * at a time, so neither the size of the index nor where the match landed in it costs anything:
 * tabulating a block's boundaries instead paid 250ms per block, and paid it again on every reindex.
 */
function alignsToGraphemes(
  index: FindTextIndex,
  start: number,
  end: number,
): boolean {
  const text = index.text;
  // Almost every match is in text that cannot join at either edge, and asking the segmenter costs
  // far more than looking. Nothing below U+0300 joins: the lowest combining mark is U+0300, the
  // lowest spacing mark U+0903, Prepend starts at U+0600, Hangul Jamo at U+1100, and everything
  // astral arrives as a surrogate. Both sides of each edge, since the query can end with one.
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

/**
 * True when a grapheme starts at `at`.
 *
 * True as well where there is no segmenter to ask, which is every candidate taken unchecked, as
 * before this fence existed. Firefox shipped `Intl.Segmenter` in 125 and is the last engine to;
 * enumerating UAX 29 by hand for the ones behind it is a second implementation to keep current
 * with Unicode, and the alternative is only that those readers keep today's behaviour.
 */
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
  if (cost.spent <= budget) {
    if (cost.since === 0) cost.since = performance.now();
    const answer = segments.containing(at)?.index === at;
    cost.seen = (cost.seen ?? 0) + 1;
    if (cost.seen % SEEK_BLOCK === 0) {
      cost.spent += performance.now() - cost.since;
      cost.since = 0;
    }
    return answer;
  }
  // Past the budget, and a capped search anchored near the end walks the candidates up to three
  // times, so this is bought once and answers every question after it.
  const marks = new Uint8Array(text.length + 1);
  for (const { index: start } of segments) marks[start] = 1;
  marks[text.length] = 1;
  boundaryCache.set(index, marks);
  return marks[at] === 1;
}

function escapeForRegex(text: string): string {
  return text.replace(REGEX_META_PATTERN, "\\$&");
}

/** Null for a plain scan. Whitespace flexes because a soft-wrapped paragraph renders as one line
 *  while its node holds the newline; the separator is not whitespace, so blocks stay shut. */
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
  endSeekWindow(index);
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
