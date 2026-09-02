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
  /** Offsets where the walk left something out, so nothing here can say whether a grapheme
   *  carries on across them. A cut in the middle of the document leaves two, on each side of the
   *  separator standing in for what was dropped; one at the end of the walk leaves one. */
  unsafe: ReadonlySet<number>;
}

export const EMPTY_TEXT_INDEX: FindTextIndex = {
  text: "",
  segments: [],
  truncated: false,
  unsafe: new Set(),
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
  /** See `FindTextIndex.unsafe`. */
  const unsafe = new Set<number>();
  /** Whether the separator now due stands for dropped text rather than a block boundary. */
  let pendingClip = false;
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
          // A separator was already due before this text, so what is being left out is behind a
          // break and cannot join what the index ends on: that end is still a boundary.
          if (!pendingSeparator) unsafe.add(length);
          full = true;
          return;
        }
        let separated = false;
        if (pendingSeparator) {
          pendingSeparator = false;
          if (length > 0) {
            parts.push(BLOCK_SEPARATOR);
            length += 1;
            separated = true;
            // The far side of a separator that stands for dropped text is no safer than the near
            // side: this node begins where something was cut, and may carry on from it.
            if (pendingClip) unsafe.add(length);
          }
          pendingClip = false;
        }
        // A share, not all: one huge node given the rest leaves out everything after it.
        const take = Math.min(ceiling - length, MAX_NODE_CHARS);
        if (take <= 0) {
          truncated = true;
          if (!separated) unsafe.add(length);
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
          unsafe.add(length);
          pendingSeparator = true;
          pendingClip = true;
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
  return { text: foldText(parts.join("")), segments, truncated, unsafe };
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
/** A leading Jamo, and the vowel that makes it a syllable. */
const HANGUL_LEADING_PATTERN = /[\u1100-\u115f\ua960-\ua97c]/;
const HANGUL_VOWEL_PATTERN = /[\u1160-\u11a7\ud7b0-\ud7c6]/;
/** Characters that join to whatever follows them (UAX 29 Prepend). All 27 of them, most of which
 *  are supplementary: the BMP half alone let a match begin inside an Indic or Kaithi cluster. */
const PREPEND_PATTERN =
  /[\u{600}-\u{605}\u{6dd}\u{70f}\u{890}\u{891}\u{8e2}\u{d4e}\u{110bd}\u{110cd}\u{111c2}\u{111c3}\u{113d1}\u{1193f}\u{11941}\u{11a84}-\u{11a89}\u{11d46}\u{11f02}]/u;
/** Breaks on both sides of itself, whatever that is (GB4/GB5). `BLOCK_SEPARATOR` is one, which is
 *  what makes searching across blocks safe. */
const CONTROL_PATTERN = /[\u0000-\u001f\u007f-\u009f]/;
/** Never begins a grapheme: Extend and ZWJ (GB9), SpacingMark (GB9a), and the trailing half of a
 *  surrogate pair, which is not a character at all. UAX 29 derives Extend as Grapheme_Extend OR
 *  Emoji_Modifier, and a skin tone is only the second of those: it is `Sk`, so the marks alone
 *  left one showing as its own grapheme. */
const EXTEND_PATTERN = /[\p{Grapheme_Extend}\p{Emoji_Modifier}]/u;
const EXTENDS_LEFT_PATTERN =
  /[\p{Grapheme_Extend}\p{Emoji_Modifier}\p{Mc}\u200d]/u;

/** The viramas that join the consonant after them to the one before (GB9c). Not derivable from a
 *  property escape, which has no `InCB`, so this is the set the segmenter itself joins on: every
 *  `ccc=9` code point for which a consonant, that virama and a consonant make one cluster. */
const LINKER_PATTERN =
  /[\u{94d}\u{9cd}\u{acd}\u{b4d}\u{c4d}\u{d4d}\u{e3a}\u{eba}\u{1039}\u{17d2}\u{1a60}\u{1b44}\u{1bab}\u{a9c0}\u{aaf6}\u{10a3f}\u{11133}\u{1193e}\u{11a47}\u{11a99}\u{11f42}]/u;

/** The code point ending at `end`, and where it starts. */
function pointBefore(text: string, end: number): [string, number] {
  const start = end - (isTrailingHalf(text.charCodeAt(end - 1)) ? 2 : 1);
  return [text.slice(start, end), start];
}

/** Walk back over extenders from `end` and say whether `found` matches what they sit on: GB11 wants
 *  a pictograph on the far side of the ZWJ, GB9c a linker on the far side of the marks. */
function reachesBack(text: string, end: number, found: RegExp): boolean {
  for (let at = end; at > 0; ) {
    const [point, start] = pointBefore(text, at);
    if (found.test(point)) return true;
    if (!EXTEND_PATTERN.test(point)) return false;
    at = start;
  }
  return false;
}

/** The second half of a surrogate pair. */
function isTrailingHalf(unit: number): boolean {
  return unit >= 0xdc00 && unit <= 0xdfff;
}
const REGIONAL_INDICATOR_PATTERN = /^[\u{1f1e6}-\u{1f1ff}]/u;
const PICTOGRAPHIC_PATTERN = /\p{Extended_Pictographic}/u;

/** L, V, T, and the precomposed syllables, which are LV when nothing trails and LVT when a Jamo
 *  does. Hangul fills 28 code points a syllable, the first of them the bare LV. */
function hangulClass(ch: string): string | null {
  if (HANGUL_LEADING_PATTERN.test(ch)) return "L";
  if (HANGUL_VOWEL_PATTERN.test(ch)) return "V";
  if (HANGUL_TRAILING_PATTERN.test(ch)) return "T";
  const cp = ch.charCodeAt(0);
  if (cp < 0xac00 || cp > 0xd7a3) return null;
  return (cp - 0xac00) % 28 === 0 ? "LV" : "LVT";
}

/** GB6 (L before anything but a trailing Jamo), GB7 and GB8. */
function hangulJoins(before: string, after: string | null): boolean {
  if (after === null) return false;
  if (before === "L") return after !== "T";
  if (before === "V" || before === "LV") return after === "V" || after === "T";
  return after === "T";
}

/**
 * Whether a grapheme carries on across `at`, for engines with no `Intl.Segmenter`: Firefox shipped
 * one only in 125, Vite's default target reaches back to 114, and ESR 115 is still in the field. On
 * those the alternative is taking every candidate unchecked, which is the defect this file exists
 * to fix.
 *
 * The rules this feature can actually run into, by Unicode property rather than by hand-listed
 * range: the joiners, Prepend, Hangul and regional indicator parity. The aksara and pictographic
 * rules are left to the engines that have a segmenter, so an Indic cluster can still be split here.
 */
function continuesGrapheme(text: string, at: number): boolean {
  // Whole code points, not code units: a property escape asked about half a surrogate pair sees a
  // lone surrogate and answers no, which is how a skin tone read as its own grapheme.
  if (isTrailingHalf(text.charCodeAt(at))) return true;
  const after = String.fromCodePoint(text.codePointAt(at) as number);
  const before = isTrailingHalf(text.charCodeAt(at - 1))
    ? text.slice(at - 2, at)
    : text[at - 1];
  if (CONTROL_PATTERN.test(before) || CONTROL_PATTERN.test(after)) return false;
  if (EXTENDS_LEFT_PATTERN.test(after)) return true;
  // GB11 proper, both sides: a ZWJ joins a pictograph to a pictograph, so an emoji sequence holds
  // together while a ZWJ merely following a letter still ends its cluster.
  if (before === "\u200d") {
    return (
      PICTOGRAPHIC_PATTERN.test(after) &&
      reachesBack(text, at - 1, PICTOGRAPHIC_PATTERN)
    );
  }
  if (PREPEND_PATTERN.test(before)) return true;
  // GB9c, as far as it can be told without the consonant sets: a linker behind, whatever it is that
  // follows. Wrong only by declining a boundary, never by cutting a cluster.
  if (reachesBack(text, at, LINKER_PATTERN)) return true;
  if (REGIONAL_INDICATOR_PATTERN.test(after)) {
    // A run pairs off from its start, so it is the count behind that decides (GB12/GB13). Two code
    // units apiece.
    let run = 0;
    while (
      REGIONAL_INDICATOR_PATTERN.test(
        text.slice(at - 2 - run * 2, at - run * 2),
      )
    ) {
      run += 1;
    }
    return run % 2 === 1;
  }
  const left = hangulClass(before);
  return left !== null && hangulJoins(left, hangulClass(after));
}

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
    // Longest first, as `canonicalVariants` is: alternation takes the first that fits, so a short
    // spelling that is a prefix of a long one wins and the rest of the cluster is left outside the
    // match. `i` before `i` plus its combining dot ended the match inside the grapheme, and the
    // boundary check then threw the occurrence away rather than reaching for the longer spelling.
    if (spellings.length > 1) spellings.sort((a, b) => b.length - a.length);
    out +=
      spellings.length === 1
        ? escapeForRegex(spellings[0])
        : `(?:${spellings.map(escapeForRegex).join("|")})`;
  }
  return out;
}

/** The index's segmentation, made at the first match that needs one and kept for as long as the
 *  index lives. Making it walks nothing: `containing` seeks to the offset asked about, so a 4M
 *  index costs 4ms once and a fraction of a microsecond a question, where segmenting a block to
 *  fill a table of its boundaries cost 250ms for every block a match landed in. */
const segmentsCache = new WeakMap<FindTextIndex, GraphemeSegments>();

/** Anything that could extend or be extended into a grapheme. See `alignsToGraphemes`. */
const JOINS_GRAPHEME = /[^\u0000-\u02ff]/;

interface GraphemeSegments {
  containing(at: number): { index: number } | undefined;
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
 * Asked of the platform rather than answered here. Seven rounds of review found seven ways to land
 * inside a cluster, each a Unicode range that had not been enumerated, and there is no end to that
 * list. The segmenter already knows the whole of UAX 29 and is kept current with it.
 *
 * Asked one offset at a time, not by segmenting anything: the segmentation seeks, so neither the
 * size of the index nor where in it the match landed costs anything. Segmenting whole blocks to
 * fill a table of their boundaries, which is what this did before, paid 250ms per block for the
 * one or two offsets a match actually asks about, and paid it again on every reindex.
 */
function alignsToGraphemes(
  index: FindTextIndex,
  start: number,
  end: number,
): boolean {
  const text = index.text;
  const cut =
    index.unsafe.size > 0 && (index.unsafe.has(start) || index.unsafe.has(end));
  // Almost every match is in text that cannot join at either edge, and asking the segmenter costs
  // far more than looking. Nothing below U+0300 extends a grapheme: the lowest combining mark is
  // U+0300, the lowest spacing mark U+0903, Prepend starts at U+0600, Hangul Jamo at U+1100, and
  // everything astral arrives here as a surrogate. Both sides of each edge, since the query can
  // itself begin or end with one. Latin prose therefore pays four comparisons.
  if (
    !cut &&
    !(start > 0 && JOINS_GRAPHEME.test(text[start - 1])) &&
    !JOINS_GRAPHEME.test(text[start]) &&
    !JOINS_GRAPHEME.test(text[end - 1]) &&
    !(end < text.length && JOINS_GRAPHEME.test(text[end]))
  ) {
    return true;
  }
  return startsGrapheme(index, start) && startsGrapheme(index, end);
}

/** True when a grapheme starts at `at`. */
function startsGrapheme(index: FindTextIndex, at: number): boolean {
  const text = index.text;
  if (index.unsafe.has(at)) {
    // Where the walk stopped, what it left out is still on the page and anything at all can be
    // extended by what follows it, so this end cannot be vouched for.
    if (at === text.length || text[at] === BLOCK_SEPARATOR) return false;
    // The far side of such a cut is answerable, though: nothing below U+0300 can continue a
    // grapheme, so a character there starts one whatever was dropped in front of it.
    return !JOINS_GRAPHEME.test(text[at]);
  }
  if (at === 0 || at === text.length) return true;
  const platform = graphemeSegmenter();
  if (platform === null) return !continuesGrapheme(text, at);
  let segments = segmentsCache.get(index);
  if (segments === undefined) {
    segments = platform.segment(text);
    segmentsCache.set(index, segments);
  }
  return segments.containing(at)?.index === at;
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
