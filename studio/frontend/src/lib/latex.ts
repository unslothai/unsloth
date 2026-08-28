// Adapted from LibreChat's latex.ts
// https://github.com/danny-avila/LibreChat/blob/main/client/src/utils/latex.ts
//
// Two jobs, in order:
//   1. Convert LaTeX bracket delimiters (`\[...\]`, `\(...\)`) into the dollar
//      forms remark-math understands (`$$...$$`, `$...$`). remark-math only
//      tokenizes dollar delimiters, so models that emit `\[...\]` / `\(...\)`
//      would otherwise render as literal text.
//   2. Escape currency dollar signs so they are not misinterpreted as LaTeX
//      math delimiters when singleDollarTextMath is enabled.

import {
  EMPTY_LIST_STATE,
  NO_QUOTE,
  containerContent,
  indentWidth,
  itemContent,
  openLists,
  quoteDepth,
  quoteState,
} from "./markdown-list-columns.ts";
import { parseMarkdownIntoBlocks } from "streamdown";
import { codeSpans, codeSpansWithOpenTail } from "./markdown-code-spans.ts";

/**
 * Matches a single $ followed by a number pattern (currency), e.g.:
 *   $5, $1,000, $5.99, $100K, $3.5M
 *
 * Does NOT match:
 *   $$ (display math), \$ (already escaped), $\alpha (LaTeX command)
 */
const CURRENCY_REGEX =
  /(?<![\\$])\$(?!\$)(?=\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?(?:\s|$|[^a-zA-Z\d]))/g;

/**
 * Union of two span lists, each ascending by start (overlap within a list is
 * fine, non-ascending input silently drops spans). The sorted, non-overlapping
 * result is the shape `isInRegion`'s binary search needs.
 */
function mergeRegions(
  left: ReadonlyArray<readonly [number, number]>,
  right: ReadonlyArray<readonly [number, number]>,
): Array<[number, number]> {
  const merged: Array<[number, number]> = [];
  let leftIndex = 0;
  let rightIndex = 0;
  while (leftIndex < left.length || rightIndex < right.length) {
    const takeLeft =
      rightIndex >= right.length ||
      (leftIndex < left.length && left[leftIndex][0] <= right[rightIndex][0]);
    const next = takeLeft ? left[leftIndex++] : right[rightIndex++];
    const last = merged[merged.length - 1];
    if (last && next[0] < last[1]) {
      if (next[1] > last[1]) {
        last[1] = next[1];
      }
      continue;
    }
    merged.push([next[0], next[1]]);
  }
  return merged;
}

const FENCE_LINE_RE = /^ {0,3}(`{3,}|~{3,})([^\r\n]*)$/;
const FENCE_CANDIDATE_RE =
  /(^|\r\n|\n|\r)((?:(?: {0,3}>[ \t]?)|(?:[ \t]*(?:[-+*]|\d{1,9}[.)])[ \t]+))*[ \t]*)(`{3,}|~{3,})([^\r\n]*)/g;
const INDENTED_CODE_CANDIDATE_RE =
  /(^|\r\n|\n|\r)(?: {0,3}>[ \t]?)*(?:(?: {4}| {0,3}\t)| {0,3}(?:[-+*]|\d{1,9}[.)])(?: {5,}|[ \t]*\t[ \t]*))/g;
const BLOCK_LINE_RE =
  /^ {0,3}(?:#{1,6}([ \t]|$)|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$|>|=+[ \t]*$)/;
const LINK_DEFINITION_RE = /^ {0,3}\[(?:[^\[\]\\]|\\.)+\]:/;
const CODE_SPAN_BLOCK_BOUNDARY_RE =
  /(?:\r?\n[ \t]*\r?\n|(?:^|[\r\n]) {0,3}(?:#{1,6}(?:[ \t]|$)|>|[-+*](?:[ \t]|$)|\d{1,9}[.)](?:[ \t]|$)|`{3,}|~{3,}))/;
const NON_LINE_ENDING_RE = /[^\r\n]/g;

/** `line` with up to `columns` columns of leading whitespace removed. */
function stripIndent(line: string, columns: number): string {
  let width = 0;
  let index = 0;
  while (index < line.length && width < columns) {
    const char = line[index];
    if (char !== " " && char !== "\t") break;
    width += char === " " ? 1 : 4 - (width % 4);
    index += 1;
  }
  return line.slice(index);
}

/** Display columns occupied by a Markdown container prefix. */
function columnWidth(prefix: string): number {
  let width = 0;
  for (const char of prefix) {
    width += char === "\t" ? 4 - (width % 4) : 1;
  }
  return width;
}

/**
 * Find code regions (fenced, indented, and inline) to skip.
 * Returns a sorted, non-overlapping array of [start, end] index pairs.
 */
function normalizedMarkdownOffsets(content: string): {
  text: string;
  offsets: number[];
} {
  let text = "";
  const offsets = [0];
  for (let index = 0; index < content.length; index += 1) {
    if (content[index] === "\r") {
      if (content[index + 1] === "\n") {
        index += 1;
      }
      text += "\n";
    } else {
      text += content[index];
    }
    offsets.push(index + 1);
  }
  return { text, offsets };
}

function findInlineCodeRegions(
  content: string,
  includeOpenTail: boolean,
  blockRegions: Array<[number, number]>,
): Array<[number, number]> {
  let masked = content;
  if (blockRegions.length > 0) {
    const parts: string[] = [];
    let cursor = 0;
    for (const [start, end] of blockRegions) {
      parts.push(content.slice(cursor, start));
      parts.push(content.slice(start, end).replace(NON_LINE_ENDING_RE, " "));
      cursor = end;
    }
    parts.push(content.slice(cursor));
    masked = parts.join("");
  }

  const spans = codeSpans(masked);
  const crossesBlock = spans.some(({ start, end }) =>
    CODE_SPAN_BLOCK_BOUNDARY_RE.test(masked.slice(start, end)),
  );
  if (!includeOpenTail && !crossesBlock) {
    return spans.map(({ start, end }) => [start, end]);
  }

  const normalized = normalizedMarkdownOffsets(masked);
  const blocks = parseMarkdownIntoBlocks(normalized.text);
  const regions: Array<[number, number]> = [];
  let blockStart = 0;
  for (const block of blocks) {
    const blockSpans = includeOpenTail
      ? codeSpansWithOpenTail(block)
      : codeSpans(block);
    for (const span of blockSpans) {
      regions.push([
        normalized.offsets[blockStart + span.start] ?? content.length,
        normalized.offsets[blockStart + span.end] ?? content.length,
      ]);
    }
    blockStart += block.length;
  }
  return regions;
}

export function findCodeBlockRegions(
  content: string,
  includeOpenInlineTail = false,
): Array<[number, number]> {
  const fenced: Array<[number, number]> = [];
  let match: RegExpExecArray | null;
  const indented: Array<[number, number]> = [];

  FENCE_CANDIDATE_RE.lastIndex = 0;
  INDENTED_CODE_CANDIDATE_RE.lastIndex = 0;
  const needsBlockScan =
    FENCE_CANDIDATE_RE.test(content) ||
    INDENTED_CODE_CANDIDATE_RE.test(content);
  if (needsBlockScan) {
    const lines = content.matchAll(/[^\r\n]*(?:\r\n|\n|\r|$)/g);
    let openFence: {
      start: number;
      marker: string;
      column: number;
      quotes: number;
    } | null = null;
    let indentedStart = -1;
    let indentedEnd = -1;
    let afterParagraph = false;
    let lists = EMPTY_LIST_STATE;
    let quote = NO_QUOTE;

    for (const lineMatch of lines) {
      const line = lineMatch[0];
      if (!line) break;
      const start = lineMatch.index;
      const text = line.replace(/(?:\r\n|\n|\r)$/, "");
      const above = quote;
      quote = NO_QUOTE;
      const quotes = quoteDepth(text);

      if (openFence !== null) {
        const quoted = containerContent(
          text,
          EMPTY_LIST_STATE,
          openFence.quotes,
        );
        const leftContainer =
          quotes < openFence.quotes ||
          (quoted.trim() !== "" &&
            openFence.column > 0 &&
            indentWidth(quoted) < openFence.column);
        if (leftContainer) {
          fenced.push([openFence.start, start]);
          openFence = null;
        }
      }

      const activeFence = openFence;
      const container = containerContent(
        text,
        lists,
        activeFence?.quotes ?? quotes,
      );
      const fenceSource: string = activeFence
        ? stripIndent(container, activeFence.column)
        : itemContent(container, afterParagraph);
      const fence = FENCE_LINE_RE.exec(fenceSource);
      if (fence !== null) {
        lists = openLists(text, lists, afterParagraph, above.quoted);
        const marker = fence[1] ?? "";
        const tail = fence[2] ?? "";
        if (openFence === null) {
          if (marker[0] !== "`" || !tail.includes("`")) {
            const quoted = containerContent(text, EMPTY_LIST_STATE, quotes);
            openFence = {
              start,
              marker,
              column: columnWidth(
                quoted.slice(0, quoted.length - fenceSource.length),
              ),
              quotes,
            };
          }
        } else if (
          marker[0] === openFence.marker[0] &&
          marker.length >= openFence.marker.length &&
          !tail.trim()
        ) {
          fenced.push([openFence.start, start + line.length]);
          openFence = null;
        }
        afterParagraph = false;
        continue;
      }

      if (openFence !== null) {
        lists = openLists("", lists, afterParagraph, above.quoted);
        afterParagraph = false;
        continue;
      }

      lists = openLists(text, lists, afterParagraph, above.quoted);
      const inner = itemContent(
        containerContent(text, lists, quoteDepth(text)),
        afterParagraph,
      );
      const blank = /^\s*$/.test(inner);
      const code = indentWidth(inner) >= 4;
      if (indentedStart >= 0) {
        if (code || blank) {
          indentedEnd = start + line.length;
          continue;
        }
        indented.push([indentedStart, indentedEnd]);
        indentedStart = -1;
      }
      if (code && !afterParagraph && !blank) {
        indentedStart = start;
        indentedEnd = start + line.length;
        afterParagraph = false;
        continue;
      }
      afterParagraph =
        !blank &&
        !BLOCK_LINE_RE.test(text) &&
        (afterParagraph || !LINK_DEFINITION_RE.test(text));
      quote = quoteState(text, above.inQuote);
    }
    if (openFence !== null) fenced.push([openFence.start, content.length]);
    if (indentedStart >= 0) indented.push([indentedStart, indentedEnd]);
  }

  const blocks = mergeRegions(fenced, indented);
  const inline = findInlineCodeRegions(content, includeOpenInlineTail, blocks);

  // An inline span can CONTAIN a fence (`` `~~~a~~~ $5` ``); that overlap made
  // the binary search land on the inner span and miss the outer one.
  return mergeRegions(blocks, inline);
}

/**
 * Match an inline link/image `[text](DEST)`, capturing the destination as group 1
 * with the `d` flag so its span is read straight from `match.indices` (the text
 * can contain an escaped `\](`, so a string search for the separator is unsafe).
 * The text disallows unescaped `]`; the destination allows escapes and one level
 * of balanced parens.
 */
const LINK_DEST_RE =
  /!?\[(?:\\.|[^\]\\])*?\]\(((?:\\.|[^()\\]|\([^()]*\))*)\)/dg;
const AUTOLINK_RE =
  /<[a-zA-Z][a-zA-Z\d+.-]{1,31}:[^<>\s]*>|<[a-zA-Z\d.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z\d.-]+>/g;
const HTML_TAG_RE =
  /<\/?[a-zA-Z][a-zA-Z\d-]*(?:\s+(?:[^"'<>]|"[^"]*"|'[^']*')*)?\s*\/?>/g;
const HTML_OPAQUE_RE =
  /<!--[\s\S]*?(?:-->|$)|<\?[\s\S]*?(?:\?>|$)|<!\[CDATA\[[\s\S]*?(?:\]\]>|$)|<![a-zA-Z][^>]*(?:>|$)/g;
const HTML_BLOCK_OPEN_RE = /^ {0,3}<\/?([a-zA-Z][a-zA-Z\d-]*)(?=[\s/>]|$)/;
const HTML_BLOCK_TAG_ONLY_RE =
  /^ {0,3}<\/?[a-zA-Z][a-zA-Z\d-]*(?:\s+(?:[^"'<>]|"[^"]*"|'[^']*')*)?\s*\/?>\s*$/;
const HTML_BLOCK_TAGS = new Set(
  `address article aside base basefont blockquote body caption center col colgroup
   dd details dialog dir div dl dt fieldset figcaption figure footer form frame
   frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li link main menu
   menuitem nav noframes ol optgroup option p param search section summary table
   tbody td tfoot th thead title tr track ul`.split(/\s+/),
);
const HTML_LITERAL_TAGS = new Set([
  "code",
  "pre",
  "script",
  "style",
  "textarea",
]);

/**
 * Find the destination spans of inline links/images, so a `\(...\)` written with
 * escaped parens inside a URL isn't rewritten as math (which would break the
 * link). Only the destination is returned, not the link text, so math in the
 * visible text still converts. Sorted, non-overlapping (matches are disjoint).
 */
function findLinkDestinationRegions(content: string): Array<[number, number]> {
  if (!content.includes("](")) return [];
  const regions: Array<[number, number]> = [];
  let match: RegExpExecArray | null;
  LINK_DEST_RE.lastIndex = 0;
  while ((match = LINK_DEST_RE.exec(content)) !== null) {
    // `indices` is present (the `d` flag); group 1 spans the destination.
    regions.push(match.indices![1]);
  }
  return regions;
}

function findHtmlRegions(content: string): Array<[number, number]> {
  const tags: Array<[number, number]> = [];
  const literals: Array<[number, number]> = [];
  const openLiterals: Array<{ name: string; start: number }> = [];
  let match: RegExpExecArray | null;

  HTML_TAG_RE.lastIndex = 0;
  while ((match = HTML_TAG_RE.exec(content)) !== null) {
    const end = match.index + match[0].length;
    tags.push([match.index, end]);
    const name = /^<\/?([a-zA-Z][a-zA-Z\d-]*)/
      .exec(match[0])?.[1]
      ?.toLowerCase();
    if (!name || !HTML_LITERAL_TAGS.has(name)) {
      continue;
    }
    if (match[0].startsWith("</")) {
      let opener = openLiterals.length - 1;
      while (opener >= 0 && openLiterals[opener]?.name !== name) {
        opener -= 1;
      }
      if (opener >= 0) {
        literals.push([openLiterals[opener].start, end]);
        openLiterals.length = opener;
      }
    } else if (!match[0].endsWith("/>") && name !== "code") {
      openLiterals.push({ name, start: match.index });
    } else if (name === "code" && !match[0].endsWith("/>")) {
      openLiterals.push({ name, start: match.index });
    }
  }
  for (const opener of openLiterals) {
    literals.push([opener.start, content.length]);
  }

  const blocks: Array<[number, number]> = [];
  let blockStart = -1;
  let previousBlank = true;
  for (const lineMatch of content.matchAll(/[^\r\n]*(?:\r\n|\n|\r|$)/g)) {
    const line = lineMatch[0];
    if (!line) {
      break;
    }
    const text = line.replace(/(?:\r\n|\n|\r)$/, "");
    if (blockStart >= 0) {
      if (!text.trim()) {
        blocks.push([blockStart, lineMatch.index]);
        blockStart = -1;
      }
      previousBlank = !text.trim();
      continue;
    }
    const name = HTML_BLOCK_OPEN_RE.exec(text)?.[1]?.toLowerCase();
    if (
      (name && HTML_BLOCK_TAGS.has(name)) ||
      (previousBlank && HTML_BLOCK_TAG_ONLY_RE.test(text))
    ) {
      blockStart = lineMatch.index;
    }
    previousBlank = !text.trim();
  }
  if (blockStart >= 0) {
    blocks.push([blockStart, content.length]);
  }

  const autolinks: Array<[number, number]> = [];
  AUTOLINK_RE.lastIndex = 0;
  while ((match = AUTOLINK_RE.exec(content)) !== null) {
    autolinks.push([match.index, match.index + match[0].length]);
  }
  HTML_OPAQUE_RE.lastIndex = 0;
  while ((match = HTML_OPAQUE_RE.exec(content)) !== null) {
    autolinks.push([match.index, match.index + match[0].length]);
  }
  return mergeRegions(
    mergeRegions(tags, literals),
    mergeRegions(blocks, autolinks),
  );
}

/**
 * Binary search to check if a position falls inside any region. Regions must be
 * sorted by start and non-overlapping.
 */
export function isInRegion(
  position: number,
  regions: Array<[number, number]>,
): boolean {
  let lo = 0;
  let hi = regions.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >>> 1;
    const [start, end] = regions[mid];
    if (position < start) {
      hi = mid - 1;
    } else if (position >= end) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

function intersectsRegion(
  start: number,
  end: number,
  regions: Array<[number, number]>,
): boolean {
  let low = 0;
  let high = regions.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (regions[middle][1] <= start) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low < regions.length && regions[low][0] < end;
}

/** A whitespace-free token that looks purely like currency, e.g. `5`, `1,000`, `5.99`, `100K`, `3.5M`. */
const CURRENCY_BODY_RE = /^\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?$/;

/** Body characters that almost always indicate real LaTeX. */
const LATEX_CHAR_RE = /[\\^_{}]/;

/**
 * Operators that strongly suggest math. Omits `^` and `_` since
 * `LATEX_CHAR_RE` short-circuits on those before this regex runs.
 */
const MATH_OP_RE = /[=+\-<>/*]/;

/**
 * Trailing chars stripped before the currency check: prose punctuation
 * plus `-` and `/` from compact ranges like `$5-$10`; without them the
 * body `5-` or `5/` would slip through the single-token math shortcut.
 */
const TRAIL_PUNCT_RE = /[.,;:!?\-/]+$/;

/**
 * A standalone single-letter variable, not part of a longer word, so
 * prose like "5 to attend" isn't misread as math with variable `t`.
 */
const LONE_LETTER_RE = /(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])/;

/**
 * Numeric or single-letter operands joined by math operators (optional
 * whitespace): `2 + 2`, `100 < 200`, `1,000 - 500`, `x + y`. Recognises
 * numeric-only expressions like `$2 + 2$` without a lone variable token.
 */
const SIMPLE_MATH_RE =
  /^(?:\d+(?:,\d{3})*(?:\.\d+)?|[a-zA-Z])(?:\s*[=+\-<>/*]\s*(?:\d+(?:,\d{3})*(?:\.\d+)?|[a-zA-Z]))+$/;

/**
 * True if the substring between two `$` delimiters looks like LaTeX
 * rather than prose between two currency tokens.
 *
 * Rule of thumb:
 *   - `$30^\circ$`  -> math (LaTeX chars)
 *   - `$x$`         -> math (single non-currency token)
 *   - `$90 - x$`    -> math (math op + lone variable)
 *   - `$5 to $10`   -> NOT math (multi-token prose, no math op)
 *   - `$5, $10`     -> NOT math (currency-like token + trailing punct)
 *   - `$1,000$`     -> NOT math (single currency-like token)
 */
function looksLikeMathBody(body: string): boolean {
  if (LATEX_CHAR_RE.test(body)) return true;
  const trimmed = body.trim().replace(TRAIL_PUNCT_RE, "");
  if (!trimmed) return false;
  if (CURRENCY_BODY_RE.test(trimmed)) return false;
  // Numeric-only operator forms: `2 + 2`, `100 < 200`, `1,000 - 500`.
  // Recognised without requiring a lone-variable letter.
  if (SIMPLE_MATH_RE.test(trimmed)) return true;
  if (!/\s/.test(trimmed)) return true;
  if (!MATH_OP_RE.test(trimmed)) return false;
  return LONE_LETTER_RE.test(trimmed);
}

const OPERATOR_ONLY_RE = /^[=+\-<>/*]+$/;
const PLAIN_WORD_EXPRESSION_RE = /^[a-zA-Z]{2,}(?:[-/][a-zA-Z]{2,})+$/;

function looksLikeEscapedMathBody(body: string): boolean {
  const trimmed = body.trim();
  if (
    OPERATOR_ONLY_RE.test(trimmed) ||
    PLAIN_WORD_EXPRESSION_RE.test(trimmed) ||
    (/^[a-zA-Z]+$/.test(trimmed) && trimmed.length > 1)
  ) {
    return false;
  }
  return looksLikeMathBody(trimmed);
}

/**
 * True if the `$` at `offset` opens a balanced inline math span (`$...$`)
 * on the same line. The closer must be unescaped, not part of `$$`, and
 * within 200 chars. The body must look like LaTeX so we don't pair two
 * currency tokens on a line (e.g. "$5 to $10"). Bold-wrapped spans
 * (`**$X$**`, `__$X$__`) are always math: LLMs use that for "bold math"
 * and the heuristic would otherwise reject prose-shaped bodies like "90 - x".
 */
function hasInlineMathCloser(
  content: string,
  offset: number,
  mathRegions: Array<[number, number]>,
): boolean {
  const maxSpan = 200;
  const limit = Math.min(content.length, offset + 1 + maxSpan);
  for (let i = offset + 1; i < limit; i++) {
    const c = content[i];
    if (c === "\n") return false;
    if (c !== "$") continue;
    if (content[i - 1] === "\\") continue;
    // A `$` opening a generated span (from `\(...\)`) is not a currency closer;
    // pairing with it would swallow the price into math (`$5 + x \(y\)`).
    if (isInRegion(i, mathRegions)) return false;
    if (content[i + 1] === "$") {
      i++;
      continue;
    }
    // A `$` followed by a digit is more likely another currency token than
    // the closer. Keep scanning so prose like `$5 + a $10 add-on` doesn't
    // pair the two currency markers as a math span.
    if (/\d/.test(content[i + 1] ?? "")) {
      continue;
    }
    if (offset >= 2) {
      const op = content[offset - 1];
      if (
        (op === "*" || op === "_") &&
        content[offset - 2] === op &&
        content[i + 1] === op &&
        content[i + 2] === op
      ) {
        return true;
      }
    }
    return looksLikeMathBody(content.slice(offset + 1, i));
  }
  return false;
}

function isEscapedDollar(content: string, offset: number): boolean {
  if (content[offset] !== "$") return false;
  let slashes = 0;
  for (let i = offset - 1; i >= 0 && content[i] === "\\"; i -= 1) {
    slashes += 1;
  }
  return slashes % 2 === 1;
}

/** Find existing bracket-delimited math outside code and link destinations. */
function findBracketMathRegions(
  content: string,
  skipRegions: Array<[number, number]>,
): Array<[number, number]> {
  const regions: Array<[number, number]> = [];
  for (const [opener, closer] of [
    ["\\(", "\\)"],
    ["\\[", "\\]"],
  ] as const) {
    let opening = -1;
    for (let i = 0; i < content.length - 1; i += 1) {
      if (content[i] !== "\\") continue;
      const token = content.slice(i, i + 2);
      if (isInRegion(i, skipRegions)) {
        if (opening >= 0 && token === closer) opening = -1;
        if (token === opener || token === closer) i += 1;
        continue;
      }
      if (opening < 0) {
        if (token === opener && content[i - 1] !== "\\") opening = i;
      } else if (token === closer) {
        regions.push([opening, i + 2]);
        opening = -1;
      }
      if (token === opener || token === closer) i += 1;
    }
  }
  const sorted = regions.sort((a, b) => a[0] - b[0]);
  const nonOverlapping: Array<[number, number]> = [];
  for (const region of sorted) {
    const previous = nonOverlapping[nonOverlapping.length - 1];
    if (!previous || region[0] >= previous[1]) nonOverlapping.push(region);
  }
  return nonOverlapping;
}

/** Find existing `$...$` and `$$...$$` regions outside code and links. */
function findDollarMathRegions(
  content: string,
  skipRegions: Array<[number, number]>,
): Array<[number, number]> {
  const displayRegions: Array<[number, number]> = [];
  let displayOpening = -1;
  for (let i = 0; i < content.length - 1; i += 1) {
    if (
      content[i] !== "$" ||
      content[i + 1] !== "$" ||
      isEscapedDollar(content, i) ||
      isInRegion(i, skipRegions)
    ) {
      continue;
    }
    if (displayOpening < 0) {
      displayOpening = i;
    } else {
      displayRegions.push([displayOpening, i + 2]);
      displayOpening = -1;
    }
    i += 1;
  }

  const inlineRegions: Array<[number, number]> = [];
  const allSkipRegions = mergeRegions(skipRegions, displayRegions);
  let inlineOpening = -1;
  for (let i = 0; i < content.length; i += 1) {
    if (content[i] === "\n" || content[i] === "\r") {
      inlineOpening = -1;
      continue;
    }
    if (
      content[i] !== "$" ||
      isEscapedDollar(content, i) ||
      content[i - 1] === "$" ||
      content[i + 1] === "$" ||
      isInRegion(i, allSkipRegions)
    ) {
      continue;
    }
    if (inlineOpening < 0) {
      inlineOpening = i;
      continue;
    }
    if (i - inlineOpening - 1 > 200) {
      inlineOpening = i;
      continue;
    }
    if (/\d/.test(content[i + 1] ?? "")) continue;
    if (looksLikeMathBody(content.slice(inlineOpening + 1, i))) {
      inlineRegions.push([inlineOpening, i + 1]);
      inlineOpening = -1;
    } else {
      inlineOpening = i;
    }
  }
  return [...displayRegions, ...inlineRegions].sort((a, b) => a[0] - b[0]);
}

const CONVERT_LATEX_DELIM_RE =
  /(?<!\\)\\\[([\s\S]{0,4096}?)\\\]|(?<!\\)\\\(([\s\S]{0,4096}?)\\\)/g;

/**
 * recover math-looking `\$...\$` spans emitted by local models. existing
 * math, currency, code, links, even-backslash runs, and incomplete spans stay
 * literal. same-line pairs are consumed once.
 */
function normalizeEscapedInlineMath(content: string): string {
  if (!content.includes("\\$")) return content;

  let skipRegions = mergeRegions(
    findCodeBlockRegions(content, true),
    findLinkDestinationRegions(content),
  );
  skipRegions = mergeRegions(skipRegions, findHtmlRegions(content));
  skipRegions = mergeRegions(
    skipRegions,
    findBracketMathRegions(content, skipRegions),
  );
  skipRegions = mergeRegions(
    skipRegions,
    findDollarMathRegions(content, skipRegions),
  );
  const parts: string[] = [];
  let last = 0;
  let lastChar = "";
  const append = (chunk: string): void => {
    if (!chunk) return;
    if (lastChar === "$" && chunk.startsWith("$")) parts.push(" ");
    parts.push(chunk);
    lastChar = chunk[chunk.length - 1];
  };

  for (let opening = 0; opening < content.length; opening += 1) {
    if (
      content[opening] !== "$" ||
      !isEscapedDollar(content, opening) ||
      content[opening - 1] === "$" ||
      content[opening + 1] === "$" ||
      isInRegion(opening, skipRegions)
    ) {
      continue;
    }

    const opener = opening;
    for (let closing = opening + 1; closing < content.length; closing += 1) {
      if (content[closing] === "\n" || content[closing] === "\r") break;
      if (content[closing] !== "$" || !isEscapedDollar(content, closing)) {
        continue;
      }
      if (
        content[closing + 1] === "$" ||
        isInRegion(closing, skipRegions) ||
        intersectsRegion(opener, closing + 1, skipRegions)
      ) {
        opening = closing - 1;
        break;
      }

      const body = content.slice(opener + 1, closing - 1);
      const trimmed = body.trim();
      if (body.length > 200) {
        opening = closing;
        break;
      }
      if (!looksLikeEscapedMathBody(trimmed)) {
        opening = closing - 1;
        break;
      }

      append(content.slice(last, opener - 1));
      append(`$${trimmed}$`);
      last = closing + 1;
      opening = closing;
      break;
    }
  }

  if (parts.length === 0) return content;
  append(content.slice(last));
  return parts.join("");
}

/**
 * Rewrite `\[...\]` -> block `$$...$$` and `\(...\)` -> inline `$...$` so
 * remark-math can tokenize them. Bodies are trimmed: remark-math won't open an
 * inline span on `$ ` (a `$` followed by whitespace), and display fences must
 * sit on their own line to render as a centered block (not inline math), so
 * `\[...\]` becomes `\n$$\n...\n$$\n`.
 *
 * Spans inside code blocks/spans are left intact (a code sample showing `\(x\)`
 * must not be rewritten).
 *
 * A space is inserted between a converted span and a following `$` so their
 * delimiters can't fuse (`\(a\)\(b\)` -> `$a$$b$` would mis-tokenize into one
 * broken span). A preceding currency (`$5\(x\)`) is instead broken later by the
 * currency escape pass.
 *
 * Returns the rewritten text and the `[start, end)` ranges (in the rewritten
 * string) of every span it produced, so the currency pass can skip them.
 */
function convertLatexDelimiters(content: string): {
  text: string;
  mathRegions: Array<[number, number]>;
} {
  if (!content.includes("\\[") && !content.includes("\\(")) {
    return { text: content, mathRegions: [] };
  }

  const codeRegions = findCodeBlockRegions(content);
  const linkRegions = findLinkDestinationRegions(content);
  const inSkipZone = (pos: number) =>
    isInRegion(pos, codeRegions) || isInRegion(pos, linkRegions);
  // Pushed in ascending, non-overlapping order (offset only grows), so this
  // stays valid for isInRegion's binary search without a sort.
  const mathRegions: Array<[number, number]> = [];
  // Accumulate into an array, not a string: reading the last char off a growing
  // `+=` accumulator flattens its rope every append (O(n^2) over many spans, on
  // the per-frame streaming path), so track the tail char and length instead.
  const parts: string[] = [];
  let offset = 0;
  let lastChar = "";
  let last = 0;
  // Append a chunk, separating a trailing `$` from a leading `$` so two spans
  // can't fuse. Returns where the chunk landed (after any inserted space).
  const append = (chunk: string): number => {
    if (!chunk) return offset;
    if (lastChar === "$" && chunk.startsWith("$")) {
      parts.push(" ");
      offset += 1;
    }
    const start = offset;
    parts.push(chunk);
    offset += chunk.length;
    lastChar = chunk[chunk.length - 1];
    return start;
  };
  let match: RegExpExecArray | null;
  CONVERT_LATEX_DELIM_RE.lastIndex = 0;
  while ((match = CONVERT_LATEX_DELIM_RE.exec(content)) !== null) {
    const matchEnd = match.index + match[0].length;
    // Skip if either delimiter is inside code or a link destination: an opener
    // outside such a zone must not consume a closer inside one and rewrite
    // across the boundary. Resume right after this opener (not past the whole
    // match) so a valid span that this match spanned across (a stray code `\(`
    // paired with a real closer) is still found on the next pass, not swallowed.
    if (inSkipZone(match.index) || inSkipZone(matchEnd - 1)) {
      CONVERT_LATEX_DELIM_RE.lastIndex = match.index + 1;
      continue;
    }
    const isDisplay = match[1] !== undefined;
    const body = (isDisplay ? match[1] : match[2]).trim();
    // Leave an empty span (`\(\)`) literal; a bare `$$` would open a stray
    // display block that swallows following text.
    if (!body) {
      continue;
    }
    append(content.slice(last, match.index));
    let wrapped: string;
    if (isDisplay) {
      // Keep the opener's leading indentation so a `$$` block inside a list item
      // stays in the container instead of breaking out at column 0. Only when the
      // opener is whitespace-prefixed, so inline `text \[x\]` keeps column 0.
      const lineStart =
        match.index > 0 ? content.lastIndexOf("\n", match.index - 1) + 1 : 0;
      const prefix = content.slice(lineStart, match.index);
      const indent = /^\s*$/.test(prefix) ? prefix : "";
      // Indent every body line, not just the first, so multi-line display math
      // (`\[a\nb\]`) stays wholly inside the container.
      const inner = indent ? body.replace(/\n/g, `\n${indent}`) : body;
      wrapped = `\n${indent}$$\n${indent}${inner}\n${indent}$$\n`;
    } else {
      wrapped = `$${body}$`;
    }
    const start = append(wrapped);
    mathRegions.push([start, offset]);
    last = matchEnd;
  }
  append(content.slice(last));
  return { text: parts.join(""), mathRegions };
}

/**
 * Preprocess a markdown string so LaTeX renders: convert bracket delimiters to
 * dollar forms, then escape currency dollar signs so they are not parsed as
 * math delimiters.
 *
 * - `\[E = mc^2\]` becomes a `$$` display block on its own lines (display math)
 * - `\(\alpha\)` becomes `$\alpha$` (inline math)
 * - `\(x\)` in a code span is untouched
 * - `$5` alone becomes `\$5` (currency, not math)
 * - `$\alpha$` is untouched (real LaTeX)
 * - `$30^\circ$` is untouched (LaTeX whose body starts with a digit)
 * - `**$30^\circ$**` is untouched (LaTeX wrapped in bold)
 * - `$$E = mc^2$$` is untouched (display math)
 * - Currency inside code blocks/spans is untouched
 */
export function preprocessLaTeX(content: string): string {
  const normalized = normalizeEscapedInlineMath(content);
  const { text, mathRegions } = convertLatexDelimiters(normalized);

  if (!text.includes("$")) return text;

  const codeRegions = findCodeBlockRegions(text);

  return text.replace(CURRENCY_REGEX, (match, offset) => {
    if (isInRegion(offset, codeRegions)) {
      return match;
    }
    // Skip the spans we just created from `\(...\)` so a numeric body like
    // `$5$` isn't re-escaped back to literal `\$5$`.
    if (isInRegion(offset, mathRegions)) {
      return match;
    }
    if (hasInlineMathCloser(text, offset, mathRegions)) {
      return match;
    }
    return "\\" + match;
  });
}
