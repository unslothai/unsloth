// Adapted from LibreChat's latex.ts
// https://github.com/danny-avila/LibreChat/blob/main/client/src/utils/latex.ts
//
// Converts `\[...\]` / `\(...\)` to the dollar forms remark-math tokenizes, then escapes currency
// dollars so singleDollarTextMath does not read them as math.

import { parseMarkdownIntoBlocks } from "./parse-markdown-blocks.ts";
import {
  codeSpans,
  crossesCodeSpanBlockBoundary,
} from "./markdown-code-spans.ts";
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

/** matches currency bodies after `$`, including separators, decimals, and K/M/B suffixes. */
const CURRENCY_REGEX = /\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?(?:\s|$|[^a-zA-Z\d])/y;

const HEADING_LINE_RE = / {0,3}#{1,6}(?=[ \t\r\n]|$)/y;
const TABLE_ROW_RE = /[ \t]*\|/y;
const BLOCK_BREAK_RE =
  /\n[ \t\r]*(?:\n|#{1,6}(?=[ \t\r\n])|[>|]|[-*+][ \t]|1[.)][ \t]|```|~~~|(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,}|=+[ \t]*)(?=\r?(?:\n|$)))/;

/** matches prose-like `$NAME ... $word` spans without math symbols. */
// non-ASCII letters and CJK punctuation are prose (KaTeX rejects them in math mode).
const VARIABLE_PROSE_RE =
  /^(?!\w+\s+$)(?:[A-Za-z]{2,}\w*|_\w+|\{[A-Za-z_]\w*\})[\w\s.,;:!?'"()/`|&<>=*\-\p{L}\p{M}\u3000-\u303f\uff00-\uff65]*(?:[\s/:,.;|<>=\-\u3000-\u303f\uff00-\uff65]|[^\P{L}\p{ASCII}]|[\s(]["'(`])$/u;
// VARIABLE_PROSE_RE backtracks ~cubically: longer spans, or any past the per-message budget, stay math.
const MAX_VARIABLE_PROSE_SPAN = 128;
const VARIABLE_PROSE_BUDGET = 2048;
const NEW_TOKEN_RE = /[\w{\\]/;
const isAsciiLetter = (c: number) => (c | 32) >= 97 && (c | 32) <= 122;
const isWordChar = (c: number) =>
  isAsciiLetter(c) || (c >= 48 && c <= 57) || c === 95;

/** necessary prefix of VARIABLE_PROSE_RE, checked before slicing the span. */
function startsLikeVariable(text: string, at: number): boolean {
  const a = text.charCodeAt(at);
  const b = text.charCodeAt(at + 1);
  if (isAsciiLetter(a)) return isAsciiLetter(b);
  if (a === 95) return isWordChar(b);
  return a === 123 && (isAsciiLetter(b) || b === 95);
}
// an entity stays literal in Markdown without showing an escape slash in raw HTML.
const VARIABLE_DOLLAR = "&#36;";

/** merges ascending spans; overlaps merge and non-ascending spans drop. */
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
const NON_LINE_ENDING_RE = /[^\r\n]/g;

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

function columnWidth(prefix: string): number {
  let width = 0;
  for (const char of prefix) {
    width += char === "\t" ? 4 - (width % 4) : 1;
  }
  return width;
}

/** `content` with CRLF/CR as LF, plus each normalized index's offset in the original. */
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
  blockRegions: Array<[number, number]>,
): Array<[number, number]> {
  // Skip the O(n) rebuild when no backtick lies outside a block; streaming calls this per frame.
  let spanTickOutsideBlock = false;
  for (
    let index = content.indexOf("`");
    index !== -1;
    index = content.indexOf("`", index + 1)
  ) {
    if (!isInRegion(index, blockRegions)) {
      spanTickOutsideBlock = true;
      break;
    }
  }
  if (!spanTickOutsideBlock) return [];

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
    crossesCodeSpanBlockBoundary(masked.slice(start, end)),
  );
  if (!crossesBlock) {
    return spans.map(({ start, end }) => [start, end]);
  }

  const normalized = normalizedMarkdownOffsets(masked);
  const blocks = parseMarkdownIntoBlocks(normalized.text);
  const regions: Array<[number, number]> = [];
  let blockStart = 0;
  for (const block of blocks) {
    for (const span of codeSpans(block)) {
      regions.push([
        normalized.offsets[blockStart + span.start] ?? content.length,
        normalized.offsets[blockStart + span.end] ?? content.length,
      ]);
    }
    blockStart += block.length;
  }
  return regions;
}

export function findCodeBlockRegions(content: string): Array<[number, number]> {
  const fenced: Array<[number, number]> = [];
  let match: RegExpExecArray | null;
  const indented: Array<[number, number]> = [];

  FENCE_CANDIDATE_RE.lastIndex = 0;
  INDENTED_CODE_CANDIDATE_RE.lastIndex = 0;
  let simpleFenceStart = -1;
  let simpleFenceMarker = "";
  let needsBlockScan = false;
  while ((match = FENCE_CANDIDATE_RE.exec(content)) !== null) {
    const prefix = match[2] ?? "";
    if (prefix.trim() || indentWidth(prefix) > 3) {
      needsBlockScan = true;
      break;
    }
    const marker = match[3] ?? "";
    const tail = match[4] ?? "";
    const lineStart = match.index + (match[1]?.length ?? 0);
    if (simpleFenceStart < 0) {
      if (marker[0] !== "`" || !tail.includes("`")) {
        simpleFenceStart = lineStart;
        simpleFenceMarker = marker;
      }
    } else if (
      marker[0] === simpleFenceMarker[0] &&
      marker.length >= simpleFenceMarker.length &&
      !tail.trim()
    ) {
      fenced.push([
        simpleFenceStart,
        lineStart + prefix.length + marker.length,
      ]);
      simpleFenceStart = -1;
      simpleFenceMarker = "";
    }
  }
  if (!needsBlockScan && simpleFenceStart >= 0) {
    fenced.push([simpleFenceStart, content.length]);
  }
  if (!needsBlockScan) {
    while ((match = INDENTED_CODE_CANDIDATE_RE.exec(content)) !== null) {
      if (!isInRegion(match.index + match[0].length - 1, fenced)) {
        needsBlockScan = true;
        break;
      }
    }
  }
  if (needsBlockScan) {
    fenced.length = 0;
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
      let fenceSource: string;
      if (activeFence) {
        fenceSource = stripIndent(container, activeFence.column);
      } else {
        fenceSource = container;
        let previous: string;
        let itemAfterParagraph = afterParagraph;
        do {
          previous = fenceSource;
          fenceSource = itemContent(fenceSource, itemAfterParagraph);
          itemAfterParagraph = false;
        } while (fenceSource !== previous);
      }
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
  const inline = findInlineCodeRegions(content, blocks);

  // merge overlaps so binary search does not miss a containing inline span.
  return mergeRegions(blocks, inline);
}

/** destinations allow escapes and one level of balanced parentheses. */
const LINK_DEST_RE =
  /!?\[(?:\\.|[^\]\\])*?\]\(((?:\\.|[^()\\]|\([^()]*\))*)\)/dg;
const AUTOLINK_RE =
  /<(?:[a-z][a-z0-9+.-]*:[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>|\b(?:https?:\/\/|www\.)[^\s<]+/gi;

/** shields link destinations from math rewrites; link text stays convertible. */
function findLinkDestinationRegions(content: string): Array<[number, number]> {
  if (!content.includes("](")) return [];
  const regions: Array<[number, number]> = [];
  let match: RegExpExecArray | null;
  LINK_DEST_RE.lastIndex = 0;
  while ((match = LINK_DEST_RE.exec(content)) !== null) {
    // escaped separators make searches unsafe; use the `d` flag's bounds.
    regions.push(match.indices![1]);
  }
  return regions;
}

// raw-text element bodies do not decode character references.
const RAW_TEXT_ELEMENT_RE =
  /<(script|style|xmp|iframe|noembed|noframes)\b[^>]*>[\s\S]*?(?:<\/\1\s*>|$)/gi;

function findRawTextRegions(content: string): Array<[number, number]> {
  if (!content.includes("<")) return [];
  const regions: Array<[number, number]> = [];
  for (const match of content.matchAll(RAW_TEXT_ELEMENT_RE)) {
    regions.push([match.index, match.index + match[0].length]);
  }
  return regions;
}

function findAutolinkRegions(content: string): Array<[number, number]> {
  const regions: Array<[number, number]> = [];
  AUTOLINK_RE.lastIndex = 0;
  for (const match of content.matchAll(AUTOLINK_RE)) {
    regions.push([match.index, match.index + match[0].length]);
  }
  return regions;
}

/** `isInRegion` for ascending positions: amortised O(1) per query. */
function regionCursor(
  regions: Array<[number, number]>,
): (position: number) => boolean {
  let k = 0;
  return (position) => {
    while (k < regions.length && regions[k][1] <= position) k++;
    return k < regions.length && regions[k][0] <= position;
  };
}

/** regions must be sorted by start and non-overlapping. */
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

const CURRENCY_BODY_RE = /^\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?$/;

const LATEX_CHAR_RE = /[\\^_{}]/;

/** Omits `^` and `_`, which LATEX_CHAR_RE handles first. */
const MATH_OP_RE = /[=+\-<>/*]/;

/** Includes `-` and `/` from ranges like `$5-$10`. */
const TRAIL_PUNCT_RE = /[.,;:!?\-/]+$/;

/** So "5 to attend" is not read as variable `t`. */
const LONE_LETTER_RE = /(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])/;

const SIMPLE_MATH_RE =
  /^(?:\d+(?:,\d{3})*(?:\.\d+)?|[a-zA-Z])(?:\s*[=+\-<>/*]\s*(?:\d+(?:,\d{3})*(?:\.\d+)?|[a-zA-Z]))+$/;

/**
 *   - `$30^\circ$`  -> math (LaTeX chars)
 *   - `$x$`         -> math (single non-currency token)
 *   - `$90 - x$`    -> math (math op + lone variable)
 *   - `$5 to $10`   -> NOT math (multi-token prose)
 *   - `$1,000$`     -> NOT math (single currency-like token)
 */
function looksLikeMathBody(body: string): boolean {
  if (LATEX_CHAR_RE.test(body)) return true;
  const trimmed = body.trim().replace(TRAIL_PUNCT_RE, "");
  if (!trimmed) return false;
  if (CURRENCY_BODY_RE.test(trimmed)) return false;
  if (SIMPLE_MATH_RE.test(trimmed)) return true;
  if (!/\s/.test(trimmed)) return true;
  if (!MATH_OP_RE.test(trimmed)) return false;
  return LONE_LETTER_RE.test(trimmed);
}

/** Same line, unescaped, not `$$`, within 200 chars, with a LaTeX-like body. Bold-wrapped spans
 * are always math, since LLMs use that for bold math. */
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
    // A `$` opening a generated span is not a currency closer.
    if (isInRegion(i, mathRegions)) return false;
    if (content[i + 1] === "$") {
      i++;
      continue;
    }
    // A `$` before a digit is more likely another price than the closer.
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

// remark-math closes an open span at the next single `$`, escaped or not, within the block.
function findInlineMathCloser(
  content: string,
  offset: number,
  lineStart: number,
  lineEnd: number,
  tableRow: boolean,
): number {
  let i = content.indexOf("$", offset + 1);
  while (i !== -1 && content[i + 1] === "$") {
    while (content[i + 1] === "$") i++;
    i = content.indexOf("$", i + 1);
  }
  if (i === -1) return -1;
  // hot path: only multi-line spans can cross a block break or leave a heading.
  const multiline = lineEnd !== -1 && lineEnd < i;
  if (multiline) {
    if (BLOCK_BREAK_RE.test(content.slice(offset + 1, i))) return -1;
    HEADING_LINE_RE.lastIndex = lineStart;
    if (HEADING_LINE_RE.test(content)) return -1;
  }
  if (tableRow) {
    if (multiline) return -1;
    for (
      let p = content.indexOf("|", offset + 1);
      p !== -1 && p < i;
      p = content.indexOf("|", p + 1)
    ) {
      if (content[p - 1] !== "\\") return -1;
    }
  }
  return i;
}

/** caps LaTeX delimiter bodies at 4,096 characters so repeated incomplete openers stay linear while streaming. */
const CONVERT_LATEX_DELIM_RE =
  /(?<!\\)\\\[([\s\S]{0,4096}?)\\\]|(?<!\\)\\\(([\s\S]{0,4096}?)\\\)/g;

/**
 * Bodies are trimmed (remark-math will not open on `$ `), display fences go on their own lines,
 * code spans are skipped, and a space separates adjacent `$` so spans cannot fuse. Returns the
 * produced ranges so the currency pass skips them.
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
  // Ascending and non-overlapping by construction, so no sort is needed.
  const mathRegions: Array<[number, number]> = [];
  // An array, not `+=`: reading a growing string's tail flattens its rope each time (O(n^2)).
  const parts: string[] = [];
  let offset = 0;
  let lastChar = "";
  let last = 0;
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
    // Resume right after this opener, so a real span this match straddled is still found.
    if (inSkipZone(match.index) || inSkipZone(matchEnd - 1)) {
      CONVERT_LATEX_DELIM_RE.lastIndex = match.index + 1;
      continue;
    }
    const isDisplay = match[1] !== undefined;
    const body = (isDisplay ? match[1] : match[2]).trim();
    // A bare `$$` would open a stray display block.
    if (!body) {
      continue;
    }
    append(content.slice(last, match.index));
    let wrapped: string;
    if (isDisplay) {
      // Keep a whitespace-prefixed opener's indent so `$$` stays inside a list item.
      const lineStart =
        match.index > 0 ? content.lastIndexOf("\n", match.index - 1) + 1 : 0;
      const prefix = content.slice(lineStart, match.index);
      const indent = /^\s*$/.test(prefix) ? prefix : "";
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

/** converts bracketed LaTeX and protects currency or shell variables from single-dollar math. */
export function preprocessLaTeX(content: string, isStreaming = false): string {
  const { text, mathRegions } = convertLatexDelimiters(content);

  if (!text.includes("$")) return text;

  let proseBudget = VARIABLE_PROSE_BUDGET;
  const codeRegions = findCodeBlockRegions(text);
  const linkRegions = mergeRegions(
    findLinkDestinationRegions(text),
    text.includes("<") || text.includes("://") || text.includes("www.")
      ? findAutolinkRegions(text)
      : [],
  );
  const rawTextRegions = findRawTextRegions(text);
  const inCode = regionCursor(codeRegions);
  const inLink = regionCursor(linkRegions);
  const inMath = regionCursor(mathRegions);
  const inRawText = regionCursor(rawTextRegions);
  let closer = -1;
  let lineStart = 0;
  let nextNewline = text.indexOf("\n");
  let tableRow: boolean | null = null;

  const rewrite = (offset: number): string => {
    // evaluate all cursors so each advances past `offset`.
    const code = inCode(offset);
    const link = inLink(offset);
    const math = inMath(offset);
    if (code || link) {
      return "$";
    }
    // preserve converted numeric math to avoid currency re-escaping
    if (math) {
      return "$";
    }
    const digit = text.charCodeAt(offset + 1) - 48;
    CURRENCY_REGEX.lastIndex = offset + 1;
    const currency = digit >= 0 && digit <= 9 && CURRENCY_REGEX.test(text);
    if (currency && !hasInlineMathCloser(text, offset, mathRegions)) {
      return "\\$";
    }
    if (offset === closer) {
      return "$";
    }
    while (nextNewline !== -1 && nextNewline < offset) {
      lineStart = nextNewline + 1;
      nextNewline = text.indexOf("\n", lineStart);
      tableRow = null;
    }
    if (tableRow === null) {
      TABLE_ROW_RE.lastIndex = lineStart;
      tableRow = TABLE_ROW_RE.test(text);
    }
    const next = findInlineMathCloser(
      text,
      offset,
      lineStart,
      nextNewline,
      tableRow,
    );
    if (
      !currency &&
      next !== -1 &&
      startsLikeVariable(text, offset + 1) &&
      // a closer at the end of a streaming reply may still be followed by a name.
      (next + 1 === text.length
        ? isStreaming
        : NEW_TOKEN_RE.test(text[next + 1])) &&
      next - offset - 1 <= MAX_VARIABLE_PROSE_SPAN &&
      !inRawText(offset) &&
      (proseBudget -= next - offset - 1) >= 0 &&
      VARIABLE_PROSE_RE.test(text.slice(offset + 1, next))
    ) {
      return VARIABLE_DOLLAR;
    }
    closer = next;
    return "$";
  };

  // a single `$`: not escaped, not part of `$$`. A plain scan, since a regex replace
  // callback per dollar dominated the streaming re-render cost.
  const parts: string[] = [];
  let last = 0;
  for (
    let offset = text.indexOf("$");
    offset !== -1;
    offset = text.indexOf("$", offset + 1)
  ) {
    const prev = text[offset - 1];
    if (prev === "\\" || prev === "$" || text[offset + 1] === "$") continue;
    const out = rewrite(offset);
    if (out !== "$") {
      parts.push(text.slice(last, offset), out);
      last = offset + 1;
    }
  }
  if (last === 0) return text;
  parts.push(text.slice(last));
  return parts.join("");
}
