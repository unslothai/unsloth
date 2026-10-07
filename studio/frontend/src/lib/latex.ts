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

/** Currency like $5, $1,000, $5.99, $100K; not $$, \$ or $\alpha. */
const CURRENCY_REGEX =
  /(?<![\\$])\$(?!\$)(?=\d+(?:,\d{3})*(?:\.\d+)?[KMBkmb]?(?:\s|$|[^a-zA-Z\d]))/g;

/** Inputs must each ascend by start, or spans are silently dropped. */
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

  // An inline span can contain a fence, which broke the binary search.
  return mergeRegions(blocks, inline);
}

/** Group 1 is the destination, read via the `d` flag since the text may contain `\](`. */
const LINK_DEST_RE =
  /!?\[(?:\\.|[^\]\\])*?\]\(((?:\\.|[^()\\]|\([^()]*\))*)\)/dg;

/** Destinations only, so escaped parens in a URL are not math but link text still converts. */
function findLinkDestinationRegions(content: string): Array<[number, number]> {
  if (!content.includes("](")) return [];
  const regions: Array<[number, number]> = [];
  let match: RegExpExecArray | null;
  LINK_DEST_RE.lastIndex = 0;
  while ((match = LINK_DEST_RE.exec(content)) !== null) {
    regions.push(match.indices![1]);
  }
  return regions;
}

/** Regions must be sorted by start and non-overlapping. */
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

/** The body is capped so repeated incomplete openers stay linear during streaming. */
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

/**
 * - `\[E = mc^2\]` becomes a `$$` display block on its own lines
 * - `\(\alpha\)` becomes `$\alpha$`; inside a code span it is untouched
 * - `$5` alone becomes `\$5`
 * - `$\alpha$`, `$30^\circ$`, `**$30^\circ$**` and `$$...$$` are untouched
 */
export function preprocessLaTeX(content: string): string {
  const { text, mathRegions } = convertLatexDelimiters(content);

  if (!text.includes("$")) return text;

  const codeRegions = findCodeBlockRegions(text);

  return text.replace(CURRENCY_REGEX, (match, offset) => {
    if (isInRegion(offset, codeRegions)) {
      return match;
    }
    // Skip spans created from `\(...\)` so `$5$` is not re-escaped.
    if (isInRegion(offset, mathRegions)) {
      return match;
    }
    if (hasInlineMathCloser(text, offset, mathRegions)) {
      return match;
    }
    return "\\" + match;
  });
}
