// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** CommonMark code spans, scanned by hand since older Safari rejects lookbehind. */

export interface CodeSpan {
  start: number;
  end: number;
  // Between the delimiters, with one space of padding removed.
  content: string;
}

const BLOCK_BOUNDARY_RE =
  /(?:\r?\n[ \t]*\r?\n|(?:^|[\r\n]) {0,3}(?:#{1,6}(?:[ \t]|$)|>|[-+*](?:[ \t]|$)|\d{1,9}[.)](?:[ \t]|$)|`{3,}|~{3,}|(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,}|=+[ \t]*)(?=$|[\r\n])))/;

function runLength(text: string, index: number): number {
  let end = index;
  while (text[end] === "`") {
    end += 1;
  }
  return end - index;
}

function escaped(text: string, index: number): boolean {
  let slashes = 0;
  while (text[index - 1 - slashes] === "\\") {
    slashes += 1;
  }
  return slashes % 2 === 1;
}

function stripPadding(content: string): string {
  if (
    content.length > 1 &&
    content.startsWith(" ") &&
    content.endsWith(" ") &&
    content.trim() !== ""
  ) {
    return content.slice(1, -1);
  }
  return content;
}

function scanCodeSpans(text: string, includeOpenTail: boolean): CodeSpan[] {
  const spans: CodeSpan[] = [];
  let index = 0;

  while (index < text.length) {
    if (text[index] !== "`" || escaped(text, index)) {
      index += 1;
      continue;
    }
    const ticks = runLength(text, index);
    const contentStart = index + ticks;

    let cursor = contentStart;
    let closed = false;
    while (cursor < text.length) {
      // Escapes do not apply inside a span, so a run after a backslash closes it.
      if (text[cursor] !== "`") {
        cursor += 1;
        continue;
      }
      const candidate = runLength(text, cursor);
      if (candidate === ticks) {
        spans.push({
          start: index,
          end: cursor + ticks,
          content: stripPadding(text.slice(contentStart, cursor)),
        });
        index = cursor + ticks;
        closed = true;
        break;
      }
      cursor += candidate;
    }
    if (!closed) {
      if (includeOpenTail && ticks < 3) {
        spans.push({
          start: index,
          end: text.length,
          content: stripPadding(text.slice(contentStart)),
        });
        break;
      }
      index = contentStart;
    }
  }
  return spans;
}

export function codeSpans(text: string): CodeSpan[] {
  return scanCodeSpans(text, false);
}

/** Closed spans plus the unmatched tail that incomplete-Markdown repair makes code. */
export function codeSpansWithOpenTail(text: string): CodeSpan[] {
  return scanCodeSpans(text, true);
}

export function openCodeSpanTail(text: string): CodeSpan | undefined {
  const closed = codeSpans(text);
  const withOpenTail = codeSpansWithOpenTail(text);
  const tail = withOpenTail.at(-1);
  if (
    !tail ||
    closed.some((span) => span.start === tail.start && span.end === tail.end) ||
    crossesCodeSpanBlockBoundary(text.slice(tail.start, tail.end))
  ) {
    return undefined;
  }
  return tail;
}

export function crossesCodeSpanBlockBoundary(text: string): boolean {
  return BLOCK_BOUNDARY_RE.test(text);
}

export function parkCodeSpans(
  text: string,
  park: (content: string) => string,
): string {
  const spans = codeSpans(text);
  if (spans.length === 0) {
    return text;
  }
  let out = "";
  let cursor = 0;
  for (const span of spans) {
    out += text.slice(cursor, span.start) + park(span.content);
    cursor = span.end;
  }
  return out + text.slice(cursor);
}

export function insideSpan(spans: CodeSpan[], index: number): boolean {
  let low = 0;
  let high = spans.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    const span = spans[mid];
    if (span === undefined || index < span.start) {
      high = mid - 1;
    } else if (index >= span.end) {
      low = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}
