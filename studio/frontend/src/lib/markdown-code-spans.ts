// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Code span boundaries, the way CommonMark defines them: a run of backticks is
 * closed only by a run of exactly the same length. A regular expression cannot
 * express that without lookbehind, which older Safari refuses to parse, so the
 * runs are scanned by hand.
 */

export interface CodeSpan {
  // Offsets of the whole span, delimiters included.
  start: number;
  end: number;
  // Text between the delimiters, with the one space of padding removed.
  content: string;
}

/** Length of the backtick run starting at `index`. */
function runLength(text: string, index: number): number {
  let end = index;
  while (text[end] === "`") {
    end += 1;
  }
  return end - index;
}

/**
 * CommonMark strips one leading and trailing space when the content has both
 * and is not all spaces, so `` ` a ` `` renders as "a".
 */
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

/** Every code span in `text`, in order. Unclosed runs are ordinary text. */
export function codeSpans(text: string): CodeSpan[] {
  const spans: CodeSpan[] = [];
  let index = 0;

  while (index < text.length) {
    if (text[index] !== "`") {
      index += 1;
      continue;
    }
    const ticks = runLength(text, index);
    const contentStart = index + ticks;

    let cursor = contentStart;
    let closed = false;
    while (cursor < text.length) {
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
      // Nothing closes this run, so it is literal text: carry on after it.
      index = contentStart;
    }
  }
  return spans;
}

/** Replaces every code span with `park(content)`, leaving the rest as is. */
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

/** True when `index` falls inside one of `spans`. */
export function insideSpan(spans: CodeSpan[], index: number): boolean {
  return spans.some((span) => index >= span.start && index < span.end);
}
