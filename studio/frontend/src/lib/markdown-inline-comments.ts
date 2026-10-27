// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * A mid-sentence HTML comment is inline raw HTML: its `-->` may arrive later in the same
 * paragraph, but past the paragraph the `<!--` is ordinary text.
 */

import { interruptsParagraph } from "@/lib/markdown-list-columns";

const COMMENT_CLOSE = "-->";
// Lines that end a paragraph. Leading punctuation is not one: `-->` alone is the usual close.
// Indented code and link definitions are absent: neither may interrupt (spec 0.31.2 4.4, 4.7).
const BLANK = /^[ \t]*$/;
const ATX_HEADING = /^ {0,3}#{1,6}([ \t]|$)/;
const FENCE = /^ {0,3}(?:`{3,}|~{3,})/;
const THEMATIC_BREAK =
  /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
const SETEXT_UNDERLINE = /^ {0,3}(?:=+|-+)[ \t]*$/;
// HTML block type 7 cannot interrupt a paragraph, but treating it as a break is harmless here.
const HTML_LINE = /^ {0,3}</;

function startsBlock(line: string): boolean {
  return (
    BLANK.test(line) ||
    ATX_HEADING.test(line) ||
    FENCE.test(line) ||
    THEMATIC_BREAK.test(line) ||
    SETEXT_UNDERLINE.test(line) ||
    HTML_LINE.test(line) ||
    interruptsParagraph(line)
  );
}

/** Per line, whether a `-->` is reachable within its paragraph; read at `index + 1` for `index`. */
export function commentClosesBelow(lines: string[]): boolean[] {
  const closes: boolean[] = new Array(lines.length + 1).fill(false);
  for (let at = lines.length - 1; at >= 0; at -= 1) {
    const line = lines[at] ?? "";
    closes[at] =
      !startsBlock(line) &&
      (line.includes(COMMENT_CLOSE) || (closes[at + 1] ?? false));
  }
  return closes;
}
