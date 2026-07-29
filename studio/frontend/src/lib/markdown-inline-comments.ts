// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * An HTML comment written mid-sentence is inline raw HTML, not a block, so it
 * belongs to the paragraph it is written in. Its `-->` may therefore arrive on
 * a later line of that same paragraph, and everything between renders as
 * nothing; past the paragraph the `<!--` is the ordinary text a renderer shows.
 * Both changelog scanners need the same answer to "does this opener close
 * before its paragraph ends", so they read it from here.
 *
 * The backend needs none of this: a heading closes the paragraph it is written
 * under, so no heading can ever sit inside one of these comments.
 */

import { interruptsParagraph } from "@/lib/markdown-list-columns";

const COMMENT_CLOSE = "-->";
// A line that cannot be more of the paragraph above it: blank, or the start of
// a block that may interrupt one. A leading punctuation character is not one:
// `-->` on its own line is how a multiline comment is ordinarily closed, and a
// continuation may open with emphasis, so reading either as a break left the
// comment unclosed and its text on show. An indented code block and a link
// reference definition are absent because neither may interrupt a paragraph
// (spec 0.31.2 sections 4.4 and 4.7), so both are more of it.
const BLANK = /^[ \t]*$/;
const ATX_HEADING = /^ {0,3}#{1,6}([ \t]|$)/;
const FENCE = /^ {0,3}(?:`{3,}|~{3,})/;
const THEMATIC_BREAK =
  /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
// A row of `=` or `-` alone makes the paragraph above it a setext heading,
// which ends that paragraph as surely as a block written below it would.
const SETEXT_UNDERLINE = /^ {0,3}(?:=+|-+)[ \t]*$/;
// A tag, a comment or a declaration written at the start of a line. HTML block
// types 1 to 6 interrupt a paragraph; type 7 does not, but reading one as a
// break only leaves the opener as the plain text a single line already makes
// of it, which is what a `<` at the start of a line has always meant here.
const HTML_LINE = /^ {0,3}</;

/** Whether `line` starts a block of its own rather than continuing a paragraph. */
function startsBlock(line: string): boolean {
  return (
    BLANK.test(line) ||
    ATX_HEADING.test(line) ||
    FENCE.test(line) ||
    THEMATIC_BREAK.test(line) ||
    SETEXT_UNDERLINE.test(line) ||
    HTML_LINE.test(line) ||
    // A blockquote, and a list item with content that an ordered marker may
    // only open at 1: the rule the rest of the scanners already share.
    interruptsParagraph(line)
  );
}

/**
 * For each line, whether a `-->` is reachable from it without leaving the
 * paragraph it starts in. Read at `index + 1` it answers whether an inline
 * comment opened on `index` and left unclosed there is a comment at all.
 */
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
