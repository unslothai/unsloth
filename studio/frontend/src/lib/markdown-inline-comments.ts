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

const COMMENT_CLOSE = "-->";
// A line that cannot be more of the paragraph above it: blank, or the start of
// a block of its own. Read generously, since calling a continuation a break
// only leaves the opener as the plain text a single line already makes of it.
const MAY_START_BLOCK =
  /^[ \t]*(?:$|[<>=*+_|-]|`{3,}|~{3,}|#{1,6}([ \t]|$)|\d{1,9}[.)]([ \t]|$))/;

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
      !MAY_START_BLOCK.test(line) &&
      (line.includes(COMMENT_CLOSE) || (closes[at + 1] ?? false));
  }
  return closes;
}
