// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * CommonMark measures a block's indentation from its container, not from the
 * left margin: inside a list item the content column moves right, so four
 * spaces at document level and four spaces under a bullet mean different
 * things. Tracking the open items lets both changelog scanners ask "is this
 * indented code?" the way a renderer would.
 *
 * Ported from the backend's `_open_lists` in studio/backend/utils/changelog.py
 * so the three scanners classify the same line the same way.
 */

/** The open list items, innermost last, by the column their content starts. */
export interface ListState {
  columns: number[];
  // True while the innermost item has had no content since its marker.
  emptyItem: boolean;
}

export const EMPTY_LIST_STATE: ListState = { columns: [], emptyItem: false };

// The marker needs whitespace after it, so `2.0` is a version, not an item.
const LIST_ITEM = /^[ \t]*([-*+]|\d{1,9}[.)])([ \t]+|$)/;
const THEMATIC_BREAK =
  /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
const BLOCK_QUOTE = /^ {0,3}>/;
// Blocks that are not paragraph text, so they cannot continue one lazily.
const PARAGRAPH_TEXT = /^ {0,3}(?![-*+>]([ \t]|$)|\d{1,9}[.)]([ \t]|$))\S/;
const NOT_PARAGRAPH =
  /^ {0,3}(?:#{1,6}([ \t]|$)|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$|\[(?:[^[\]\\]|\\.)+\]:)/;
const SETEXT_UNDERLINE = /^ {0,3}(=+|-+)[ \t]*$/;
const FENCE = /^ {0,3}(?:`{3,}|~{3,})/;
const HTML_BLOCK_OPEN = /^ {0,3}<\/?([a-zA-Z][a-zA-Z0-9-]*)(?=[\s/>]|$)/;
const HTML_BLOCK_TAGS = new Set(
  `address article aside base basefont blockquote body caption center col colgroup
   dd details dialog dir div dl dt fieldset figcaption figure footer form frame
   frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li link main menu
   menuitem nav noframes ol optgroup option p param search section summary table
   tbody td tfoot th thead title tr track ul`.split(/\s+/),
);
// Content indented more than this after a marker is an indented code block, so
// the item's content starts one column past the marker instead.
const MAX_ITEM_PADDING = 4;

/** Columns of leading whitespace, counting a tab to the next stop of four. */
export function indentWidth(line: string): number {
  let width = 0;
  for (const char of line) {
    if (char === " ") {
      width += 1;
    } else if (char === "\t") {
      width += 4 - (width % 4);
    } else {
      break;
    }
  }
  return width;
}

/**
 * Whether `line` starts a block that can break into an open paragraph. A quote
 * marker always can. A list item can only when it has content, and an ordered
 * one only when it starts at 1: anything else is text of the paragraph it
 * appears to interrupt.
 */
function interruptsParagraph(line: string): boolean {
  if (BLOCK_QUOTE.test(line)) {
    return true;
  }
  const item = THEMATIC_BREAK.test(line) ? null : LIST_ITEM.exec(line);
  if (item === null) {
    return false;
  }
  const marker = item[1] ?? "";
  if (!line.slice(item[0].length).trim()) {
    return false;
  }
  const ordered = marker.endsWith(".") || marker.endsWith(")");
  return !ordered || marker.slice(0, -1) === "1";
}

/**
 * Whether a marker-shaped line is really text of the paragraph above it. Only
 * a marker inside the paragraph's own item interrupts it; one to the left
 * closes that item and opens a sibling.
 */
function continuesItem(
  line: string,
  columns: number[],
  indent: number,
): boolean {
  const inside = columns.length === 0 || indent >= (columns.at(-1) ?? 0);
  return inside && !interruptsParagraph(line);
}

/** `columns` with every item whose content starts past `indent` closed. */
function dropDeeper(columns: number[], indent: number): number[] {
  let open = columns.length;
  while (open > 0 && (columns[open - 1] ?? 0) > indent) {
    open -= 1;
  }
  return open === columns.length ? columns : columns.slice(0, open);
}

/**
 * Whether `line` can continue a paragraph it is indented out of. Only plain
 * text can: a heading, a fence, a break, an underline or an HTML block starts a
 * block of its own, which closes the item instead.
 */
function mayBeLazy(line: string): boolean {
  const named = HTML_BLOCK_OPEN.exec(line);
  // Types 1 to 6 interrupt a paragraph, so a `<div>` left of an open item
  // closes it. Type 7 cannot, and is deliberately excluded here.
  const htmlBlock =
    named !== null && HTML_BLOCK_TAGS.has((named[1] ?? "").toLowerCase());
  return (
    PARAGRAPH_TEXT.test(line) &&
    !NOT_PARAGRAPH.test(line) &&
    !SETEXT_UNDERLINE.test(line) &&
    !FENCE.test(line) &&
    !htmlBlock
  );
}

/**
 * The list items still open after `line`. A dedented line closes an item,
 * unless it is a lazy paragraph continuation. A new marker nests under a deeper
 * column and replaces a sibling.
 */
export function openLists(
  line: string,
  state: ListState,
  afterParagraph: boolean,
): ListState {
  let columns = state.columns;
  if (!line.trim()) {
    // A blank line leaves the list open, unless the item is still empty: an
    // item may begin with one blank line, and later content is outside it.
    return {
      columns: state.emptyItem ? columns.slice(0, -1) : columns,
      emptyItem: false,
    };
  }
  const indent = indentWidth(line);
  const item = THEMATIC_BREAK.test(line) ? null : LIST_ITEM.exec(line);
  const empty = item !== null && !line.slice(item[0].length).trim();
  if (item !== null && afterParagraph && continuesItem(line, columns, indent)) {
    // A lazy continuation or an underline, so the open items are untouched.
    return state;
  }
  if (!(afterParagraph && mayBeLazy(line))) {
    // A dedented line closes every item whose content starts to its right.
    columns = dropDeeper(columns, indent);
  }
  if (item === null) {
    return { columns, emptyItem: false };
  }
  const marker = item[1] ?? "";
  let padding = indentWidth(item[2] ?? "");
  if (padding === 0 || padding > MAX_ITEM_PADDING) {
    // An empty or over-indented item still holds one column of content.
    padding = 1;
  }
  // A sibling marker replaces the item it lines up with.
  return {
    columns: [...dropDeeper(columns, indent), indent + marker.length + padding],
    emptyItem: empty,
  };
}
