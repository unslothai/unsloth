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
const QUOTE_MARKER = /^ {0,3}>[ \t]?/;
// Blocks that are not paragraph text, so they cannot continue one lazily.
const PARAGRAPH_TEXT = /^ {0,3}(?![-*+>]([ \t]|$)|\d{1,9}[.)]([ \t]|$))\S/;
// Blocks that break into an open paragraph, so one they are written below is
// closed rather than continued. A link reference definition is not one of them.
const INTERRUPTS =
  /^ {0,3}(?:#{1,6}([ \t]|$)|(?:\*[ \t]*){3,}$|(?:-[ \t]*){3,}$|(?:_[ \t]*){3,}$)/;
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
// Columns past its container at which a line becomes an indented code block.
const INDENTED_CODE = 4;
// Stands in for a line the renderer hides. `#` is a block in its own right, so
// list tracking reads it the way it reads a comment: never a marker, never a
// lazy paragraph continuation.
const HIDDEN_BLOCK = "#";
const LEADING_SPACE = /^[ \t]*/;

/**
 * `line` as list tracking sees it once the renderer hides its text. A comment
 * or a raw HTML block renders nothing, but it is still a block written at its
 * own column, so it closes the items it sits to the left of. Only the
 * indentation survives: what the block hides is not Markdown and must not open
 * a list of its own. Ported from `_hidden_structure` on the backend.
 */
export function hiddenStructure(line: string): string {
  const indent = LEADING_SPACE.exec(line)?.[0] ?? "";
  return line.trim() ? `${indent}${HIDDEN_BLOCK}` : "";
}

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
 * Whether a marker-shaped `line` is really text of the paragraph above it. Only
 * a marker inside the paragraph's own item interrupts it; one to the left
 * closes that item and opens a sibling. A quote owns the paragraph its lines
 * hold, so a marker written outside the quote opens a list of its own.
 */
export function lazyMarker(
  line: string,
  state: ListState,
  afterParagraph: boolean,
  quoted: boolean,
): boolean {
  const item = THEMATIC_BREAK.test(line) ? null : LIST_ITEM.exec(line);
  const columns = state.columns;
  const inside =
    columns.length === 0 || indentWidth(line) >= (columns.at(-1) ?? 0);
  return (
    item !== null &&
    afterParagraph &&
    !quoted &&
    inside &&
    !interruptsParagraph(line)
  );
}

/** `columns` with every item whose content starts past `indent` closed. */
function dropDeeper(columns: number[], indent: number): number[] {
  let open = columns.length;
  while (open > 0 && (columns[open - 1] ?? 0) > indent) {
    open -= 1;
  }
  return open === columns.length ? columns : columns.slice(0, open);
}

/** `line` with up to `columns` columns of leading whitespace removed. */
function stripIndent(line: string, columns: number): string {
  let width = 0;
  let index = 0;
  while (index < line.length && width < columns) {
    const char = line[index];
    if (char !== " " && char !== "\t") {
      break;
    }
    width += char === " " ? 1 : 4 - (width % 4);
    index += 1;
  }
  return line.slice(index);
}

/**
 * Whether `line` can continue a paragraph it is indented out of. Only plain
 * text can: a heading, a fence, a break or an HTML block starts a block of its
 * own, which closes the item instead. An underline is not one of them: it may
 * never be lazy, so `===` written left of an open item is read as more of the
 * item's paragraph. Nor is a definition, which is a block of its own but may
 * not interrupt a paragraph. A row of dashes still closes the item, as
 * `INTERRUPTS` reads three or more as the thematic break they are.
 */
function mayBeLazy(line: string): boolean {
  const named = HTML_BLOCK_OPEN.exec(line);
  // Types 1 to 6 interrupt a paragraph, so a `<div>` left of an open item
  // closes it. Type 7 cannot, and is deliberately excluded here.
  const htmlBlock =
    named !== null && HTML_BLOCK_TAGS.has((named[1] ?? "").toLowerCase());
  return (
    PARAGRAPH_TEXT.test(line) &&
    !INTERRUPTS.test(line) &&
    !FENCE.test(line) &&
    !htmlBlock
  );
}

/**
 * Whether `line` reads as more of a paragraph open in its container. Measured
 * from `column`, where that container's content starts: four columns past it
 * the line is an indented code block, which may not interrupt a paragraph, so
 * indentation alone never closes the one above it.
 */
export function continuesParagraph(line: string, column: number): boolean {
  const inner = stripIndent(line, column);
  return indentWidth(inner) >= INDENTED_CODE || mayBeLazy(inner);
}

/** What a blockquote line holds, with its markers stripped. */
function quoteContent(line: string): string {
  let rest = line;
  let marker = QUOTE_MARKER.exec(rest);
  while (marker !== null) {
    rest = rest.slice(marker[0].length);
    marker = QUOTE_MARKER.exec(rest);
  }
  return rest;
}

/** Whether a blockquote owns the paragraph the line below could continue. */
export interface QuoteState {
  // True while a quoted paragraph is open, so plain text below is more of it.
  inQuote: boolean;
  // True whenever that paragraph is the quote's rather than the document's.
  quoted: boolean;
}

export const NO_QUOTE: QuoteState = { inQuote: false, quoted: false };

/**
 * The quote state after `line`, given the state after the line above and the
 * content column of the item `line` sits in. A quote owns the paragraph its own
 * lines hold, so a list marker written outside the quote opens a list of its own
 * instead of reading as more of that paragraph. Ported from the backend's
 * `in_quote` tracking in changelog.py.
 */
export function quoteState(
  line: string,
  inQuote: boolean,
  column = 0,
): QuoteState {
  if (BLOCK_QUOTE.test(line)) {
    // An empty quote holds no paragraph, so the line below starts a new one.
    return { inQuote: mayBeLazy(quoteContent(line)), quoted: true };
  }
  const open = inQuote && continuesParagraph(line, column);
  return { inQuote: open, quoted: open };
}

/**
 * `columns` with every item `line` is written to the left of closed. Read
 * inside the container the item sits in, not from the margin: a line that only
 * looks indented there is lazy text of the item's paragraph, which leaves the
 * item open rather than closing it.
 */
function closeDedented(
  columns: number[],
  line: string,
  indent: number,
  afterParagraph: boolean,
): number[] {
  let open = columns.length;
  while (open > 0 && (columns[open - 1] ?? 0) > indent) {
    const outer = open > 1 ? (columns[open - 2] ?? 0) : 0;
    if (afterParagraph && continuesParagraph(line, outer)) {
      break;
    }
    open -= 1;
  }
  return open === columns.length ? columns : columns.slice(0, open);
}

/**
 * The list items still open after `line`. A dedented line closes an item,
 * unless it is a lazy paragraph continuation. A new marker nests under a deeper
 * column and replaces a sibling. `quoted` marks a paragraph the blockquote
 * above owns: a marker written outside the quote is not text of it, so it opens
 * a list of its own.
 */
export function openLists(
  line: string,
  state: ListState,
  afterParagraph: boolean,
  quoted = false,
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
  if (lazyMarker(line, state, afterParagraph, quoted)) {
    // A lazy continuation or an underline, so the open items are untouched.
    return state;
  }
  columns = closeDedented(columns, line, indent, afterParagraph);
  // Four columns past its container the marker is an indented code block, or
  // lazy text of the paragraph above it, so it opens no list of its own.
  if (item === null || indent - (columns.at(-1) ?? 0) >= INDENTED_CODE) {
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
