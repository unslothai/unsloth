// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * CommonMark measures indentation from the container, not the margin. Ported from `_open_lists`
 * in studio/backend/utils/release_notes.py so all three scanners classify lines alike.
 */

export interface ListState {
  // Content columns of the open items, innermost last.
  columns: number[];
  emptyItem: boolean;
}

export const EMPTY_LIST_STATE: ListState = { columns: [], emptyItem: false };

// The marker needs whitespace after it, so `2.0` is a version, not an item.
const LIST_ITEM = /^[ \t]*([-*+]|\d{1,9}[.)])([ \t]+|$)/;
const THEMATIC_BREAK =
  /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
const BLOCK_QUOTE = /^ {0,3}>/;
const QUOTE_MARKER = /^ {0,3}>[ \t]?/;
const PARAGRAPH_TEXT = /^ {0,3}(?![-*+>]([ \t]|$)|\d{1,9}[.)]([ \t]|$))\S/;
// A link reference definition does not interrupt a paragraph.
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
// Content padded past this after a marker is indented code, so content starts one column in.
const MAX_ITEM_PADDING = 4;
const INDENTED_CODE = 4;
// Stands in for a hidden line: `#` is never a marker nor a lazy continuation.
const HIDDEN_BLOCK = "#";
const LEADING_SPACE = /^[ \t]*/;

/** A hidden comment or HTML block keeps only its indent and item marker. Ports `_hidden_structure`. */
export function hiddenStructure(line: string, marker = ""): string {
  if (marker) {
    return `${marker}${HIDDEN_BLOCK}`;
  }
  const indent = LEADING_SPACE.exec(line)?.[0] ?? "";
  return line.trim() ? `${indent}${HIDDEN_BLOCK}` : "";
}

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

/** A quote marker always interrupts; a list item only with content, an ordered one only at 1. */
export function interruptsParagraph(line: string): boolean {
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

/** Only a marker inside the paragraph's own item is lazy text; one to the left opens a sibling. */
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

function dropDeeper(columns: number[], indent: number): number[] {
  let open = columns.length;
  while (open > 0 && (columns[open - 1] ?? 0) > indent) {
    open -= 1;
  }
  return open === columns.length ? columns : columns.slice(0, open);
}

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

/** Only plain text can be lazy. `===` stays item text; three or more dashes close the item. */
function mayBeLazy(line: string): boolean {
  const named = HTML_BLOCK_OPEN.exec(line);
  // HTML block type 7 cannot interrupt a paragraph, so it is deliberately excluded.
  const htmlBlock =
    named !== null && HTML_BLOCK_TAGS.has((named[1] ?? "").toLowerCase());
  return (
    PARAGRAPH_TEXT.test(line) &&
    !INTERRUPTS.test(line) &&
    !FENCE.test(line) &&
    !htmlBlock
  );
}

/** Indented code may not interrupt a paragraph, so indentation alone never closes one. */
export function continuesParagraph(line: string, column: number): boolean {
  const inner = stripIndent(line, column);
  return indentWidth(inner) >= INDENTED_CODE || mayBeLazy(inner);
}

function stripQuotes(line: string, depth: number): [string, number] {
  let rest = line;
  let removed = 0;
  let marker = removed < depth ? QUOTE_MARKER.exec(rest) : null;
  while (marker !== null) {
    rest = rest.slice(marker[0].length);
    removed += 1;
    marker = removed < depth ? QUOTE_MARKER.exec(rest) : null;
  }
  return [rest, removed];
}

function quoteContent(line: string): string {
  return stripQuotes(line, Number.POSITIVE_INFINITY)[0];
}

export function quoteDepth(line: string): number {
  return stripQuotes(line, Number.POSITIVE_INFINITY)[1];
}

/** Measured from the container (spec 0.31.2 5.1, 5.2), so `> ~~~` still opens a fence. */
export function containerContent(
  line: string,
  state: ListState,
  quotes: number,
): string {
  const [inner] = stripQuotes(line, quotes);
  if (quotes > 0) {
    // This tracker follows document level only, so its columns do not apply inside a quote.
    return inner;
  }
  const columns = dropDeeper(state.columns, indentWidth(inner));
  return stripIndent(inner, columns.at(-1) ?? 0);
}

/** Padding is capped like `openLists`, so ``-     ``` `` stays indented code, not a fence. */
export function itemContent(line: string, afterParagraph: boolean): string {
  if (
    indentWidth(line) >= INDENTED_CODE ||
    (afterParagraph && !interruptsParagraph(line))
  ) {
    return line;
  }
  const item = THEMATIC_BREAK.test(line) ? null : LIST_ITEM.exec(line);
  if (item === null) {
    return line;
  }
  const padding = indentWidth(item[2] ?? "");
  const over = padding > MAX_ITEM_PADDING ? padding - 1 : 0;
  return `${" ".repeat(over)}${line.slice(item[0].length)}`;
}

export interface QuoteState {
  inQuote: boolean;
  // True when the open paragraph is the quote's rather than the document's.
  quoted: boolean;
}

export const NO_QUOTE: QuoteState = { inQuote: false, quoted: false };

/** A quote owns its paragraph, so a marker outside the quote opens a new list. */
export function quoteState(
  line: string,
  inQuote: boolean,
  column = 0,
): QuoteState {
  if (BLOCK_QUOTE.test(line)) {
    return { inQuote: mayBeLazy(quoteContent(line)), quoted: true };
  }
  const open = inQuote && continuesParagraph(line, column);
  return { inQuote: open, quoted: open };
}

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

function sameLineListColumns(
  line: string,
  firstItem: RegExpExecArray,
  firstIndent: number,
  outerColumns: number[],
): number[] {
  let item = firstItem;
  let itemIndent = firstIndent;
  let rest = line;
  const columns = [...outerColumns];

  while (true) {
    const marker = item[1] ?? "";
    const rawPadding = indentWidth(item[2] ?? "");
    const padding =
      rawPadding === 0 || rawPadding > MAX_ITEM_PADDING ? 1 : rawPadding;
    const contentColumn = itemIndent + marker.length + padding;
    columns.push(contentColumn);
    if (rawPadding === 0 || rawPadding > MAX_ITEM_PADDING) {
      break;
    }

    rest = rest.slice(item[0].length);
    const relativeIndent = indentWidth(rest);
    if (relativeIndent >= INDENTED_CODE || THEMATIC_BREAK.test(rest)) {
      break;
    }
    const nested = LIST_ITEM.exec(rest);
    if (nested === null) {
      break;
    }
    item = nested;
    itemIndent = contentColumn + relativeIndent;
  }
  return columns;
}

export function openLists(
  line: string,
  state: ListState,
  afterParagraph: boolean,
  quoted = false,
): ListState {
  let columns = state.columns;
  if (!line.trim()) {
    // An item may begin with one blank line; content after that is outside it.
    return {
      columns: state.emptyItem ? columns.slice(0, -1) : columns,
      emptyItem: false,
    };
  }
  const indent = indentWidth(line);
  const item = THEMATIC_BREAK.test(line) ? null : LIST_ITEM.exec(line);
  const empty = item !== null && !line.slice(item[0].length).trim();
  if (lazyMarker(line, state, afterParagraph, quoted)) {
    return state;
  }
  columns = closeDedented(columns, line, indent, afterParagraph);
  if (item === null || indent - (columns.at(-1) ?? 0) >= INDENTED_CODE) {
    return { columns, emptyItem: false };
  }
  return {
    columns: sameLineListColumns(
      line,
      item,
      indent,
      dropDeeper(columns, indent),
    ),
    emptyItem: empty,
  };
}
