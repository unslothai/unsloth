


/**
 * CommonMark measures a block's indentation from its container, not the left
 * margin: four spaces at document level and four under a bullet mean different
 * things. Tracking the open items lets both release-note scanners ask "is this
 * indented code?" the way a renderer would.
 *
 * Ported from `_open_lists` in studio/backend/utils/release_notes.py so the three
 * scanners classify a line the same way.
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
// Blocks that break into an open paragraph, closing it rather than continuing
// it. A link reference definition is not one of them.
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
// Stands in for a line the renderer hides. `#` is a block of its own, so list
// tracking reads it like a comment: never a marker, never a lazy continuation.
const HIDDEN_BLOCK = "#";
const LEADING_SPACE = /^[ \t]*/;

/**
 * `line` as list tracking sees it once the renderer hides its text. A comment or
 * raw HTML block renders nothing but is still a block at its own column, so it
 * closes the items it sits left of. Only the indentation survives: what the block
 * hides is not Markdown and must not open a list. `marker` is the part opening
 * the item the block is content of, which survives too. Ported from
 * `_hidden_structure` on the backend.
 */
export function hiddenStructure(line: string, marker = ""): string {
  if (marker) {
    return `${marker}${HIDDEN_BLOCK}`;
  }
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
 * marker always can; a list item only with content, an ordered one only at 1.
 * Anything else is text of the paragraph it appears to interrupt.
 */
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

/**
 * Whether a marker-shaped `line` is really text of the paragraph above. Only a
 * marker inside the paragraph's own item interrupts it; one to the left closes
 * that item and opens a sibling. A quote owns the paragraph its lines hold, so a
 * marker outside the quote opens a list of its own.
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
 * Whether `line` can continue a paragraph it is indented out of. Only plain text
 * can: a heading, fence, break or HTML block starts a block of its own, closing
 * the item instead. An underline is not one: it may never be lazy, so `===` left
 * of an open item is more of the item's paragraph. Nor is a definition, a block
 * of its own that may not interrupt a paragraph. A row of dashes still closes the
 * item: `INTERRUPTS` reads three or more as the thematic break they are.
 */
function mayBeLazy(line: string): boolean {
  const named = HTML_BLOCK_OPEN.exec(line);
  // Types 1 to 6 interrupt a paragraph, so a `<div>` left of an open item closes
  // it. Type 7 cannot, and is deliberately excluded.
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
 * Whether `line` reads as more of a paragraph open in its container, measured
 * from `column` where that container's content starts: four columns past it the
 * line is indented code, which may not interrupt a paragraph, so indentation
 * alone never closes the one above.
 */
export function continuesParagraph(line: string, column: number): boolean {
  const inner = stripIndent(line, column);
  return indentWidth(inner) >= INDENTED_CODE || mayBeLazy(inner);
}

/** `line` with up to `depth` blockquote markers removed, and how many went. */
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

/** What a blockquote line holds, with its markers stripped. */
function quoteContent(line: string): string {
  return stripQuotes(line, Number.POSITIVE_INFINITY)[0];
}

/** How many blockquotes `line` is written inside. */
export function quoteDepth(line: string): number {
  return stripQuotes(line, Number.POSITIVE_INFINITY)[1];
}

/**
 * `line` as the container it is written in sees it, with `quotes` blockquote
 * markers and the open item's content column removed. CommonMark measures a block
 * from its container, not the margin (spec 0.31.2 sections 5.1, 5.2), so `> ~~~`
 * and a fence under a nested bullet are openers despite sitting more than three
 * columns in.
 */
export function containerContent(
  line: string,
  state: ListState,
  quotes: number,
): string {
  const [inner] = stripQuotes(line, quotes);
  if (quotes > 0) {
    // A list inside a quote is the quote's own; this tracker follows document
    // level only, so its columns do not apply here.
    return inner;
  }
  const columns = dropDeeper(state.columns, indentWidth(inner));
  return stripIndent(inner, columns.at(-1) ?? 0);
}

/**
 * `line` read from the content column of a list item that opens on it. A block
 * written as an item's first content sits inside that item, so ``- ``` `` opens a
 * fence even though its marker is not within three columns of the container (spec
 * 0.31.2 section 5.2). Padding is capped the way `openLists` caps it, or
 * ``-     ``` `` would read as a fence rather than the indented code it is. A
 * marker the paragraph above swallows opens no item, so its line is returned
 * whole, as is one four columns past its container.
 */
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
  // Over-indented content starts one column past the marker; the rest of the
  // padding is the content's own indentation.
  const over = padding > MAX_ITEM_PADDING ? padding - 1 : 0;
  return `${" ".repeat(over)}${line.slice(item[0].length)}`;
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
 * lines hold, so a marker written outside the quote opens a list of its own
 * rather than reading as more of that paragraph. Ported from `in_quote` tracking
 * in release_notes.py.
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
 * `columns` with every item `line` is written to the left of closed. Read inside
 * the container the item sits in, not from the margin: a line that only looks
 * dedented there is lazy text of the item's paragraph, leaving the item open.
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
 * The list items still open after `line`. A dedented line closes an item unless
 * it is a lazy paragraph continuation. A new marker nests under a deeper column
 * and replaces a sibling. `quoted` marks a paragraph the blockquote above owns:
 * a marker outside the quote is not text of it, so it opens a list of its own.
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
