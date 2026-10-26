// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { codeSpans, parkCodeSpans } from "@/lib/markdown-code-spans";
import { commentClosesBelow } from "@/lib/markdown-inline-comments";
import {
  EMPTY_LIST_STATE,
  type ListState,
  NO_QUOTE,
  type QuoteState,
  hiddenStructure,
  indentWidth,
  itemContent,
  openLists,
  quoteState,
} from "@/lib/markdown-list-columns";

export const RELEASE_NOTES_PREVIEW_ITEMS = 4;
const PREVIEW_ITEM_MAX_CHARS = 120;
const NESTED_INDENT_TOLERANCE = 1;
const TAB_WIDTH = 4;
const INDENTED_CODE_INDENT = 4;

const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
// Matches _HEADING_PATTERN; a whitespace class would match NBSP, and a bare `##` still ends a bullet.
const HEADING = /^#{1,6}(?:[ \t]|$)/;
const BULLET = /^(?:[-*+]|(\d{1,9})[.)])[ \t]+(.*)$/;
const BLOCKQUOTE = /^ {0,3}>[ \t]?/;
const TABLE_DELIMITER_CELL = /^:?-+:?$/;
const THEMATIC_BREAK =
  /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
const DESTINATION = "\\((?:\\\\.|[^()\\\\]|\\([^()]*\\))*\\)";
const LABEL = "((?:[^\\[\\]\\\\]|\\\\.|\\[(?:[^\\[\\]\\\\]|\\\\.)*\\])*)";
const IMAGE = new RegExp(`!\\[${LABEL}\\]${DESTINATION}`, "g");
const LINK = new RegExp(`\\[${LABEL}\\]${DESTINATION}`, "g");
const IMAGE_REFERENCE = new RegExp(`!\\[${LABEL}\\](?:\\[([^\\]]*)\\])?`, "g");
const LINK_REFERENCE = new RegExp(`\\[${LABEL}\\](?:\\[([^\\]]*)\\])?`, "g");
const DEFINITION = /^ {0,3}\[((?:[^\[\]\\]|\\.)+)\]:/;
const ESCAPE = /\\([!-/:-@[-`{-~])/g;
// Private-use sentinels park code spans, so document text cannot contain them.
const SENTINELS = /[\uE000\uE001]/g;
const LINE_ENDINGS = /\r\n?/g;
const TABS = /\t/g;
// A name character must follow "<", so "Python <3.15 and >3.9" keeps its operators.
const HTML_TAG = /<\/?[a-zA-Z][^>]*>/g;
const AUTOLINK = /<([a-zA-Z][a-zA-Z0-9+.-]*:[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>/g;
// Type 1 HTML blocks end at any of the closing tags, not only the opener's.
const RAW_HTML_OPEN = /^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)/i;
const RAW_HTML_CLOSE = /<\/(pre|script|style|textarea)\s*>/i;
const RAW_BLOCKS: [RegExp, RegExp][] = [
  [RAW_HTML_OPEN, RAW_HTML_CLOSE],
  [/^ {0,3}<\?/, /\?>/],
  [/^ {0,3}<!\[CDATA\[/, /\]\]>/],
  // A declaration needs an uppercase letter, so `<!note` stays ordinary text.
  [/^ {0,3}<![A-Z]/, />/],
];
// Type 6 and 7 blocks run to the next blank line; type 7 cannot interrupt a paragraph.
const HTML_BLOCK_OPEN = /^ {0,3}<\/?([a-zA-Z][a-zA-Z0-9-]*)(?=[\s/>]|$)/;
const HTML_ATTRIBUTE =
  "(?:\\s+[a-zA-Z_:][a-zA-Z0-9_.:-]*(?:\\s*=\\s*(?:[^\\s\"'=<>`]+|'[^']*'|\"[^\"]*\"))?)";
const HTML_TAG_ONLY_LINE = new RegExp(
  `^ {0,3}(?:<[a-zA-Z][a-zA-Z0-9-]*${HTML_ATTRIBUTE}*\\s*/?>|</[a-zA-Z][a-zA-Z0-9-]*\\s*>)\\s*$`,
);
const HTML_BLOCK_TAGS = new Set(
  `address article aside base basefont blockquote body caption center col colgroup
   dd details dialog dir div dl dt fieldset figcaption figure footer form frame
   frameset h1 h2 h3 h4 h5 h6 head header hr html iframe legend li link main menu
   menuitem nav noframes ol optgroup option p param search section summary table
   tbody td tfoot th thead title tr track ul`.split(/\s+/),
);
const NON_SPACE = /[^ \t]/;
const HEADING_LINE = /^ {0,3}#{1,6}(?:[ \t]|$)/;
const COMMENT_BLOCK_OPEN = /^ {0,3}<!--/;
const COMMENT_OPEN = "<!--";
const COMMENT_CLOSE = "-->";
// Underscores inside identifiers are literal, so UNSLOTH_DISABLE_UPDATE_CHECK keeps its name.
const BOLD_STAR = /\*\*(?=\S)([\s\S]*?\S)\*\*/g;
const BOLD_UNDERSCORE = /(^|[^\w])__(?=\S)([\s\S]*?\S)__(?=[^\w]|$)/g;
const ITALIC_STAR = /\*(?=\S)([^*\n]*?\S)\*/g;
const ITALIC_UNDERSCORE = /(^|[^\w])_(?=\S)([^_\n]*?\S)_(?=[^\w]|$)/g;
const BACKTICK = /`/g;
// Streamdown decodes entities like `AT&amp;T`, so the preview does too.
const NAMED_ENTITIES: Record<string, string> = {
  amp: "&",
  lt: "<",
  gt: ">",
  quot: '"',
  apos: "'",
  nbsp: "\u00a0",
};
const ENTITY = /&(#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[a-zA-Z][a-zA-Z0-9]{1,31});/g;
const PARKED = /\uE000(\d+)\uE001/g;
const WHITESPACE = /\s+/g;
const SENTENCE_BREAK = /[.!?]\s+(?=["'“‘]?[A-Z0-9])/g;
const TRAILING_WORD = /(\S+)$/;
const ABBREVIATIONS = new Set([
  "e.g.",
  "i.e.",
  "etc.",
  "vs.",
  "cf.",
  "approx.",
  "no.",
  "fig.",
  "al.",
  "dr.",
  "mr.",
  "mrs.",
  "ms.",
  "prof.",
  "inc.",
  "ltd.",
  "st.",
  "jr.",
  "sr.",
]);
const INITIAL = /^[A-Za-z]\.$/;
const MIN_LEAD_CHARS = 12;

/** Strip until stable, so a removal cannot re-form a tag. */
function stripHtmlTags(text: string): string {
  let out = text;
  let previous: string;
  do {
    previous = out;
    out = out.replace(HTML_TAG, "");
  } while (out !== previous);
  return out;
}

export interface ReleaseNotesPreviewItem {
  lead: string;
  rest: string;
}

export interface ReleaseNotesPreview {
  items: ReleaseNotesPreviewItem[];
  remaining: number;
}

interface Bullet {
  text: string;
  indent: number;
}

function definedLabel(
  labels: Set<string> | undefined,
  reference: string | undefined,
  text: string,
): boolean {
  if (labels === undefined) {
    return false;
  }
  const label = (reference?.trim() ? reference : text)
    .trim()
    .replace(WHITESPACE, " ")
    .toLowerCase();
  return labels.has(label);
}

function decodeEntity(match: string, body: string): string {
  if (body.startsWith("#")) {
    const hex = body[1] === "x" || body[1] === "X";
    const code = Number.parseInt(
      hex ? body.slice(2) : body.slice(1),
      hex ? 16 : 10,
    );
    return Number.isFinite(code) && code > 0 && code <= 0x10ffff
      ? String.fromCodePoint(code)
      : match;
  }
  return NAMED_ENTITIES[body.toLowerCase()] ?? match;
}

function toPlainText(markdown: string, labels?: Set<string>): string {
  const codes: string[] = [];
  const park = (text: string): string => {
    codes.push(text);
    return `\uE000${codes.length - 1}\uE001`;
  };
  const parked = parkCodeSpans(markdown, park).replace(ESCAPE, (_match, char) =>
    park(char),
  );

  return stripHtmlTags(
    parked
      .replace(AUTOLINK, "$1")
      .replace(IMAGE, "")
      .replace(LINK, "$1")
      .replace(IMAGE_REFERENCE, (match, text, ref) =>
        definedLabel(labels, ref, text) ? "" : match,
      )
      .replace(LINK_REFERENCE, (match, text, ref) =>
        definedLabel(labels, ref, text) ? text : match,
      ),
  )
    .replace(BOLD_STAR, "$1")
    .replace(BOLD_UNDERSCORE, "$1$2")
    .replace(ITALIC_STAR, "$1")
    .replace(ITALIC_UNDERSCORE, "$1$2")
    .replace(BACKTICK, "")
    .replace(ENTITY, decodeEntity)
    .replace(PARKED, (_match, index: string) => codes[Number(index)] ?? "")
    .replace(WHITESPACE, " ")
    .trim();
}

function truncate(text: string): string {
  if (text.length <= PREVIEW_ITEM_MAX_CHARS) {
    return text;
  }
  const clipped = text.slice(0, PREVIEW_ITEM_MAX_CHARS);
  const lastSpace = clipped.lastIndexOf(" ");
  return `${(lastSpace > 40 ? clipped.slice(0, lastSpace) : clipped).trimEnd()}...`;
}

interface ContentLine {
  text: string;
  indent: number;
  quoted: boolean;
  // CommonMark measures indentation from here, so `indent - column` is the real depth.
  column: number;
}

/**
 * Only a line-start comment hides whole lines; a mid-line one is inline HTML whose `-->` may
 * arrive later in the paragraph (`closesBelow`).
 */
function stripCommentSpans(
  line: string,
  startInComment: boolean,
  runOn: boolean,
  closesBelow: boolean,
  blockOpen: boolean,
): [string, boolean, boolean] {
  if (startInComment) {
    return ["", !line.includes(COMMENT_CLOSE), false];
  }

  let visible = "";
  let index = 0;
  if (runOn) {
    const closed = line.indexOf(COMMENT_CLOSE);
    if (closed === -1) {
      return ["", false, true];
    }
    index = closed + COMMENT_CLOSE.length;
  } else if (blockOpen) {
    // `<!-->` and `<!--->` are complete comments; searching past the opener would hide later releases.
    return ["", !line.includes(COMMENT_CLOSE), false];
  }

  const spans = codeSpans(line);
  while (index < line.length) {
    const open = line.indexOf(COMMENT_OPEN, index);
    if (open === -1) {
      visible += line.slice(index);
      break;
    }
    const span = spans.find(
      (candidate) => candidate.start <= open && candidate.end > open,
    );
    if (span) {
      visible += line.slice(index, span.end);
      index = span.end;
      continue;
    }
    const close = line.indexOf(COMMENT_CLOSE, open + COMMENT_OPEN.length);
    if (close === -1) {
      if (closesBelow) {
        return [visible + line.slice(index, open), false, true];
      }
      visible += line.slice(index);
      break;
    }
    visible += line.slice(index, open);
    index = close + COMMENT_CLOSE.length;
  }
  return [visible, false, false];
}

function stripRawHtml(
  line: string,
  openBlock: number | null,
): [string, number | null] {
  if (openBlock !== null) {
    return RAW_BLOCKS[openBlock]?.[1].test(line) ? ["", null] : ["", openBlock];
  }
  for (const [index, [opener, closer]] of RAW_BLOCKS.entries()) {
    const open = opener.exec(line);
    if (!open) {
      continue;
    }
    const rest = line.slice(open[0].length);
    return closer.test(rest) ? ["", null] : ["", index];
  }
  return [line, null];
}

function opensHtmlBlock(line: string, afterParagraph: boolean): boolean {
  const named = HTML_BLOCK_OPEN.exec(line);
  if (named && HTML_BLOCK_TAGS.has((named[1] ?? "").toLowerCase())) {
    return true;
  }
  return !afterParagraph && HTML_TAG_ONLY_LINE.test(line);
}

/** A hidden block still keeps its column (and marker), since its indent can close a list item. */
function structuralLine(
  line: string,
  visible: string,
  hidden: boolean,
  marker: string,
): string {
  if (visible.trim() || hidden) {
    return visible;
  }
  return hiddenStructure(line, marker);
}

interface ScanState {
  openFence: string | null;
  blockColumn: number;
  inComment: boolean;
  runOn: boolean;
  inRawHtml: number | null;
  inHtmlBlock: boolean;
  afterParagraph: boolean;
}

interface ScannedLine {
  // "" for structure and hidden blocks; null for fenced content so it cannot split a bullet.
  text: string | null;
  structural: string;
}

function visibleText(
  line: string,
  state: ScanState,
  closesBelow: boolean,
): ScannedLine {
  // Raw HTML first: its contents are literal, so a fence inside it is not one.
  if (state.inRawHtml !== null) {
    const [after, stillInRaw] = stripRawHtml(line, state.inRawHtml);
    state.inRawHtml = stillInRaw;
    return { text: after, structural: "" };
  }
  if (state.inHtmlBlock) {
    state.inHtmlBlock = line.trim() !== "";
    return { text: "", structural: "" };
  }
  const commented = state.inComment || state.runOn;
  const fence = commented
    ? null
    : FENCE.exec(
        state.openFence === null
          ? itemContent(line, state.afterParagraph)
          : line,
      );
  if (
    fence &&
    (state.openFence !== null || opensFence(fence[1] ?? "", fence[2] ?? ""))
  ) {
    state.openFence = nextFence(
      state.openFence,
      fence[1] ?? "",
      fence[2] ?? "",
    );
    return { text: "", structural: line };
  }
  if (state.openFence !== null) {
    return { text: null, structural: "" };
  }
  return visibleContent(line, state, closesBelow);
}

function visibleContent(
  line: string,
  state: ScanState,
  closesBelow: boolean,
): ScannedLine {
  const hidden = state.inComment || state.inRawHtml !== null;
  const carried = state.runOn;
  const content = itemContent(line, state.afterParagraph);
  const opensComment =
    !(state.inComment || carried) && COMMENT_BLOCK_OPEN.test(content);
  const [uncommented, stillInComment, stillRunOn] = stripCommentSpans(
    line,
    state.inComment,
    state.runOn,
    closesBelow,
    opensComment,
  );
  state.inComment = stillInComment;
  state.runOn = stillRunOn;
  const [visible, stillInRaw] = stripRawHtml(uncommented, state.inRawHtml);
  state.inRawHtml = stillInRaw;
  // Taken before the opener is hidden: its indent still closes a list item, and its marker opens one.
  const marker = opensComment
    ? line.slice(0, line.length - content.length)
    : "";
  const structural = carried
    ? line
    : structuralLine(line, visible, hidden, marker);
  if (
    !carried &&
    stillInRaw === null &&
    visible.trim() &&
    opensHtmlBlock(visible, state.afterParagraph)
  ) {
    state.inHtmlBlock = true;
    return { text: "", structural };
  }
  return { text: visible, structural };
}

/** Only within three columns of the item's content column; deeper is indented code. */
function opensDeepFence(line: ContentLine): string | null {
  if (
    line.indent < INDENTED_CODE_INDENT ||
    line.indent - line.column >= INDENTED_CODE_INDENT
  ) {
    return null;
  }
  const fence = FENCE.exec(line.text);
  return fence ? (fence[1] ?? null) : null;
}

/** A fence inside a list item ends with the item, as `fence_column` does on the backend. */
function endsDeepFence(
  marker: string,
  column: number,
  line: ContentLine,
): boolean {
  return line.indent < column || closesDeepFence(marker, line);
}

function closesDeepFence(marker: string, line: ContentLine): boolean {
  const fence = FENCE.exec(line.text);
  if (!fence) {
    return false;
  }
  const closer = fence[1] ?? "";
  return (
    closer[0] === marker[0] &&
    closer.length >= marker.length &&
    !NON_SPACE.test(fence[2] ?? "")
  );
}

/** Leading and trailing pipes are delimiters, and an escaped pipe is literal. */
function tableCells(text: string): string[] | null {
  if (!text.includes("|")) {
    return null;
  }
  const cells: string[] = [];
  let cell = "";
  for (let at = 0; at < text.length; at += 1) {
    const char = text[at];
    if (char === "\\") {
      cell += char + (text[at + 1] ?? "");
      at += 1;
      continue;
    }
    if (char === "|") {
      cells.push(cell);
      cell = "";
      continue;
    }
    cell += char;
  }
  cells.push(cell);
  if (cells.length > 1 && text.startsWith("|")) {
    cells.shift();
  }
  if (cells.length > 1 && text.endsWith("|")) {
    cells.pop();
  }
  return cells;
}

function delimiterWidth(text: string): number | null {
  const cells = tableCells(text);
  if (cells === null || cells.length === 0) {
    return null;
  }
  return cells.every((cell) => TABLE_DELIMITER_CELL.test(cell.trim()))
    ? cells.length
    : null;
}

/** Tables render as a grid, so the preview drops them like code blocks. */
function opensTable(
  header: ContentLine | undefined,
  delimiter: ContentLine | undefined,
): boolean {
  if (header === undefined || delimiter === undefined) {
    return false;
  }
  if (!header.text || header.quoted) {
    return false;
  }
  if (header.indent - header.column >= INDENTED_CODE_INDENT) {
    return false;
  }
  const width = delimiterWidth(delimiter.text);
  const cells = tableCells(header.text);
  return width !== null && cells !== null && cells.length === width;
}

function breaksTable(line: ContentLine | undefined): boolean {
  return (
    !line?.text ||
    line.quoted ||
    HEADING.test(line.text) ||
    BULLET.test(line.text) ||
    line.indent - line.column >= INDENTED_CODE_INDENT
  );
}

function tableLines(lines: ContentLine[]): Set<number> {
  const rows = new Set<number>();
  let at = 0;
  while (at + 1 < lines.length) {
    if (!opensTable(lines[at], lines[at + 1])) {
      at += 1;
      continue;
    }
    rows.add(at);
    rows.add(at + 1);
    let row = at + 2;
    while (row < lines.length && !breaksTable(lines[row])) {
      rows.add(row);
      row += 1;
    }
    at = row;
  }
  return rows;
}

/** A backtick fence's info string may not contain a backtick. */
function opensFence(marker: string, rest: string): boolean {
  return marker[0] !== "`" || !rest.includes("`");
}

function nextFence(
  open: string | null,
  marker: string,
  rest: string,
): string | null {
  if (open === null) {
    return opensFence(marker, rest) ? marker : null;
  }
  const closes =
    marker[0] === open[0] &&
    marker.length >= open.length &&
    !NON_SPACE.test(rest);
  return closes ? null : open;
}

function inBlock(state: ScanState): boolean {
  return (
    state.openFence !== null ||
    state.inRawHtml !== null ||
    state.inHtmlBlock ||
    state.inComment
  );
}

/** Lazy continuation reaches into no fence, comment or HTML block, so dedenting ends them. */
function closeDedentedBlock(line: string, state: ScanState): void {
  if (state.blockColumn === 0 || !inBlock(state)) {
    return;
  }
  if (line.trim() && indentWidth(line) < state.blockColumn) {
    state.openFence = null;
    state.inRawHtml = null;
    state.inHtmlBlock = false;
    state.inComment = false;
    state.blockColumn = 0;
  }
}

function scopeBlock(
  state: ScanState,
  wasInBlock: boolean,
  lists: ListState,
): void {
  if (!inBlock(state)) {
    state.blockColumn = 0;
    return;
  }
  if (!wasInBlock) {
    state.blockColumn = lists.columns.at(-1) ?? 0;
  }
}

function contentLines(markdown: string): ContentLine[] {
  const lines: ContentLine[] = [];
  const state: ScanState = {
    openFence: null,
    blockColumn: 0,
    inComment: false,
    runOn: false,
    inRawHtml: null,
    inHtmlBlock: false,
    afterParagraph: false,
  };
  let lists: ListState = EMPTY_LIST_STATE;
  let quote: QuoteState = NO_QUOTE;

  const rawLines = markdown
    .split("\n")
    .map((raw) => raw.replace(TABS, " ".repeat(TAB_WIDTH)));
  const closesBelow = commentClosesBelow(rawLines);
  for (const [index, line] of rawLines.entries()) {
    closeDedentedBlock(line, state);
    const wasInBlock = inBlock(state);
    const carried = state.runOn;
    const { text: visible, structural } = visibleText(
      line,
      state,
      closesBelow[index + 1] ?? false,
    );
    const above = quote;
    quote = NO_QUOTE;
    lists = openLists(structural, lists, state.afterParagraph, above.quoted);
    scopeBlock(state, wasInBlock, lists);
    if (visible === null) {
      continue;
    }
    if (carried && !visible.trim()) {
      continue;
    }
    if (!visible.trim() || THEMATIC_BREAK.test(visible)) {
      // A rule separates notes, so it breaks a bullet like a blank line.
      state.afterParagraph = false;
      lines.push({ text: "", indent: 0, quoted: false, column: 0 });
      continue;
    }
    const quoted = BLOCKQUOTE.test(visible);
    const stripped = visible.replace(BLOCKQUOTE, "");
    const indent = stripped.length - stripped.trimStart().length;
    const column = quoted ? 0 : (lists.columns.at(-1) ?? 0);
    const startsCode =
      !state.afterParagraph && indent - column >= INDENTED_CODE_INDENT;
    state.afterParagraph = !HEADING_LINE.test(stripped) && !startsCode;
    quote = quoteState(visible, above.inQuote);
    lines.push({ text: stripped.trim(), indent, quoted, column });
  }
  return lines;
}

/** Conservative: the next sentence must start like one, so "unsloth.ai in the docs" is not split. */
function splitLeadSentence(text: string): ReleaseNotesPreviewItem {
  SENTENCE_BREAK.lastIndex = 0;
  let match = SENTENCE_BREAK.exec(text);
  while (match) {
    const cut = match.index + 1;
    const word =
      TRAILING_WORD.exec(text.slice(0, cut))?.[1]?.toLowerCase() ?? "";
    const isAbbreviation = ABBREVIATIONS.has(word) || INITIAL.test(word);
    if (!isAbbreviation && cut >= MIN_LEAD_CHARS) {
      return { lead: text.slice(0, cut).trim(), rest: text.slice(cut).trim() };
    }
    match = SENTENCE_BREAK.exec(text);
  }
  return { lead: text, rest: "" };
}

interface Collector {
  bullets: Bullet[];
  prose: string[];
  current: Bullet | null;
  paragraph: string;
  // A quote owns its paragraph: a marker outside the quote opens a list rather than continuing it.
  quotedParagraph: boolean;
}

function flush(collector: Collector): void {
  if (collector.current?.text) {
    collector.bullets.push({
      text: truncate(collector.current.text),
      indent: collector.current.indent,
    });
  }
  collector.current = null;
  if (collector.paragraph) {
    collector.prose.push(truncate(collector.paragraph));
    collector.paragraph = "";
  }
  collector.quotedParagraph = false;
}

function takeBullet(
  collector: Collector,
  text: string,
  line: ContentLine,
  labels: Set<string>,
): void {
  flush(collector);
  const item = toPlainText(text, labels);
  // A quoted list is example output, never a headline bullet.
  if (!line.quoted) {
    collector.current = { text: item, indent: line.indent };
  } else if (item) {
    collector.prose.push(truncate(item));
  }
}

function takeText(
  collector: Collector,
  text: string,
  labels: Set<string>,
  quoted: boolean,
): void {
  const plain = toPlainText(text, labels);
  if (!plain) {
    return;
  }
  if (collector.current === null) {
    collector.paragraph = collector.paragraph
      ? `${collector.paragraph} ${plain}`
      : plain;
    collector.quotedParagraph = quoted;
    return;
  }
  collector.current = {
    text: `${collector.current.text} ${plain}`,
    indent: collector.current.indent,
  };
}

function collectBullets(markdown: string): {
  bullets: Bullet[];
  prose: string[];
} {
  const collector: Collector = {
    bullets: [],
    prose: [],
    current: null,
    paragraph: "",
    quotedParagraph: false,
  };

  const lines = contentLines(markdown);
  const labels = new Set<string>();
  // A definition-shaped line inside code is literal, and a real one never indents past three spaces.
  let labelFence: string | null = null;
  let labelColumn = 0;
  for (const line of lines) {
    if (labelFence !== null && !endsDeepFence(labelFence, labelColumn, line)) {
      continue;
    }
    if (labelFence !== null) {
      const dedented = line.indent < labelColumn;
      labelFence = null;
      if (!dedented) {
        continue;
      }
    }
    const opener = opensDeepFence(line);
    if (opener !== null) {
      labelFence = opener;
      labelColumn = line.column;
      continue;
    }
    if (line.indent - line.column >= INDENTED_CODE_INDENT) {
      continue;
    }
    const definition = DEFINITION.exec(line.text);
    if (definition) {
      labels.add(
        (definition[1] ?? "").trim().replace(WHITESPACE, " ").toLowerCase(),
      );
    }
  }

  const tables = tableLines(lines);
  let deepFence: string | null = null;
  let deepColumn = 0;
  for (const [index, line] of lines.entries()) {
    if (!line.text || HEADING.test(line.text)) {
      flush(collector);
      continue;
    }
    if (tables.has(index)) {
      flush(collector);
      continue;
    }
    if (collector.current === null && DEFINITION.test(line.text)) {
      continue;
    }
    // A fence indented past three spaces belongs to a list item, so the line scanner missed it.
    if (deepFence !== null && !endsDeepFence(deepFence, deepColumn, line)) {
      continue;
    }
    if (deepFence !== null) {
      const dedented = line.indent < deepColumn;
      deepFence = null;
      if (!dedented) {
        continue;
      }
    }
    const opener = opensDeepFence(line);
    if (opener !== null) {
      deepFence = opener;
      deepColumn = line.column;
      continue;
    }
    const insideBlock =
      collector.current !== null || collector.paragraph !== "";
    if (!insideBlock && line.indent - line.column >= INDENTED_CODE_INDENT) {
      continue;
    }
    const bullet = BULLET.exec(line.text);
    // Only an ordered list starting at 1 may interrupt a paragraph.
    const interrupts =
      collector.current === null &&
      collector.paragraph !== "" &&
      !collector.quotedParagraph;
    if (
      bullet &&
      !(interrupts && bullet[1] !== undefined && bullet[1] !== "1")
    ) {
      takeBullet(collector, bullet[2] ?? "", line, labels);
      continue;
    }
    takeText(collector, line.text, labels, line.quoted);
  }
  flush(collector);

  return { bullets: collector.bullets, prose: collector.prose };
}

export function releaseNotesPreview(
  markdown: string | null | undefined,
  limit: number = RELEASE_NOTES_PREVIEW_ITEMS,
): ReleaseNotesPreview {
  if (!markdown) {
    return { items: [], remaining: 0 };
  }

  // Updater bodies arrive with CRLF; sentinels would collide with parking.
  const text = markdown.replace(LINE_ENDINGS, "\n").replace(SENTINELS, "");
  const { bullets, prose } = collectBullets(text);
  // Shallowest bullet defines top level, so a uniformly indented list still previews.
  const baseIndent = bullets.reduce(
    (min, bullet) => Math.min(min, bullet.indent),
    Number.POSITIVE_INFINITY,
  );
  const topLevel = bullets
    .filter((bullet) => bullet.indent <= baseIndent + NESTED_INDENT_TOLERANCE)
    .map((bullet) => bullet.text);

  const source = topLevel.length > 0 ? topLevel : prose;
  return {
    items: source.slice(0, limit).map(splitLeadSentence),
    remaining: Math.max(source.length - limit, 0),
  };
}
