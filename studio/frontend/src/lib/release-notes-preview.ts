


// Top changelog bullets, shown in the collapsed update popup.
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
// Bullets indented past the shallowest one are nested detail, not headlines.
const NESTED_INDENT_TOLERANCE = 1;
const TAB_WIDTH = 4;
// Four spaces starts an indented code block in Markdown.
const INDENTED_CODE_INDENT = 4;

// At most three leading spaces: deeper is indented code, not a fence.
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
// An ATX heading needs a space, tab or line end after the marker, as in
// _HEADING_PATTERN. `\s` would match a non-breaking space and eat prose, and a
// bare `##` is an empty heading that still ends a bullet.
const HEADING = /^#{1,6}(?:[ \t]|$)/;
const BULLET = /^(?:[-*+]|(\d{1,9})[.)])[ \t]+(.*)$/;
// At most three leading spaces, as everywhere else: deeper is indented code,
// so a quoted line inside a code sample cannot reach the collector.
const BLOCKQUOTE = /^ {0,3}>[ \t]?/;
// A GFM delimiter cell is hyphens with an optional alignment colon each side.
const TABLE_DELIMITER_CELL = /^:?-+:?$/;
// "- - -" and "***" are horizontal rules, not bullets and not notes.
const THEMATIC_BREAK =
  /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
// Destinations may escape or balance parentheses, and labels may nest one
// level so `[![alt](img)](link)` still resolves.
const DESTINATION = "\\((?:\\\\.|[^()\\\\]|\\([^()]*\\))*\\)";
const LABEL = "((?:[^\\[\\]\\\\]|\\\\.|\\[(?:[^\\[\\]\\\\]|\\\\.)*\\])*)";
const IMAGE = new RegExp(`!\\[${LABEL}\\]${DESTINATION}`, "g");
const LINK = new RegExp(`\\[${LABEL}\\]${DESTINATION}`, "g");
// Reference forms: `[text][label]`, `[text][]` and the shortcut `[text]`.
const IMAGE_REFERENCE = new RegExp(`!\\[${LABEL}\\](?:\\[([^\\]]*)\\])?`, "g");
const LINK_REFERENCE = new RegExp(`\\[${LABEL}\\](?:\\[([^\\]]*)\\])?`, "g");
// A definition line renders as nothing at all.
const DEFINITION = /^ {0,3}\[((?:[^\[\]\\]|\\.)+)\]:/;
// CommonMark: a backslash escapes ASCII punctuation.
const ESCAPE = /\\([!-/:-@[-`{-~])/g;
// Private-use sentinels park code spans, so document text cannot contain them.
const SENTINELS = /[\uE000\uE001]/g;
const LINE_ENDINGS = /\r\n?/g;
const TABS = /\t/g;
// Real tags only: a name character must follow "<", so a version constraint
// like "Support Python <3.15 and >3.9" keeps its operators.
const HTML_TAG = /<\/?[a-zA-Z][^>]*>/g;
// <https://x> and <a@b.c> are Markdown autolinks: keep the text they render.
const AUTOLINK = /<([a-zA-Z][a-zA-Z0-9+.-]*:[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>/g;
// CommonMark type 1 HTML blocks render literally until a closing tag, which
// the spec says need not be the one that opened the block.
const RAW_HTML_OPEN = /^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)/i;
const RAW_HTML_CLOSE = /<\/(pre|script|style|textarea)\s*>/i;
// Types 3 to 5 (processing instructions, declarations, CDATA) are literal too,
// each ending on its own delimiter. Comments open mid-line, handled separately.
const RAW_BLOCKS: [RegExp, RegExp][] = [
  [RAW_HTML_OPEN, RAW_HTML_CLOSE],
  [/^ {0,3}<\?/, /\?>/],
  [/^ {0,3}<!\[CDATA\[/, /\]\]>/],
  // A declaration needs an uppercase letter, so `<!note` stays ordinary text.
  [/^ {0,3}<![A-Z]/, />/],
];
// Type 6 and 7 blocks run to the next blank line, so `<details>` holds Markdown
// only after one. Type 7 (any complete tag alone) cannot interrupt a paragraph.
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
// Only spaces and tabs may follow a closing fence.
const NON_SPACE = /[^ \t]/;
const HEADING_LINE = /^ {0,3}#{1,6}(?:[ \t]|$)/;
const COMMENT_BLOCK_OPEN = /^ {0,3}<!--/;
const COMMENT_OPEN = "<!--";
const COMMENT_CLOSE = "-->";
// Paired emphasis only. Underscores inside identifiers are literal, so
// UNSLOTH_DISABLE_UPDATE_CHECK keeps its name.
const BOLD_STAR = /\*\*(?=\S)([\s\S]*?\S)\*\*/g;
const BOLD_UNDERSCORE = /(^|[^\w])__(?=\S)([\s\S]*?\S)__(?=[^\w]|$)/g;
const ITALIC_STAR = /\*(?=\S)([^*\n]*?\S)\*/g;
const ITALIC_UNDERSCORE = /(^|[^\w])_(?=\S)([^_\n]*?\S)_(?=[^\w]|$)/g;
const BACKTICK = /`/g;
// A closer is a run of the same length, so `` `x` `` keeps its backticks.
// Streamdown renders `AT&amp;T` as "AT&T", so the preview decodes entities too.
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
// Sentence end followed by something that actually starts a sentence.
const SENTENCE_BREAK = /[.!?]\s+(?=["'“‘]?[A-Z0-9])/g;
const TRAILING_WORD = /(\S+)$/;
// A period here ends an abbreviation, not the sentence.
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

/** Strip tags until stable, so a removal cannot re-form a tag. */
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
  // Leading sentence, highlighted in the preview.
  lead: string;
  // Rest of the bullet, de-emphasised. Empty for single-sentence bullets.
  rest: string;
}

export interface ReleaseNotesPreview {
  items: ReleaseNotesPreviewItem[];
  // Bullets past the preview limit, for a "+N more" affordance.
  remaining: number;
}

interface Bullet {
  text: string;
  indent: number;
}

/** Whether a reference points at a definition the document actually has. */
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

/** One entity as the character it renders as, or unchanged if unknown. */
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

/** Inline markdown stripped to plain text. */
function toPlainText(markdown: string, labels?: Set<string>): string {
  // Park code spans first: their contents are literal and must survive below.
  const codes: string[] = [];
  const park = (text: string): string => {
    codes.push(text);
    return `\uE000${codes.length - 1}\uE001`;
  };
  // Escaped punctuation is literal too, so `\*not italic\*` keeps its stars.
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
  // Blockquoted lines are quoted examples, not the release's own bullets.
  quoted: boolean;
  // Content column of the innermost open list item. CommonMark measures
  // indentation from here, so `indent - column` is the real depth.
  column: number;
}

/**
 * `line` with its comments removed, whether a comment block stays open, and
 * whether an inline comment runs on into the line below.
 *
 * Only a comment starting a line opens a block, which hides whole lines to the
 * one holding `-->`. One written mid-sentence is inline HTML belonging to its
 * paragraph, so its `-->` may arrive on a later line and only the text up to it
 * is hidden. `closesBelow` says one does; without it the opener is ordinary text
 * and hides nothing below.
 *
 * "Starting a line" is read inside the container, so `blockOpen` comes from the
 * item's content rather than the raw line.
 */
function stripCommentSpans(
  line: string,
  startInComment: boolean,
  runOn: boolean,
  closesBelow: boolean,
  blockOpen: boolean,
): [string, boolean, boolean] {
  if (startInComment) {
    // The closing line belongs to the block, tail included.
    return ["", !line.includes(COMMENT_CLOSE), false];
  }

  let visible = "";
  let index = 0;
  if (runOn) {
    const closed = line.indexOf(COMMENT_CLOSE);
    if (closed === -1) {
      return ["", false, true];
    }
    // Only up to the closer: the tail is the paragraph's own text again.
    index = closed + COMMENT_CLOSE.length;
  } else if (blockOpen) {
    // `<!-->` and `<!--->` are complete comments, so the closer may overlap the
    // opener; searching past it would hide every later release.
    return ["", !line.includes(COMMENT_CLOSE), false];
  }

  const spans = codeSpans(line);
  while (index < line.length) {
    const open = line.indexOf(COMMENT_OPEN, index);
    if (open === -1) {
      visible += line.slice(index);
      break;
    }
    // A delimiter inside inline code is literal, not a comment opener.
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
        // The paragraph carries the comment on, so this line and the next are in it.
        return [visible + line.slice(index, open), false, true];
      }
      // Nothing closes it at all, so the renderer shows it as text.
      visible += line.slice(index);
      break;
    }
    visible += line.slice(index, open);
    index = close + COMMENT_CLOSE.length;
  }
  return [visible, false, false];
}

/** Strips raw block content. State is the open block's index, or null. */
function stripRawHtml(
  line: string,
  openBlock: number | null,
): [string, number | null] {
  if (openBlock !== null) {
    return RAW_BLOCKS[openBlock]?.[1].test(line) ? ["", null] : ["", openBlock];
  }
  // A block only opens at the start of a line; mid-line tags are inline HTML.
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

/** True if `line` starts a CommonMark type 6 or type 7 HTML block. */
function opensHtmlBlock(line: string, afterParagraph: boolean): boolean {
  const named = HTML_BLOCK_OPEN.exec(line);
  if (named && HTML_BLOCK_TAGS.has((named[1] ?? "").toLowerCase())) {
    return true;
  }
  return !afterParagraph && HTML_TAG_ONLY_LINE.test(line);
}

/**
 * The line as list tracking sees it. A comment or raw block renders nothing, but
 * the line opening one is still a block at its own column, so it closes a list
 * item it sits left of. Only the column survives, since the text it hides is not
 * Markdown. A line inside a block already open is that block's content, so it
 * keeps neither. A marker the hidden block is the content of survives with the
 * column, so the item it opens is still tracked.
 */
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
  // Content column of the list item the open block belongs to, 0 at document
  // level. A fence and an HTML block are scoped to their container, so the item's
  // end closes them. Only one of the three is ever open.
  blockColumn: number;
  inComment: boolean;
  // True while an inline comment opened above runs on into this line, carried by
  // the paragraph holding it.
  runOn: boolean;
  inRawHtml: number | null;
  inHtmlBlock: boolean;
  afterParagraph: boolean;
}

interface ScannedLine {
  // What a reader would see: "" for structure and hidden blocks, null for
  // fenced content, which is skipped so it cannot split a bullet.
  text: string | null;
  // The same line as list tracking sees it: blank wherever nothing renders,
  // but kept whole where an indent still closes an open item.
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
    // A blank line is the only thing that ends a type 6 or 7 block.
    state.inHtmlBlock = line.trim() !== "";
    return { text: "", structural: "" };
  }
  // An opener is read past a marker on the same line, since a fence written as a
  // list item's first content opens inside it. Only an opener: fenced content is
  // literal and a closer carries no marker.
  const commented = state.inComment || state.runOn;
  const fence = commented
    ? null
    : FENCE.exec(
        state.openFence === null
          ? itemContent(line, state.afterParagraph)
          : line,
      );
  // A backtick fence whose info string holds a backtick is prose, not a fence.
  if (
    fence &&
    (state.openFence !== null || opensFence(fence[1] ?? "", fence[2] ?? ""))
  ) {
    state.openFence = nextFence(
      state.openFence,
      fence[1] ?? "",
      fence[2] ?? "",
    );
    // Hidden from the collector, but its indent still closes an item.
    return { text: "", structural: line };
  }
  if (state.openFence !== null) {
    return { text: null, structural: "" };
  }
  return visibleContent(line, state, closesBelow);
}

/** `visibleText` for a line no fence or HTML block already owns. */
function visibleContent(
  line: string,
  state: ScanState,
  closesBelow: boolean,
): ScannedLine {
  // A block already open owns this line, so it is content rather than a block
  // written at the column it happens to start in.
  const hidden = state.inComment || state.inRawHtml !== null;
  const carried = state.runOn;
  // A comment is an HTML block too, so one written as a list item's first content
  // opens inside that item exactly as a fence does: read past a marker on the
  // same line rather than from the margin.
  const content = itemContent(line, state.afterParagraph);
  const opensComment =
    !(state.inComment || carried) && COMMENT_BLOCK_OPEN.test(content);
  // Commented-out notes are not rendered, so they are not previewed either.
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
  // Taken before the opener is hidden: it renders as nothing, but its indent still
  // closes a list item it sits left of, and a marker on its line still opens one.
  // A line an inline comment runs on into is still a line of the paragraph that
  // carries it, so only its text is hidden, never its block structure.
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

/**
 * Marker of a fence the line scanner skipped because it is indented. Only a line
 * within three columns of its item's content column is one: deeper than that it
 * is an indented code block, which a dedented bullet ends.
 */
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

/**
 * True when `line` is the first one outside the deep fence opened with `marker`
 * at `column`. A fence inside a list item runs only to the end of that item, so a
 * line left of the item's content column closes both, as `fence_column` does on
 * the backend.
 */
function endsDeepFence(
  marker: string,
  column: number,
  line: ContentLine,
): boolean {
  return line.indent < column || closesDeepFence(marker, line);
}

/** True when `line` closes the deep fence opened with `marker`. */
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

/**
 * Cells of a GFM table row, or null when the line holds no pipe at all. The
 * optional leading and trailing pipes are delimiters, not empty cells, and a
 * `\|` is literal text inside one.
 */
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

/** Width of a GFM delimiter row such as `| --- |:-:|`, or null if not one. */
function delimiterWidth(text: string): number | null {
  const cells = tableCells(text);
  if (cells === null || cells.length === 0) {
    return null;
  }
  return cells.every((cell) => TABLE_DELIMITER_CELL.test(cell.trim()))
    ? cells.length
    : null;
}

/**
 * Line indices that belong to a GFM table. A table needs a header row and a
 * delimiter row of the same width, and runs to a blank line or another block. Its
 * cells render as a grid, not prose, so the preview drops them like a code block.
 */
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

/** A blank line, a heading or a list marker: where GFM breaks a table. */
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
    // Only spaces or tabs may follow a closer, per CommonMark.
    !NON_SPACE.test(rest);
  return closes ? null : open;
}

/** Whether a fence, a raw block, a comment or an HTML block is open. */
function inBlock(state: ScanState): boolean {
  return (
    state.openFence !== null ||
    state.inRawHtml !== null ||
    state.inHtmlBlock ||
    state.inComment
  );
}

/**
 * A fence, comment or HTML block inside a list item runs only to the end of that
 * item, so a line dedented out of the item closes both. Lazy continuation reaches
 * into none of them, so any content left of the item ends it.
 */
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

/** Ties a block just opened to the list item it is written inside. */
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
    // The opener closed the items it is dedented out of first, so this is the
    // column of the item the block really sits in.
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
    // The quote state from the line above, which is what list tracking asks about.
    // Only a line of text below rewrites it, so a fenced, blank or hidden line
    // leaves no quoted paragraph open behind it.
    const above = quote;
    quote = NO_QUOTE;
    // Taken with the paragraph state from the line above, as a renderer would.
    lists = openLists(structural, lists, state.afterParagraph, above.quoted);
    scopeBlock(state, wasInBlock, lists);
    if (visible === null) {
      continue;
    }
    if (carried && !visible.trim()) {
      // Wholly inside a comment its paragraph carries: no text, and no break.
      continue;
    }
    if (!visible.trim() || THEMATIC_BREAK.test(visible)) {
      // A rule separates notes, so it breaks a bullet just like a blank line.
      state.afterParagraph = false;
      lines.push({ text: "", indent: 0, quoted: false, column: 0 });
      continue;
    }
    const quoted = BLOCKQUOTE.test(visible);
    const stripped = visible.replace(BLOCKQUOTE, "");
    const indent = stripped.length - stripped.trimStart().length;
    // A quoted line is measured inside its quote, where the document's open
    // list items do not reach.
    const column = quoted ? 0 : (lists.columns.at(-1) ?? 0);
    // Only ordinary text continues a paragraph; a heading or indented code line
    // (four columns past its container, outside a paragraph) ends one.
    const startsCode =
      !state.afterParagraph && indent - column >= INDENTED_CODE_INDENT;
    state.afterParagraph = !HEADING_LINE.test(stripped) && !startsCode;
    quote = quoteState(visible, above.inQuote);
    lines.push({ text: stripped.trim(), indent, quoted, column });
  }
  return lines;
}

/**
 * Split a bullet at its first sentence boundary. Conservative: the next
 * sentence must start like one, so "CHANGELOG.md in the repo" is not a break.
 */
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

/** Bullets in document order, plus prose for changelogs written as paragraphs. */
interface Collector {
  bullets: Bullet[];
  prose: string[];
  // Wrapped bullets continue on following lines and belong to one item.
  current: Bullet | null;
  paragraph: string;
  // True while the open paragraph is a quote's, which owns its own text: a
  // marker written outside the quote opens a list rather than continuing it.
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
  // A quoted list is example output: prose at best, never a headline bullet.
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
    // Wrapped paragraphs render as one block, so preview them as one item.
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
  // Skips the same code the pass below skips: a definition-shaped line inside
  // code is literal, and a real definition never indents past three spaces.
  let labelFence: string | null = null;
  let labelColumn = 0;
  for (const line of lines) {
    if (labelFence !== null && !endsDeepFence(labelFence, labelColumn, line)) {
      continue;
    }
    if (labelFence !== null) {
      const dedented = line.indent < labelColumn;
      labelFence = null;
      // Its own closing line is code too; only a dedented one is a new block.
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
    // A table renders as a grid, no more previewable than a code block, and it
    // ends whatever came before it.
    if (tables.has(index)) {
      flush(collector);
      continue;
    }
    // A link reference definition renders as nothing.
    if (collector.current === null && DEFINITION.test(line.text)) {
      continue;
    }
    // A fence indented past three spaces belongs to a list item, so the line
    // scanner missed it. Its contents are code either way.
    if (deepFence !== null && !endsDeepFence(deepFence, deepColumn, line)) {
      continue;
    }
    if (deepFence !== null) {
      const dedented = line.indent < deepColumn;
      deepFence = null;
      // Its own closing line is code too; only a dedented one is a new block.
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
    // An indented code block renders as code, so a "- cmd" line in one is not
    // a bullet. Inside an open bullet or paragraph it is just a wrapped line.
    const insideBlock =
      collector.current !== null || collector.paragraph !== "";
    if (!insideBlock && line.indent - line.column >= INDENTED_CODE_INDENT) {
      continue;
    }
    const bullet = BULLET.exec(line.text);
    // Only an ordered list starting at 1 may interrupt a paragraph, so "2. Restart
    // Studio" under prose is prose. A list item is not a paragraph.
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

/**
 * Top-level bullets of a release section, in document order. Nested bullets are
 * detail and are skipped; prose is used when a release has no bullets.
 */
export function releaseNotesPreview(
  markdown: string | null | undefined,
  limit: number = RELEASE_NOTES_PREVIEW_ITEMS,
): ReleaseNotesPreview {
  if (!markdown) {
    return { items: [], remaining: 0 };
  }

  // The updater body arrives with CRLF; sentinels would collide with parking.
  const text = markdown.replace(LINE_ENDINGS, "\n").replace(SENTINELS, "");
  const { bullets, prose } = collectBullets(text);
  // Shallowest bullet defines top level, so a uniformly indented list previews.
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
