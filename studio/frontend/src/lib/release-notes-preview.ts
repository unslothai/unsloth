// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Top changelog bullets, shown in the collapsed update popup.
export const RELEASE_NOTES_PREVIEW_ITEMS = 4;
const PREVIEW_ITEM_MAX_CHARS = 120;
// Bullets indented past the shallowest one are nested detail, not headlines.
const NESTED_INDENT_TOLERANCE = 1;
const TAB_WIDTH = 4;
// Four spaces starts an indented code block in Markdown.
const INDENTED_CODE_INDENT = 4;

// At most three leading spaces: deeper is indented code, not a fence.
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
const HEADING = /^#{1,6}\s+/;
const BULLET = /^(?:[-*+]|\d+[.)])\s+(.*)$/;
const BLOCKQUOTE = /^\s*>\s?/;
// "- - -" and "***" are horizontal rules, not bullets and not notes.
const THEMATIC_BREAK =
  /^ {0,3}(?:(?:\*[ \t]*){3,}|(?:-[ \t]*){3,}|(?:_[ \t]*){3,})$/;
const IMAGE = /!\[[^\]]*\]\([^)]*\)/g;
const LINK = /\[([^\]]*)\]\([^)]*\)/g;
// Real tags only: a name character must follow "<", so a version constraint
// like "Support Python <3.15 and >3.9" keeps its operators.
const HTML_TAG = /<\/?[a-zA-Z][^>]*>/g;
// <https://x> and <a@b.c> are Markdown autolinks: keep the text they render.
const AUTOLINK = /<([a-zA-Z][a-zA-Z0-9+.-]*:[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>/g;
const CODE_SPAN_INLINE = /(`+)[\s\S]*?\1(?!`)/g;
// CommonMark type 1 HTML blocks render literally until a closing tag, which
// the spec says need not be the one that opened the block.
const RAW_HTML_OPEN = /^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)/i;
const RAW_HTML_CLOSE = /<\/(pre|script|style|textarea)\s*>/i;
// Type 6 and 7 blocks run to the next blank line, so `<details>` holds Markdown
// only once a blank line has closed the block. Type 7 (any other complete tag
// alone on a line) cannot interrupt a paragraph.
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
const HEADING_LINE = /^ {0,3}#{1,6}(?:\s|$)/;
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
const CODE_SPAN = /(`+)([\s\S]*?)\1(?!`)/g;
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

/**
 * Strip tags until stable. The result is rendered as text, never as HTML, so
 * this is defence in depth against a removal re-forming a tag.
 */
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

/** One space of padding is dropped when a code span has it on both sides. */
function stripCodeSpanPadding(code: string): string {
  const padded =
    code.startsWith(" ") && code.endsWith(" ") && code.trim() !== "";
  return padded ? code.slice(1, -1) : code;
}

/** Inline markdown stripped to plain text. */
function toPlainText(markdown: string): string {
  // Park code spans first: their contents are literal, so tags, links and
  // emphasis inside them must survive every transformation below.
  const codes: string[] = [];
  const parked = markdown.replace(CODE_SPAN, (_match, _ticks, code: string) => {
    codes.push(stripCodeSpanPadding(code));
    return `\uE000${codes.length - 1}\uE001`;
  });

  return stripHtmlTags(
    parked.replace(AUTOLINK, "$1").replace(IMAGE, "").replace(LINK, "$1"),
  )
    .replace(BOLD_STAR, "$1")
    .replace(BOLD_UNDERSCORE, "$1$2")
    .replace(ITALIC_STAR, "$1")
    .replace(ITALIC_UNDERSCORE, "$1$2")
    .replace(BACKTICK, "")
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
}

/**
 * Lines outside fenced code blocks, with list indentation preserved. Only a
 * same-character run at least as long closes a fence, so a ``` sample inside a
 * ```` block does not end it early.
 */
function stripCommentSpans(
  line: string,
  startInComment: boolean,
): [string, boolean] {
  let visible = "";
  let index = 0;
  let inComment = startInComment;
  while (index < line.length) {
    if (inComment) {
      const close = line.indexOf(COMMENT_CLOSE, index);
      if (close === -1) {
        return [visible, true];
      }
      index = close + COMMENT_CLOSE.length;
      inComment = false;
      continue;
    }
    const open = line.indexOf(COMMENT_OPEN, index);
    if (open === -1) {
      visible += line.slice(index);
      break;
    }
    // A delimiter inside inline code is literal, not a comment opener.
    CODE_SPAN_INLINE.lastIndex = index;
    const span = CODE_SPAN_INLINE.exec(line);
    if (span && span.index <= open && span.index + span[0].length > open) {
      const end = span.index + span[0].length;
      visible += line.slice(index, end);
      index = end;
      continue;
    }
    visible += line.slice(index, open);
    index = open + COMMENT_OPEN.length;
    inComment = true;
  }
  return [visible, inComment];
}

function stripRawHtml(line: string, startInRaw: boolean): [string, boolean] {
  if (startInRaw) {
    const close = RAW_HTML_CLOSE.exec(line);
    return close
      ? [line.slice(close.index + close[0].length), false]
      : ["", true];
  }
  // A block only opens at the start of a line; mid-line tags are inline HTML.
  const open = RAW_HTML_OPEN.exec(line);
  if (!open) {
    return [line, false];
  }
  const rest = line.slice(open[0].length);
  const close = RAW_HTML_CLOSE.exec(rest);
  return close
    ? [rest.slice(close.index + close[0].length), false]
    : ["", true];
}

/** True if `line` starts a CommonMark type 6 or type 7 HTML block. */
function opensHtmlBlock(line: string, afterParagraph: boolean): boolean {
  const named = HTML_BLOCK_OPEN.exec(line);
  if (named && HTML_BLOCK_TAGS.has((named[1] ?? "").toLowerCase())) {
    return true;
  }
  return !afterParagraph && HTML_TAG_ONLY_LINE.test(line);
}

interface ScanState {
  openFence: string | null;
  inComment: boolean;
  inRawHtml: boolean;
  inHtmlBlock: boolean;
  afterParagraph: boolean;
}

/**
 * Text of `line` a reader would see: "" for structure and hidden blocks, null
 * for fenced content, which is skipped so it cannot split a bullet.
 */
function visibleText(line: string, state: ScanState): string | null {
  // Raw HTML first: its contents are literal, so a fence inside it is not a
  // fence. Only the text after the closing tag is a note.
  if (state.inRawHtml) {
    const [after, stillInRaw] = stripRawHtml(line, true);
    state.inRawHtml = stillInRaw;
    return after;
  }
  if (state.inHtmlBlock) {
    // A blank line is the only thing that ends a type 6 or 7 block.
    state.inHtmlBlock = line.trim() !== "";
    return "";
  }
  const fence = state.inComment ? null : FENCE.exec(line);
  if (fence) {
    state.openFence = nextFence(
      state.openFence,
      fence[1] ?? "",
      fence[2] ?? "",
    );
    return "";
  }
  if (state.openFence !== null) {
    return null;
  }
  // Commented-out notes are not rendered, so they are not previewed either.
  const [uncommented, stillInComment] = stripCommentSpans(
    line,
    state.inComment,
  );
  state.inComment = stillInComment;
  const [visible, stillInRaw] = stripRawHtml(uncommented, state.inRawHtml);
  state.inRawHtml = stillInRaw;
  if (
    !stillInRaw &&
    visible.trim() &&
    opensHtmlBlock(visible, state.afterParagraph)
  ) {
    state.inHtmlBlock = true;
    return "";
  }
  return visible;
}

/** Fence state after a ``` or ~~~ line. A closer carries nothing after it. */
function nextFence(
  open: string | null,
  marker: string,
  rest: string,
): string | null {
  if (open === null) {
    return marker;
  }
  const closes =
    marker[0] === open[0] && marker.length >= open.length && rest.trim() === "";
  return closes ? null : open;
}

function contentLines(markdown: string): ContentLine[] {
  const lines: ContentLine[] = [];
  const state: ScanState = {
    openFence: null,
    inComment: false,
    inRawHtml: false,
    inHtmlBlock: false,
    afterParagraph: false,
  };

  for (const rawLine of markdown.split("\n")) {
    const visible = visibleText(
      rawLine.replace(/\t/g, " ".repeat(TAB_WIDTH)),
      state,
    );
    if (visible === null) {
      continue;
    }
    if (!visible.trim() || THEMATIC_BREAK.test(visible)) {
      // A rule separates notes, so it breaks a bullet just like a blank line.
      state.afterParagraph = false;
      lines.push({ text: "", indent: 0, quoted: false });
      continue;
    }
    const quoted = BLOCKQUOTE.test(visible);
    const stripped = visible.replace(BLOCKQUOTE, "");
    const indent = stripped.length - stripped.trimStart().length;
    // Only ordinary text continues a paragraph. A heading, a block or an
    // indented code line (four spaces, outside a paragraph) ends one.
    const startsCode = !state.afterParagraph && indent >= INDENTED_CODE_INDENT;
    state.afterParagraph = !HEADING_LINE.test(stripped) && !startsCode;
    lines.push({ text: stripped.trim(), indent, quoted });
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
}

function takeBullet(
  collector: Collector,
  text: string,
  line: ContentLine,
): void {
  flush(collector);
  const item = toPlainText(text);
  // A quoted list is example output, not a change, so it never competes with
  // the release's own bullets. It stays as prose for a section without any.
  if (!line.quoted) {
    collector.current = { text: item, indent: line.indent };
  } else if (item) {
    collector.prose.push(truncate(item));
  }
}

function takeText(collector: Collector, text: string): void {
  const plain = toPlainText(text);
  if (!plain) {
    return;
  }
  if (collector.current === null) {
    // Wrapped paragraphs render as one block, so preview them as one item.
    collector.paragraph = collector.paragraph
      ? `${collector.paragraph} ${plain}`
      : plain;
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
  };

  for (const line of contentLines(markdown)) {
    if (!line.text || HEADING.test(line.text)) {
      flush(collector);
      continue;
    }
    // An indented code block renders as code, so a "- cmd" line inside one is
    // not a bullet. Indentation cannot start code inside an open bullet or
    // paragraph, where it is just a wrapped line.
    const insideBlock =
      collector.current !== null || collector.paragraph !== "";
    if (!insideBlock && line.indent >= INDENTED_CODE_INDENT) {
      continue;
    }
    const bullet = BULLET.exec(line.text);
    if (bullet) {
      takeBullet(collector, bullet[1] ?? "", line);
      continue;
    }
    takeText(collector, line.text);
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

  const { bullets, prose } = collectBullets(markdown);
  // Shallowest bullet defines top level, so a uniformly indented list still
  // previews instead of being read as all-nested.
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
