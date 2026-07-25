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
const IMAGE = /!\[[^\]]*\]\([^)]*\)/g;
const LINK = /\[([^\]]*)\]\([^)]*\)/g;
// Real tags only: a name character must follow "<", so a version constraint
// like "Support Python <3.15 and >3.9" keeps its operators.
const HTML_TAG = /<\/?[a-zA-Z][^>]*>/g;
// <https://x> and <a@b.c> are Markdown autolinks: keep the text they render.
const AUTOLINK = /<([a-zA-Z][a-zA-Z0-9+.-]*:[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>/g;
const CODE_SPAN_INLINE = /(`+)[\s\S]*?\1/g;
// CommonMark type 1 HTML blocks render literally. <details> is type 6 and
// does contain Markdown, so it is deliberately absent here.
const RAW_HTML_OPEN = /^ {0,3}<(pre|script|style|textarea)(?=[\s>]|$)/i;
const RAW_HTML_CLOSE = /<\/(pre|script|style|textarea)\s*>/i;
const COMMENT_OPEN = "<!--";
const COMMENT_CLOSE = "-->";
// Paired emphasis only. Underscores inside identifiers are literal, so
// UNSLOTH_DISABLE_UPDATE_CHECK keeps its name.
const BOLD_STAR = /\*\*(?=\S)([\s\S]*?\S)\*\*/g;
const BOLD_UNDERSCORE = /(^|[^\w])__(?=\S)([\s\S]*?\S)__(?=[^\w]|$)/g;
const ITALIC_STAR = /\*(?=\S)([^*\n]*?\S)\*/g;
const ITALIC_UNDERSCORE = /(^|[^\w])_(?=\S)([^_\n]*?\S)_(?=[^\w]|$)/g;
const BACKTICK = /`/g;
const CODE_SPAN = /`+([^`\n]+)`+/g;
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

/** Inline markdown stripped to plain text. */
function toPlainText(markdown: string): string {
  // Park code spans first: their contents are literal, so tags, links and
  // emphasis inside them must survive every transformation below.
  const codes: string[] = [];
  const parked = markdown.replace(CODE_SPAN, (_match, code: string) => {
    codes.push(code);
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

function contentLines(markdown: string): ContentLine[] {
  const lines: ContentLine[] = [];
  let openFence: string | null = null;
  let inComment = false;
  let inRawHtml = false;

  for (const rawLine of markdown.split("\n")) {
    const line = rawLine.replace(/\t/g, " ".repeat(TAB_WIDTH));
    const fence = inComment ? null : FENCE.exec(line);
    if (fence) {
      const marker = fence[1] ?? "";
      const rest = fence[2] ?? "";
      if (openFence === null) {
        openFence = marker;
      } else if (
        marker[0] === openFence[0] &&
        marker.length >= openFence.length &&
        rest.trim() === ""
      ) {
        // A closer carries nothing after it, so a ```` line with trailing
        // text inside a ```` block is content, not the end of the block.
        openFence = null;
      }
      lines.push({ text: "", indent: 0 });
      continue;
    }
    if (openFence !== null) {
      continue;
    }
    // Commented-out notes are not rendered, so they are not previewed either.
    const [uncommented, stillInComment] = stripCommentSpans(line, inComment);
    inComment = stillInComment;
    // Raw HTML blocks such as <pre> render literally, so their lines are not
    // release notes.
    const [visible, stillInRaw] = stripRawHtml(uncommented, inRawHtml);
    inRawHtml = stillInRaw;
    if (!visible.trim()) {
      lines.push({ text: "", indent: 0 });
      continue;
    }
    const stripped = visible.replace(BLOCKQUOTE, "");
    lines.push({
      text: stripped.trim(),
      indent: stripped.length - stripped.trimStart().length,
    });
  }
  return lines;
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

interface Bullet {
  text: string;
  indent: number;
}

/** Bullets in document order, plus prose for changelogs written as paragraphs. */
function collectBullets(markdown: string): {
  bullets: Bullet[];
  prose: string[];
} {
  const bullets: Bullet[] = [];
  const prose: string[] = [];
  // Wrapped bullets continue on following lines and belong to one item.
  let current: Bullet | null = null;
  let paragraph = "";

  const flush = () => {
    if (current?.text) {
      bullets.push({ text: truncate(current.text), indent: current.indent });
    }
    current = null;
    if (paragraph) {
      prose.push(truncate(paragraph));
      paragraph = "";
    }
  };

  for (const line of contentLines(markdown)) {
    if (!line.text || HEADING.test(line.text)) {
      flush();
      continue;
    }

    // An indented code block renders as code, so a "- cmd" line inside one is
    // not a bullet. Continuation lines of an open bullet still belong to it.
    if (current === null && line.indent >= INDENTED_CODE_INDENT) {
      continue;
    }

    const bullet = BULLET.exec(line.text);
    if (bullet) {
      flush();
      current = { text: toPlainText(bullet[1] ?? ""), indent: line.indent };
      continue;
    }

    const text = toPlainText(line.text);
    if (!text) {
      continue;
    }
    if (current === null) {
      // Wrapped paragraphs render as one block, so preview them as one item.
      paragraph = paragraph ? `${paragraph} ${text}` : text;
    } else {
      current = { text: `${current.text} ${text}`, indent: current.indent };
    }
  }
  flush();

  return { bullets, prose };
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
