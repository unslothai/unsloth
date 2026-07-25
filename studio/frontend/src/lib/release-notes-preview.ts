// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Top changelog bullets, shown in the collapsed update popup.
export const RELEASE_NOTES_PREVIEW_ITEMS = 4;
const PREVIEW_ITEM_MAX_CHARS = 120;
// Bullets indented past the shallowest one are nested detail, not headlines.
const NESTED_INDENT_TOLERANCE = 1;
const TAB_WIDTH = 4;

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
const CODE_SPAN_INLINE = /(`+)[\s\S]*?\1/g;
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
const SENTENCE_BREAK = /[.!?]\s+(?=["'“‘]?[A-Z0-9])/;
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
  // Code spans are literal, so park them before emphasis stripping and put
  // them back after: `__version__` must survive intact.
  const codes: string[] = [];
  const parked = stripHtmlTags(
    markdown.replace(IMAGE, "").replace(LINK, "$1"),
  ).replace(CODE_SPAN, (_match, code: string) => {
    codes.push(code);
    return `\uE000${codes.length - 1}\uE001`;
  });

  return parked
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

function contentLines(markdown: string): ContentLine[] {
  const lines: ContentLine[] = [];
  let openFence: string | null = null;
  let inComment = false;

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
    if (!uncommented.trim()) {
      lines.push({ text: "", indent: 0 });
      continue;
    }
    const stripped = uncommented.replace(BLOCKQUOTE, "");
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
  const match = SENTENCE_BREAK.exec(text);
  if (!match || match.index + 1 < MIN_LEAD_CHARS) {
    return { lead: text, rest: "" };
  }
  const cut = match.index + 1;
  return { lead: text.slice(0, cut).trim(), rest: text.slice(cut).trim() };
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

  const flush = () => {
    if (current?.text) {
      bullets.push({ text: truncate(current.text), indent: current.indent });
    }
    current = null;
  };

  for (const line of contentLines(markdown)) {
    if (!line.text || HEADING.test(line.text)) {
      flush();
      continue;
    }

    const bullet = BULLET.exec(line.text);
    if (bullet) {
      flush();
      current = { text: toPlainText(bullet[1] ?? ""), indent: line.indent };
      continue;
    }

    const text = toPlainText(line.text);
    if (current === null) {
      if (text) {
        prose.push(truncate(text));
      }
    } else if (text) {
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
