// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Top changelog bullets, shown in the collapsed update popup.
export const RELEASE_NOTES_PREVIEW_ITEMS = 4;
const PREVIEW_ITEM_MAX_CHARS = 120;

const FENCE = /^\s*(?:```|~~~)/;
const HEADING = /^\s{0,3}#{1,6}\s+/;
const BULLET = /^\s*(?:[-*+]|\d+[.)])\s+(.*)$/;
const BLOCKQUOTE = /^\s*>\s?/;
const IMAGE = /!\[[^\]]*\]\([^)]*\)/g;
const LINK = /\[([^\]]*)\]\([^)]*\)/g;
const HTML_TAG = /<[^>]+>/g;
const EMPHASIS = /(\*\*|__|\*|_|`)/g;
const WHITESPACE = /\s+/g;
// Sentence end followed by something that actually starts a sentence.
const SENTENCE_BREAK = /[.!?]\s+(?=["'“‘]?[A-Z0-9])/;
const MIN_LEAD_CHARS = 12;

/** Inline markdown stripped to plain text. */
function toPlainText(markdown: string): string {
  return markdown
    .replace(IMAGE, "")
    .replace(LINK, "$1")
    .replace(HTML_TAG, "")
    .replace(EMPHASIS, "")
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

/** Trimmed lines outside fenced code blocks; fences act as item boundaries. */
function contentLines(markdown: string): string[] {
  const lines: string[] = [];
  let inFence = false;

  for (const rawLine of markdown.split("\n")) {
    if (FENCE.test(rawLine)) {
      inFence = !inFence;
      lines.push("");
      continue;
    }
    if (!inFence) {
      lines.push(rawLine.replace(BLOCKQUOTE, "").trim());
    }
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

/**
 * Top-level bullets of a release section, in document order. Falls back to
 * prose when a release has no bullets.
 */
export function releaseNotesPreview(
  markdown: string | null | undefined,
  limit: number = RELEASE_NOTES_PREVIEW_ITEMS,
): ReleaseNotesPreview {
  if (!markdown) {
    return { items: [], remaining: 0 };
  }

  const bullets: string[] = [];
  const prose: string[] = [];
  // Wrapped bullets continue on following lines and belong to one item.
  let current: string | null = null;

  const flush = () => {
    if (current) {
      bullets.push(truncate(current));
    }
    current = null;
  };

  for (const line of contentLines(markdown)) {
    if (!line || HEADING.test(line)) {
      flush();
      continue;
    }

    const bullet = BULLET.exec(line);
    if (bullet) {
      flush();
      current = toPlainText(bullet[1] ?? "") || null;
      continue;
    }

    const text = toPlainText(line);
    if (current === null) {
      if (text) {
        prose.push(truncate(text));
      }
    } else if (text) {
      current = `${current} ${text}`;
    }
  }
  flush();

  const source = bullets.length > 0 ? bullets : prose;
  return {
    items: source.slice(0, limit).map(splitLeadSentence),
    remaining: Math.max(source.length - limit, 0),
  };
}
