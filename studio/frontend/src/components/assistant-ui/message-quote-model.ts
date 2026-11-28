// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Stable quote sources are text selectors over an assistant message's rendered
// text index. DOM Ranges are capture-time inputs only; none are stored here.

import {
  type FindTextIndex,
  findMatches,
  foldText,
} from "../../features/find-in-page/lib/find-text-index.ts";

export const QUOTE_CONTEXT_CHARS = 64;
export const QUOTE_MATCH_LIMIT = 5_000;

export type QuoteMatchStatus = "exact" | "relocated";

export type QuoteAnchor = {
  readonly messageId: string;
  readonly turn: number;
  /** Visible selected text, used for the card and fallback search. */
  readonly text: string;
  /** Compact display-only text for table selections; never used for matching. */
  readonly previewText?: string;
  /** Markdown snapshot for a future collection insertion. */
  readonly markdown: string;
  /** Offset hints in the message's rendered text index at capture. */
  readonly start: number;
  readonly end: number;
  /** Normalized context used to disambiguate repeated text. */
  readonly prefix: string;
  readonly suffix: string;
  /** 1-based occurrence among matches at capture, for tie-breaking. */
  readonly occurrence: number;
};

export type CollectedQuote = {
  readonly id: string;
  readonly anchor: QuoteAnchor;
  /** User-provided context for the assistant; empty when omitted. */
  readonly comment: string;
  readonly createdAt: number;
};

export type QuoteMatch = {
  readonly start: number;
  readonly end: number;
  readonly status: QuoteMatchStatus;
};

export type QuoteAnchorInput = {
  readonly messageId: string;
  readonly turn: number;
  readonly text: string;
  /** Display-only; omitted by non-table selections and never used for matching. */
  readonly previewText?: string;
  readonly markdown: string;
  readonly index: FindTextIndex;
  readonly start: number;
  readonly end: number;
};

/** Clipboard and Range text can carry non-breaking spaces and CRLF. */
export function normalizeQuoteText(text: string): string {
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\u00a0/g, " ")
    .trim();
}

/** Comments are optional; surrounding whitespace or a blank draft is empty. */
export function normalizeQuoteComment(comment: string): string {
  return comment.replace(/\r\n/g, "\n").trim();
}

/**
 * The index writes a separator between rendered blocks; a DOM Range's
 * textContent does not. Prefer the indexed span only when the selection and
 * span differ exactly by those line breaks, so extra spaces or chrome still
 * make the captured text win.
 */
export function canonicalQuoteText(
  capturedText: string,
  indexedSpan: string,
): string {
  const captured = normalizeQuoteText(capturedText);
  const span = normalizeQuoteText(indexedSpan);
  const withoutBreaks = (text: string) => text.replace(/\n/g, "");
  if (
    withoutBreaks(foldText(span)) !== withoutBreaks(foldText(captured))
  ) {
    return captured;
  }

  // Rebuild the indexed separators into the original captured characters so
  // case is preserved; the index itself is lowercased for matching. Newlines
  // in the capture are structural and are replaced by the indexed ones.
  const characters = captured.replace(/\n/g, "");
  let canonical = "";
  let at = 0;
  for (let index = 0; index < span.length; index += 1) {
    if (span[index] === "\n") {
      canonical += "\n";
    } else {
      if (at >= characters.length) {
        return captured;
      }
      canonical += characters[at];
      at += 1;
    }
  }
  return at === characters.length ? canonical : captured;
}

/** Block separators are one code unit, so replacing them keeps every offset. */
export function quoteSearchIndex(index: FindTextIndex): FindTextIndex {
  return index.text.includes("\u0000")
    ? { ...index, text: index.text.replace(/\u0000/g, "\n") }
    : index;
}

function clampSpan(
  text: string,
  start: number,
  end: number,
): { start: number; end: number } | null {
  if (!Number.isFinite(start) || !Number.isFinite(end)) {
    return null;
  }
  const from = Math.max(0, Math.min(Math.trunc(start), text.length));
  const to = Math.max(from, Math.min(Math.trunc(end), text.length));
  return to > from ? { start: from, end: to } : null;
}

function occurrenceAt(
  index: FindTextIndex,
  text: string,
  before: number,
): number {
  const matches = findMatches(index, text, QUOTE_MATCH_LIMIT, 0);
  let occurrence = 1;
  for (const match of matches) {
    if (match.start >= before) {
      break;
    }
    occurrence += 1;
  }
  return occurrence;
}

function contextMatches(
  text: string,
  start: number,
  end: number,
  anchor: QuoteAnchor,
): boolean {
  return (
    text.endsWith(anchor.prefix, start) &&
    text.startsWith(anchor.suffix, end)
  );
}

function commonSuffixLength(text: string, suffix: string): number {
  let length = 0;
  while (
    length < suffix.length &&
    length < text.length &&
    text[text.length - 1 - length] === suffix[suffix.length - 1 - length]
  ) {
    length += 1;
  }
  return length;
}

function commonPrefixLength(text: string, prefix: string): number {
  let length = 0;
  while (
    length < prefix.length &&
    length < text.length &&
    text[length] === prefix[length]
  ) {
    length += 1;
  }
  return length;
}

/** Builds a serializable source anchor from capture-time text-index offsets. */
export function createQuoteAnchor(
  input: QuoteAnchorInput,
): QuoteAnchor | null {
  if (input.messageId.length === 0) {
    return null;
  }
  const text = normalizeQuoteText(input.text);
  if (text.length === 0) {
    return null;
  }
  const search = quoteSearchIndex(input.index);
  const span = clampSpan(search.text, input.start, input.end);
  if (span === null) {
    return null;
  }
  const previewText =
    input.previewText === undefined
      ? ""
      : normalizeQuoteText(input.previewText);
  return {
    messageId: input.messageId,
    turn: input.turn,
    text,
    ...(previewText.length > 0 ? { previewText } : {}),
    markdown: input.markdown,
    start: span.start,
    end: span.end,
    prefix: search.text.slice(
      Math.max(0, span.start - QUOTE_CONTEXT_CHARS),
      span.start,
    ),
    suffix: search.text.slice(
      span.end,
      Math.min(search.text.length, span.end + QUOTE_CONTEXT_CHARS),
    ),
    occurrence: occurrenceAt(search, text, span.start),
  };
}

/**
 * Resolves an anchor against the current message text. Uses the capture
 * offsets when they still carry the quote and its context; otherwise searches
 * the message and accepts a relocated match only when it is unambiguous.
 */
export function resolveQuoteAnchor(
  index: FindTextIndex,
  anchor: QuoteAnchor,
): QuoteMatch | null {
  const text = normalizeQuoteText(anchor.text);
  if (text.length === 0) {
    return null;
  }
  const search = quoteSearchIndex(index);
  const span = clampSpan(search.text, anchor.start, anchor.end);
  if (
    span !== null &&
    search.text.slice(span.start, span.end) === foldText(text) &&
    contextMatches(search.text, span.start, span.end, anchor)
  ) {
    return { start: span.start, end: span.end, status: "exact" };
  }

  const matches = findMatches(search, text, QUOTE_MATCH_LIMIT, 0);
  if (matches.length === 0) {
    return null;
  }

  let best: (typeof matches)[number] | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  let bestContext = 0;
  for (let at = 0; at < matches.length; at += 1) {
    const match = matches[at];
    const context =
      commonSuffixLength(
        search.text.slice(0, match.start),
        anchor.prefix,
      ) +
      commonPrefixLength(
        search.text.slice(match.end),
        anchor.suffix,
      );
    const occurrencePenalty = Math.abs(at + 1 - anchor.occurrence);
    const distance = Math.min(
      Math.abs(match.start - anchor.start),
      1_000_000,
    );
    const score = context * 2 - occurrencePenalty * 4 - distance / 1_000;
    if (score > bestScore) {
      best = match;
      bestScore = score;
      bestContext = context;
    }
  }

  // Repeated text with no surviving context is ambiguous. A stale card is
  // safer than navigating to the wrong passage.
  if (matches.length > 1 && bestContext === 0) {
    return null;
  }
  return best === null
    ? null
    : { start: best.start, end: best.end, status: "relocated" };
}
