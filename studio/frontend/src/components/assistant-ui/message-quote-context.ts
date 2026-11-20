// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The composer quote transport only carries `{ messageId, text }`. Structured
// multi-quote context is encoded inside that text with a versioned marker and
// parsed back through a strict validator. Nothing raw ever reaches the model.

import type { CollectedQuote } from "./message-quote-model";

export const QUOTE_CONTEXT_MESSAGE_ID = "__unsloth_quote_context_v1__";
export const QUOTE_CONTEXT_PREFIX = "unsloth-quote-context:v1\n";

export type QuoteContextItem = {
  readonly order: number;
  readonly quoteId: string;
  readonly messageId: string;
  readonly turn: number;
  readonly text: string;
  readonly markdown: string;
  readonly comment: string;
};

export type QuoteContextEnvelope = {
  readonly version: 1;
  readonly contextId: string;
  readonly quotes: readonly QuoteContextItem[];
};

/** Minimal structural shape of assistant-ui's QuoteInfo. */
export type QuoteContextTransport = {
  readonly messageId: string;
  readonly text: string;
};

export type QuoteContextParseResult =
  | { readonly kind: "not-ours" }
  | { readonly kind: "invalid"; readonly reason: string }
  | { readonly kind: "valid"; readonly envelope: QuoteContextEnvelope };

export function createQuoteContextId(): string {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return cryptoApi.randomUUID();
  }
  return `quote-context-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2)}`;
}

function nonEmpty(value: string | null | undefined): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

/**
 * Copies the current collection in order. Returns null rather than emitting a
 * partial envelope if a quote is not serializable.
 */
export function buildQuoteContextEnvelope(
  quotes: readonly CollectedQuote[],
  contextId: string = createQuoteContextId(),
): QuoteContextEnvelope | null {
  if (quotes.length === 0 || !nonEmpty(contextId)) {
    return null;
  }

  const items: QuoteContextItem[] = [];
  for (const [index, quote] of quotes.entries()) {
    const { anchor } = quote;
    if (!(nonEmpty(anchor.messageId) && nonEmpty(anchor.text))) {
      return null;
    }
    if (!Number.isSafeInteger(anchor.turn) || anchor.turn < 0) {
      return null;
    }
    items.push({
      order: index + 1,
      quoteId: quote.id,
      messageId: anchor.messageId,
      turn: anchor.turn,
      text: anchor.text,
      markdown: anchor.markdown,
      comment: quote.comment,
    });
  }

  const envelope: QuoteContextEnvelope = {
    version: 1,
    contextId,
    quotes: items,
  };
  return validateEnvelope(envelope) === null ? envelope : null;
}

export function encodeQuoteContext(envelope: QuoteContextEnvelope): string {
  return `${QUOTE_CONTEXT_PREFIX}${JSON.stringify(envelope)}`;
}

export function quoteContextTransport(
  envelope: QuoteContextEnvelope,
): QuoteContextTransport {
  return {
    messageId: QUOTE_CONTEXT_MESSAGE_ID,
    text: encodeQuoteContext(envelope),
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Validates the parsed object. Null means valid; any string is a stable reason
 * code. Source message ids intentionally need not be unique because one
 * assistant response can contain several quoted passages.
 */
function validateQuoteItem(
  item: unknown,
  index: number,
  quoteIds: Set<string>,
): string | null {
  if (!isRecord(item)) {
    return "invalid-quote";
  }
  if (item.order !== index + 1) {
    return "invalid-order";
  }
  if (!nonEmpty(item.quoteId as string | null | undefined)) {
    return "invalid-quote-id";
  }
  const quoteId = item.quoteId as string;
  if (quoteIds.has(quoteId)) {
    return "duplicate-quote-id";
  }
  quoteIds.add(quoteId);
  if (!nonEmpty(item.messageId as string | null | undefined)) {
    return "invalid-message-id";
  }
  if (
    typeof item.turn !== "number" ||
    !Number.isSafeInteger(item.turn) ||
    item.turn < 0
  ) {
    return "invalid-turn";
  }
  if (!nonEmpty(item.text as string | null | undefined)) {
    return "empty-text";
  }
  if (typeof item.markdown !== "string") {
    return "invalid-markdown";
  }
  if (typeof item.comment !== "string") {
    return "invalid-comment";
  }
  return null;
}

function validateEnvelope(value: unknown): string | null {
  if (!isRecord(value)) {
    return "not-object";
  }
  if (value.version !== 1) {
    return "unsupported-version";
  }
  if (!nonEmpty(value.contextId as string | null | undefined)) {
    return "empty-context-id";
  }
  if (!Array.isArray(value.quotes) || value.quotes.length === 0) {
    return "empty-quotes";
  }

  const quoteIds = new Set<string>();
  for (const [index, item] of value.quotes.entries()) {
    const reason = validateQuoteItem(item, index, quoteIds);
    if (reason !== null) {
      return reason;
    }
  }

  return null;
}

/**
 * Only the synthetic id opts a quote into this format. A matching id with a
 * missing prefix or malformed body is invalid, never treated as plain text.
 */
export function parseQuoteContext(value: unknown): QuoteContextParseResult {
  if (!isRecord(value) || value.messageId !== QUOTE_CONTEXT_MESSAGE_ID) {
    return { kind: "not-ours" };
  }
  if (
    typeof value.text !== "string" ||
    !value.text.startsWith(QUOTE_CONTEXT_PREFIX)
  ) {
    return { kind: "invalid", reason: "missing-prefix" };
  }

  let decoded: unknown;
  try {
    decoded = JSON.parse(value.text.slice(QUOTE_CONTEXT_PREFIX.length));
  } catch {
    return { kind: "invalid", reason: "malformed-json" };
  }

  const reason = validateEnvelope(decoded);
  if (reason !== null) {
    return { kind: "invalid", reason };
  }
  return { kind: "valid", envelope: decoded as QuoteContextEnvelope };
}

/**
 * Model-only text for one user message. Validation is mandatory: foreign or
 * malformed quote metadata never becomes raw context.
 */
export function appendQuoteContextToText(text: string, quote: unknown): string {
  const parsed = parseQuoteContext(quote);
  if (parsed.kind !== "valid") {
    return text;
  }
  const context = formatQuoteContextForModel(parsed.envelope);
  return text.trim().length === 0 ? context : `${text}\n\n${context}`;
}

function blockquote(body: string): string {
  return body
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => (line.length > 0 ? `> ${line}` : ">"))
    .join("\n");
}

/**
 * Model-facing text for a validated envelope. It never includes the encoded
 * JSON, the context id, or the synthetic transport id.
 */
export function formatQuoteContextForModel(
  envelope: QuoteContextEnvelope,
): string {
  const blocks = envelope.quotes.map((quote) => {
    // The passage source is exactly one of these, never both.
    const markdown = quote.markdown.length > 0 ? quote.markdown : quote.text;
    const source =
      quote.turn > 0
        ? `Turn ${quote.turn} (message ${quote.messageId})`
        : `message ${quote.messageId}`;
    const lines = [
      `[${quote.order}] Source: ${source}`,
      "Passage:",
      blockquote(markdown),
    ];
    if (quote.comment.trim().length > 0) {
      lines.push("", `Comment: ${quote.comment}`);
    }
    return lines.join("\n");
  });

  return [
    "The user attached the following quoted passages as context for their message above. Preserve their order and treat the comments as user-provided context.",
    "",
    blocks.join("\n\n"),
  ].join("\n");
}
