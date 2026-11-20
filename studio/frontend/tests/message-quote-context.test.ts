// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The composer transport is a synthetic QuoteInfo carrying a versioned,
// strictly validated envelope. These tests prove that malformed or foreign
// quote metadata can never leak raw JSON into model-facing content.

import assert from "node:assert/strict";
import test from "node:test";

import {
  appendQuoteContextToText,
  buildQuoteContextEnvelope,
  formatQuoteContextForModel,
  parseQuoteContext,
  QUOTE_CONTEXT_MESSAGE_ID,
  QUOTE_CONTEXT_PREFIX,
  quoteContextTransport,
  type QuoteContextEnvelope,
} from "../src/components/assistant-ui/message-quote-context.ts";
import type {
  CollectedQuote,
  QuoteAnchor,
} from "../src/components/assistant-ui/message-quote-model.ts";

function anchorFor(
  messageId: string,
  text: string,
  overrides: Partial<QuoteAnchor> = {},
): QuoteAnchor {
  return {
    messageId,
    turn: 2,
    text,
    markdown: `**${text}**`,
    start: 0,
    end: text.length,
    prefix: "",
    suffix: "",
    occurrence: 1,
    ...overrides,
  };
}

function quoteFor(
  id: string,
  messageId: string,
  text: string,
  comment: string,
): CollectedQuote {
  return {
    id,
    anchor: anchorFor(messageId, text),
    comment,
    createdAt: 1,
  };
}

const envelope: QuoteContextEnvelope = {
  version: 1,
  contextId: "ctx-1",
  quotes: [
    {
      order: 1,
      quoteId: "q1",
      messageId: "message-1",
      turn: 2,
      text: "First passage",
      markdown: "**First passage**",
      comment: "Use this as the definition.",
    },
    {
      order: 2,
      quoteId: "q2",
      messageId: "message-2",
      turn: 4,
      text: "Second passage",
      markdown:
        "Second paragraph with `code`.\n\n- a list item\n- another item",
      comment: "",
    },
  ],
};

function validTransport(value: QuoteContextEnvelope = envelope) {
  return quoteContextTransport(value);
}

test("collection snapshots preserve order, sources, markdown, and comments", () => {
  const built = buildQuoteContextEnvelope(
    [
      quoteFor("q1", "message-1", "First passage", "Definition."),
      quoteFor("q2", "message-2", "Second passage", ""),
    ],
    "ctx-1",
  );

  assert.ok(built);
  assert.equal(built.version, 1);
  assert.equal(built.contextId, "ctx-1");
  assert.equal(built.quotes[0].order, 1);
  assert.equal(built.quotes[1].order, 2);
  assert.equal(built.quotes[0].messageId, "message-1");
  assert.equal(built.quotes[0].markdown, "**First passage**");
  assert.equal(built.quotes[0].comment, "Definition.");
  assert.equal(built.quotes[1].comment, "");
});

test("quotes from the same assistant message keep separate identities", () => {
  const built = buildQuoteContextEnvelope(
    [
      quoteFor("q1", "same-message", "Alpha", "first"),
      quoteFor("q2", "same-message", "Beta", "second"),
    ],
    "ctx-same-source",
  );

  assert.ok(built);
  assert.equal(built.quotes[0].messageId, built.quotes[1].messageId);
  assert.notEqual(built.quotes[0].quoteId, built.quotes[1].quoteId);
  assert.equal(parseQuoteContext(validTransport(built)).kind, "valid");
});

test("a valid transport round-trips through the versioned envelope", () => {
  const transport = validTransport();

  assert.equal(transport.messageId, QUOTE_CONTEXT_MESSAGE_ID);
  assert.equal(transport.text.startsWith(QUOTE_CONTEXT_PREFIX), true);

  const parsed = parseQuoteContext(transport);
  assert.equal(parsed.kind, "valid");
  if (parsed.kind === "valid") {
    assert.deepEqual(parsed.envelope, envelope);
  }
});

test("foreign quote metadata is left alone rather than reinterpreted", () => {
  assert.deepEqual(parseQuoteContext(null), { kind: "not-ours" });
  assert.deepEqual(parseQuoteContext(undefined), { kind: "not-ours" });
  assert.deepEqual(parseQuoteContext({ messageId: "real-message", text: "Hi" }), {
    kind: "not-ours",
  });
  assert.deepEqual(
    parseQuoteContext({ messageId: QUOTE_CONTEXT_MESSAGE_ID, text: 42 }),
    { kind: "invalid", reason: "missing-prefix" },
  );
});

test("our marker with a missing or malformed body is invalid", () => {
  assert.deepEqual(
    parseQuoteContext({
      messageId: QUOTE_CONTEXT_MESSAGE_ID,
      text: "ordinary quote text that must never be sent",
    }),
    { kind: "invalid", reason: "missing-prefix" },
  );
  assert.deepEqual(
    parseQuoteContext({
      messageId: QUOTE_CONTEXT_MESSAGE_ID,
      text: `${QUOTE_CONTEXT_PREFIX}{broken`,
    }),
    { kind: "invalid", reason: "malformed-json" },
  );
});

test("strict validation rejects wrong versions and missing structure", () => {
  const wrongVersion = { ...envelope, version: 2 };
  assert.deepEqual(
    parseQuoteContext({
      messageId: QUOTE_CONTEXT_MESSAGE_ID,
      text: `${QUOTE_CONTEXT_PREFIX}${JSON.stringify(wrongVersion)}`,
    }),
    { kind: "invalid", reason: "unsupported-version" },
  );

  assert.deepEqual(
    parseQuoteContext({
      messageId: QUOTE_CONTEXT_MESSAGE_ID,
      text: `${QUOTE_CONTEXT_PREFIX}${JSON.stringify({
        ...envelope,
        contextId: "  ",
      })}`,
    }),
    { kind: "invalid", reason: "empty-context-id" },
  );

  assert.deepEqual(
    parseQuoteContext({
      messageId: QUOTE_CONTEXT_MESSAGE_ID,
      text: `${QUOTE_CONTEXT_PREFIX}${JSON.stringify({
        ...envelope,
        quotes: [],
      })}`,
    }),
    { kind: "invalid", reason: "empty-quotes" },
  );
});

test("strict validation rejects malformed quote items", () => {
  const invalid = [
    [{ ...envelope.quotes[0], order: 2 }, "invalid-order"],
    [{ ...envelope.quotes[0], quoteId: "" }, "invalid-quote-id"],
    [{ ...envelope.quotes[0], messageId: "" }, "invalid-message-id"],
    [{ ...envelope.quotes[0], turn: -1 }, "invalid-turn"],
    [{ ...envelope.quotes[0], turn: 1.5 }, "invalid-turn"],
    [{ ...envelope.quotes[0], text: "   " }, "empty-text"],
    [{ ...envelope.quotes[0], markdown: null }, "invalid-markdown"],
    [{ ...envelope.quotes[0], comment: 5 }, "invalid-comment"],
  ] as const;

  for (const [item, reason] of invalid) {
    const parsed = parseQuoteContext(encodeTransport({ ...envelope, quotes: [item] }));
    assert.deepEqual(parsed, { kind: "invalid", reason }, reason);
  }

  assert.deepEqual(
    parseQuoteContext(
      encodeTransport({
        ...envelope,
        quotes: [
          envelope.quotes[0],
          { ...envelope.quotes[1], quoteId: envelope.quotes[0].quoteId },
        ],
      }),
    ),
    { kind: "invalid", reason: "duplicate-quote-id" },
  );
});

function encodeTransport(value: unknown) {
  return {
    messageId: QUOTE_CONTEXT_MESSAGE_ID,
    text: `${QUOTE_CONTEXT_PREFIX}${JSON.stringify(value)}`,
  };
}

test("builders refuse partial or empty collections", () => {
  assert.equal(buildQuoteContextEnvelope([], "ctx-1"), null);
  assert.equal(
    buildQuoteContextEnvelope(
      [quoteFor("q1", "message-1", "passage", "")],
      "",
    ),
    null,
  );
  assert.equal(
    buildQuoteContextEnvelope(
      [quoteFor("q1", "message-1", "   ", "")],
      "ctx-1",
    ),
    null,
  );
  assert.equal(
    buildQuoteContextEnvelope(
      [quoteFor("", "message-1", "passage", "")],
      "ctx-1",
    ),
    null,
  );
});

test("model formatting preserves order and comments without leaking the envelope", () => {
  const text = formatQuoteContextForModel(envelope);

  assert.match(text, /quoted passages as context for their message above/);
  assert.match(text, /\[1\] Source: Turn 2 \(message message-1\)/);
  assert.match(text, /\[2\] Source: Turn 4 \(message message-2\)/);
  assert.ok(text.indexOf("[1]") < text.indexOf("[2]"));
  assert.match(text, /> \*\*First passage\*\*/);
  assert.match(text, /Comment: Use this as the definition\./);
  assert.match(text, /> Second paragraph with `code`\./);
  // The second quote has no comment, so no empty label is emitted.
  assert.equal(text.match(/Comment:/g)?.length, 1);

  assert.doesNotMatch(text, /unsloth-quote-context/);
  assert.doesNotMatch(text, /__unsloth_quote_context/);
  assert.doesNotMatch(text, /"version"\s*:/);
  assert.doesNotMatch(text, /ctx-1/);
});

test("model formatting falls back to visible text when markdown is empty", () => {
  const withoutMarkdown: QuoteContextEnvelope = {
    ...envelope,
    quotes: [
      {
        ...envelope.quotes[0],
        markdown: "",
        comment: "",
      },
    ],
  };

  assert.match(
    formatQuoteContextForModel(withoutMarkdown),
    /> First passage/,
  );
});

test("the model text keeps the typed message exact and appends one context block", () => {
  const transport = quoteContextTransport(envelope);
  const typed = "Keep  this text\nwith its spacing.";
  const combined = appendQuoteContextToText(typed, transport);

  assert.ok(combined.startsWith(`${typed}\n\n`));
  assert.match(combined, /\[1\] Source: Turn 2 \(message message-1\)/);
  assert.match(combined, /Comment: Use this as the definition\./);
});

test("attachment-only text has no leading separator", () => {
  const transport = quoteContextTransport(envelope);
  const combined = appendQuoteContextToText("   \n", transport);

  assert.ok(combined.startsWith("The user attached the following"));
  assert.doesNotMatch(combined, /^\s/);
});

test("foreign or malformed quote metadata leaves model text untouched", () => {
  assert.equal(
    appendQuoteContextToText("typed", {
      messageId: "real-message",
      text: "foreign",
    }),
    "typed",
  );
  assert.equal(
    appendQuoteContextToText("typed", {
      messageId: QUOTE_CONTEXT_MESSAGE_ID,
      text: `${QUOTE_CONTEXT_PREFIX}{broken`,
    }),
    "typed",
  );
});

test("the passage uses Markdown when non-empty and text only otherwise", () => {
  const markdownEnvelope: QuoteContextEnvelope = {
    ...envelope,
    quotes: [
      {
        ...envelope.quotes[0],
        text: "Plain passage",
        markdown: "**Bold passage**",
      },
    ],
  };
  const markdownText = formatQuoteContextForModel(markdownEnvelope);
  assert.match(markdownText, /> \*\*Bold passage\*\*/);
  assert.doesNotMatch(markdownText, /> Plain passage/);

  const textOnlyEnvelope: QuoteContextEnvelope = {
    ...markdownEnvelope,
    quotes: [{ ...markdownEnvelope.quotes[0], markdown: "" }],
  };
  const textOnly = formatQuoteContextForModel(textOnlyEnvelope);
  assert.match(textOnly, /> Plain passage/);
  assert.doesNotMatch(textOnly, /Bold passage/);
});

test("parsing never returns a raw JSON body as ordinary quote text", () => {
  const parsed = parseQuoteContext(
    encodeTransport({ ...envelope, version: 99 }),
  );

  assert.equal(parsed.kind, "invalid");
  if (parsed.kind === "invalid") {
    assert.equal(parsed.reason, "unsupported-version");
    assert.doesNotMatch(parsed.reason, /First passage/);
  }
});
