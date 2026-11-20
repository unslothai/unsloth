// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The collection is session-scoped and per-thread; cards are plain anchors,
// not DOM references. These tests cover order, removal, isolation, and
// subscriptions without mounting React.

import assert from "node:assert/strict";
import test from "node:test";

import type { QuoteAnchor } from "../src/components/assistant-ui/message-quote-model.ts";
import {
  addQuote,
  clearQuotes,
  createEphemeralQuoteThreadId,
  getQuotes,
  getReservedQuoteIds,
  getUnreservedQuotes,
  releaseQuoteIds,
  removeQuote,
  reserveQuoteIds,
  resolveQuoteCollectionKey,
  settleQuoteIds,
  subscribeQuotes,
} from "../src/components/assistant-ui/message-quote-store.ts";

function anchorFor(messageId: string, text: string): QuoteAnchor {
  return {
    messageId,
    turn: 1,
    text,
    markdown: text,
    start: 0,
    end: text.length,
    prefix: "",
    suffix: "",
    occurrence: 1,
  };
}

test("quotes append in collection order", () => {
  const thread = "quote-store-order";
  const empty = getQuotes(thread);

  const first = addQuote(thread, anchorFor("message-1", "first"), {
    id: "q1",
    createdAt: 1,
  });
  const second = addQuote(thread, anchorFor("message-2", "second"), {
    id: "q2",
    createdAt: 2,
  });

  assert.deepEqual(first, {
    id: "q1",
    anchor: anchorFor("message-1", "first"),
    comment: "",
    createdAt: 1,
  });
  assert.deepEqual(
    getQuotes(thread).map((quote) => quote.id),
    ["q1", "q2"],
  );
  assert.equal(second.anchor.messageId, "message-2");
  assert.deepEqual(empty, []);
  assert.notEqual(getQuotes(thread), empty);
  assert.equal(getQuotes(thread), getQuotes(thread));
});

test("removing a middle quote preserves the rest of the order", () => {
  const thread = "quote-store-remove";
  addQuote(thread, anchorFor("message-1", "one"), { id: "q1" });
  addQuote(thread, anchorFor("message-2", "two"), { id: "q2" });
  addQuote(thread, anchorFor("message-3", "three"), { id: "q3" });

  assert.equal(removeQuote(thread, "q2"), true);
  assert.deepEqual(
    getQuotes(thread).map((quote) => quote.id),
    ["q1", "q3"],
  );
  assert.equal(removeQuote(thread, "missing"), false);
});

test("threads keep independent collections", () => {
  const left = "quote-store-left";
  const right = "quote-store-right";
  addQuote(left, anchorFor("left-message", "left"), { id: "left" });
  addQuote(right, anchorFor("right-message", "right"), { id: "right" });

  assert.deepEqual(
    getQuotes(left).map((quote) => quote.id),
    ["left"],
  );
  assert.deepEqual(
    getQuotes(right).map((quote) => quote.id),
    ["right"],
  );
});

test("subscribers see appends, removals, and clears until they unsubscribe", () => {
  const thread = "quote-store-subscribe";
  let calls = 0;
  const unsubscribe = subscribeQuotes(thread, () => {
    calls += 1;
  });

  addQuote(thread, anchorFor("message-1", "one"), { id: "q1" });
  assert.equal(calls, 1);
  addQuote(thread, anchorFor("message-2", "two"), { id: "q2" });
  assert.equal(calls, 2);
  assert.equal(removeQuote(thread, "q1"), true);
  assert.equal(calls, 3);
  assert.equal(removeQuote(thread, "missing"), false);
  assert.equal(calls, 3);
  assert.equal(clearQuotes(thread), true);
  assert.equal(calls, 4);
  assert.deepEqual(getQuotes(thread), []);
  assert.equal(clearQuotes(thread), false);
  assert.equal(calls, 4);

  unsubscribe();
  addQuote(thread, anchorFor("message-3", "three"), { id: "q3" });
  assert.equal(calls, 4);
});

test("default quote identity and timestamp are supplied when omitted", () => {
  const thread = "quote-store-defaults";
  const quote = addQuote(thread, anchorFor("message-1", "passage"));

  assert.ok(quote.id.length > 0);
  assert.equal(typeof quote.createdAt, "number");
  assert.equal(getQuotes(thread)[0], quote);
});

test("ephemeral thread ids are non-empty and unique", () => {
  const left = createEphemeralQuoteThreadId();
  const right = createEphemeralQuoteThreadId();

  assert.ok(left.length > 0);
  assert.ok(right.length > 0);
  assert.notEqual(left, right);
  assert.equal(left.startsWith("unsaved-quote-"), true);
});

test("collection keys stay with an unsaved thread when its backend id appears", () => {
  const ephemeral = createEphemeralQuoteThreadId();
  const runtime = "__LOCALID_quote-thread";

  assert.equal(resolveQuoteCollectionKey(runtime, null, ephemeral), runtime);
  // assistant-ui keeps the local item id stable while activeThreadId moves
  // from the empty value to the stored remote id.
  assert.equal(
    resolveQuoteCollectionKey(runtime, "remote-thread-id", ephemeral),
    runtime,
  );
  assert.equal(
    resolveQuoteCollectionKey(undefined, "remote-thread-id", ephemeral),
    "remote-thread-id",
  );
  assert.equal(resolveQuoteCollectionKey("", null, ephemeral), ephemeral);
  assert.notEqual(
    resolveQuoteCollectionKey("__LOCALID_other", null, ephemeral),
    runtime,
  );
});

test("reservations claim exact IDs and block removal until settled", () => {
  const thread = "quote-store-reserve";
  addQuote(thread, anchorFor("message-1", "one"), { id: "q1" });
  addQuote(thread, anchorFor("message-2", "two"), { id: "q2" });
  addQuote(thread, anchorFor("message-3", "three"), { id: "q3" });

  assert.equal(reserveQuoteIds(thread, "ctx-a", ["q1", "q2"]), true);
  assert.deepEqual([...getReservedQuoteIds(thread)], ["q1", "q2"]);
  assert.deepEqual(
    getUnreservedQuotes(thread).map((quote) => quote.id),
    ["q3"],
  );
  assert.equal(removeQuote(thread, "q1"), false);
  assert.equal(removeQuote(thread, "q3"), true);

  assert.equal(releaseQuoteIds(thread, "ctx-a"), true);
  assert.deepEqual(
    getQuotes(thread).map((quote) => quote.id),
    ["q1", "q2"],
  );
  assert.equal(reserveQuoteIds(thread, "ctx-b", ["q1"]), true);
  assert.equal(settleQuoteIds(thread, "ctx-b"), true);
  assert.deepEqual(
    getQuotes(thread).map((quote) => quote.id),
    ["q2"],
  );
});

test("reservation is atomic, deduplicated, and thread-isolated", () => {
  const left = "quote-store-reserve-left";
  const right = "quote-store-reserve-right";
  addQuote(left, anchorFor("left-message", "left"), { id: "left" });
  addQuote(right, anchorFor("right-message", "right"), { id: "right" });

  assert.equal(reserveQuoteIds(left, "ctx", ["left", "left"]), false);
  assert.equal(reserveQuoteIds(left, "ctx", ["left", "missing"]), false);
  assert.deepEqual(getReservedQuoteIds(left), new Set());
  assert.equal(reserveQuoteIds(left, "ctx", ["left"]), true);
  assert.deepEqual([...getReservedQuoteIds(left)], ["left"]);
  assert.deepEqual([...getReservedQuoteIds(right)], []);
  assert.equal(removeQuote(right, "right"), true);
});

test("comments stay attached to their passage and blank comments normalize away", () => {
  const thread = "quote-store-comments";
  const first = addQuote(thread, anchorFor("message-1", "first"), {
    id: "q1",
    comment: "  Explain this\r\nwith the next step.  ",
  });
  const second = addQuote(thread, anchorFor("message-2", "second"), {
    id: "q2",
    comment: "   ",
  });
  const third = addQuote(thread, anchorFor("message-3", "third"), {
    id: "q3",
  });

  assert.equal(first.comment, "Explain this\nwith the next step.");
  assert.equal(second.comment, "");
  assert.equal(third.comment, "");
  assert.deepEqual(
    getQuotes(thread).map((quote) => [quote.anchor.text, quote.comment]),
    [
      ["first", "Explain this\nwith the next step."],
      ["second", ""],
      ["third", ""],
    ],
  );
});
