// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The dispatch lifecycle is session-scoped. These tests pin snapshot
// ownership, exact-message reconciliation, late-callback safety, and the
// user-controlled release escape hatch.

import assert from "node:assert/strict";
import test from "node:test";

import {
  acceptQuoteDispatchForMessage,
  getPendingQuoteDispatchesForThread,
  getQuoteDispatch,
  markQuoteDispatchHistoryFailed,
  markQuoteDispatchHistoryOutcome,
  markQuoteDispatchHistoryPending,
  markQuoteDispatchSendFailed,
  markQuoteDispatchSendSettled,
  markQuoteDispatchSendStarted,
  quoteContextIdForMessage,
  quoteInfoForDispatch,
  releaseQuoteDispatchForMessage,
  releaseQuoteDispatchManually,
  reserveQuoteDispatch,
  subscribeQuoteDispatches,
  type QuoteDispatch,
} from "../src/components/assistant-ui/message-quote-dispatch.ts";
import type {
  CollectedQuote,
  QuoteAnchor,
} from "../src/components/assistant-ui/message-quote-model.ts";
import {
  addQuote,
  getQuotes,
  getReservedQuoteIds,
  removeQuote,
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

function quoteFor(id: string, text: string): CollectedQuote {
  return {
    id,
    anchor: anchorFor(`message-${id}`, text),
    comment: "",
    createdAt: 1,
  };
}

function add(thread: string, id: string, text = id): CollectedQuote {
  return addQuote(thread, quoteFor(id, text).anchor, { id });
}

function messageFor(dispatch: QuoteDispatch) {
  return {
    role: "user",
    metadata: { custom: { quote: quoteInfoForDispatch(dispatch) } },
  };
}

test("dispatch snapshots only unreserved quotes in collection order", () => {
  const thread = "dispatch-snapshot";
  add(thread, "q1", "first");
  add(thread, "q2", "second");

  const first = reserveQuoteDispatch(thread);
  assert.equal(first.kind, "reserved");
  if (first.kind !== "reserved") return;
  assert.deepEqual(first.dispatch.quoteIds, ["q1", "q2"]);
  assert.equal(first.dispatch.envelope.quotes[0].order, 1);
  assert.equal(first.dispatch.envelope.quotes[1].order, 2);
  assert.deepEqual([...getReservedQuoteIds(thread)], ["q1", "q2"]);
  assert.equal(removeQuote(thread, "q1"), false);

  add(thread, "q3", "third");
  const second = reserveQuoteDispatch(thread);
  assert.equal(second.kind, "reserved");
  if (second.kind !== "reserved") return;
  assert.deepEqual(second.dispatch.quoteIds, ["q3"]);
});

test("accepting one dispatch removes only its snapshot", () => {
  const thread = "dispatch-accept";
  add(thread, "q1");
  add(thread, "q2");
  const reserved = reserveQuoteDispatch(thread);
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;
  add(thread, "q3");

  assert.equal(
    acceptQuoteDispatchForMessage(messageFor(reserved.dispatch)),
    true,
  );
  assert.deepEqual(
    getQuotes(thread).map((quote) => quote.id),
    ["q3"],
  );
  assert.equal(removeQuote(thread, "q3"), true);
});

test("releasing keeps every quote and makes removal possible again", () => {
  const thread = "dispatch-release";
  add(thread, "q1");
  add(thread, "q2");
  const reserved = reserveQuoteDispatch(thread);
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;
  add(thread, "q3");

  assert.equal(releaseQuoteDispatchManually(reserved.dispatch.contextId), true);
  assert.deepEqual(
    getQuotes(thread).map((quote) => quote.id),
    ["q1", "q2", "q3"],
  );
  assert.equal(removeQuote(thread, "q1"), true);
});

test("a late acceptance cannot touch a newer dispatch", () => {
  const thread = "dispatch-late";
  add(thread, "q1");
  const first = reserveQuoteDispatch(thread);
  assert.equal(first.kind, "reserved");
  if (first.kind !== "reserved") return;
  assert.equal(releaseQuoteDispatchManually(first.dispatch.contextId), true);

  const second = reserveQuoteDispatch(thread);
  assert.equal(second.kind, "reserved");
  if (second.kind !== "reserved") return;

  assert.equal(
    acceptQuoteDispatchForMessage(messageFor(first.dispatch)),
    false,
  );
  assert.deepEqual([...getReservedQuoteIds(thread)], ["q1"]);
  acceptQuoteDispatchForMessage(messageFor(second.dispatch));
  assert.deepEqual(getQuotes(thread), []);
});

test("all lifecycle callbacks no-op after a manual release", () => {
  const thread = "dispatch-late-callbacks";
  add(thread, "q1");
  const reserved = reserveQuoteDispatch(thread);
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;
  const { contextId } = reserved.dispatch;

  assert.equal(markQuoteDispatchSendStarted(contextId), true);
  assert.equal(markQuoteDispatchSendSettled(contextId), true);
  assert.equal(markQuoteDispatchHistoryPending(contextId), true);
  assert.equal(releaseQuoteDispatchManually(contextId), true);

  assert.equal(markQuoteDispatchHistoryFailed(contextId), false);
  assert.equal(markQuoteDispatchHistoryOutcome(contextId), false);
  assert.equal(markQuoteDispatchSendFailed(contextId), false);
  assert.equal(markQuoteDispatchSendSettled(contextId), false);
  assert.equal(markQuoteDispatchHistoryPending(contextId), false);
  assert.equal(releaseQuoteDispatchForMessage(messageFor(reserved.dispatch)), false);
  assert.equal(getQuoteDispatch(contextId), null);
  assert.deepEqual(getQuotes(thread).map((quote) => quote.id), ["q1"]);
});

test("a pending dispatch can be released from every in-flight state", () => {
  const thread = "dispatch-manual-states";
  add(thread, "q1");

  const reserved = reserveQuoteDispatch(thread);
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind === "reserved") {
    assert.equal(
      releaseQuoteDispatchManually(reserved.dispatch.contextId),
      true,
    );
  }

  const sending = reserveQuoteDispatch(thread);
  assert.equal(sending.kind, "reserved");
  if (sending.kind === "reserved") {
    markQuoteDispatchSendStarted(sending.dispatch.contextId);
    assert.equal(
      releaseQuoteDispatchManually(sending.dispatch.contextId),
      true,
    );
  }

  const writing = reserveQuoteDispatch(thread);
  assert.equal(writing.kind, "reserved");
  if (writing.kind === "reserved") {
    markQuoteDispatchSendStarted(writing.dispatch.contextId);
    markQuoteDispatchSendSettled(writing.dispatch.contextId);
    markQuoteDispatchHistoryPending(writing.dispatch.contextId);
    assert.equal(
      releaseQuoteDispatchManually(writing.dispatch.contextId),
      true,
    );
  }

  assert.deepEqual(getPendingQuoteDispatchesForThread(thread), []);
  assert.deepEqual(getQuotes(thread).map((quote) => quote.id), ["q1"]);
});

test("invalid or empty snapshots fail without reserving anything", () => {
  const empty = "dispatch-empty";
  assert.deepEqual(reserveQuoteDispatch(empty), { kind: "none" });

  const duplicate = "dispatch-duplicate";
  add(duplicate, "q1", "first");
  add(duplicate, "q1", "second");
  assert.deepEqual(reserveQuoteDispatch(duplicate), { kind: "invalid" });
  assert.deepEqual([...getReservedQuoteIds(duplicate)], []);

  const emptyText = "dispatch-empty-text";
  addQuote(emptyText, anchorFor("message-empty", "   "), { id: "q1" });
  assert.deepEqual(reserveQuoteDispatch(emptyText), { kind: "invalid" });
});

test("pending snapshots are stable until lifecycle state changes", () => {
  const thread = "dispatch-subscribe";
  add(thread, "q1");
  const reserved = reserveQuoteDispatch(thread);
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;

  const first = getPendingQuoteDispatchesForThread(thread);
  assert.equal(getPendingQuoteDispatchesForThread(thread), first);
  markQuoteDispatchSendStarted(reserved.dispatch.contextId);
  assert.notEqual(getPendingQuoteDispatchesForThread(thread), first);
  assert.equal(
    getPendingQuoteDispatchesForThread(thread)[0]?.sendState,
    "pending",
  );
});

test("subscribers see pending lifecycle changes and manual release", () => {
  const thread = "dispatch-subscription";
  add(thread, "q1");
  let calls = 0;
  const unsubscribe = subscribeQuoteDispatches(thread, () => {
    calls += 1;
  });

  const reserved = reserveQuoteDispatch(thread);
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;
  assert.equal(calls, 1);
  markQuoteDispatchSendStarted(reserved.dispatch.contextId);
  assert.equal(calls, 2);
  releaseQuoteDispatchManually(reserved.dispatch.contextId);
  assert.equal(calls, 3);
  assert.equal(releaseQuoteDispatchManually(reserved.dispatch.contextId), false);
  assert.equal(calls, 3);

  unsubscribe();
  add(thread, "q2");
  reserveQuoteDispatch(thread);
  assert.equal(calls, 3);
});

test("a settled dispatch cannot be released again", () => {
  const thread = "dispatch-settled-release";
  add(thread, "q1");
  const reserved = reserveQuoteDispatch(thread);
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;
  assert.equal(
    acceptQuoteDispatchForMessage(messageFor(reserved.dispatch)),
    true,
  );
  assert.equal(
    releaseQuoteDispatchManually(reserved.dispatch.contextId),
    false,
  );
  assert.deepEqual(getQuotes(thread), []);
});

test("message helpers accept only valid user quote metadata", () => {
  const thread = "dispatch-message-shape";
  add(thread, "q1");
  const reserved = reserveQuoteDispatch(thread);
  assert.equal(reserved.kind, "reserved");
  if (reserved.kind !== "reserved") return;

  const message = messageFor(reserved.dispatch);
  assert.equal(
    quoteContextIdForMessage(message),
    reserved.dispatch.contextId,
  );
  assert.equal(
    quoteContextIdForMessage({ ...message, role: "assistant" }),
    null,
  );
  assert.equal(
    quoteContextIdForMessage({
      role: "user",
      metadata: { custom: { quote: { messageId: "other", text: "x" } } },
    }),
    null,
  );
});
