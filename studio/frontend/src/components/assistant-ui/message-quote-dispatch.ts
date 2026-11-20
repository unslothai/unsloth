// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The live lifecycle of one normal composer submission's quote context. The
// store is deliberately session-scoped: accepted context lives on the message
// metadata, while this record exists only long enough to reconcile delivery.

import {
  type QuoteContextEnvelope,
  type QuoteContextTransport,
  buildQuoteContextEnvelope,
  parseQuoteContext,
  quoteContextTransport,
} from "./message-quote-context.ts";
import {
  getUnreservedQuotes,
  releaseQuoteIds,
  reserveQuoteIds,
  settleQuoteIds,
} from "./message-quote-store.ts";

export type QuoteDispatchSendState =
  | "not-started"
  | "pending"
  | "settled"
  | "failed";

export type QuoteDispatchHistoryState =
  | "not-started"
  | "pending"
  | "settled"
  | "failed";

export type QuoteDispatchOrigin = "composer" | "queue";

export type QuoteDispatchOutcome = "accepted" | "released";

export type QuoteDispatch = {
  readonly contextId: string;
  readonly threadId: string;
  readonly quoteIds: readonly string[];
  readonly envelope: QuoteContextEnvelope;
  readonly origin: QuoteDispatchOrigin;
  readonly sendState: QuoteDispatchSendState;
  readonly historyState: QuoteDispatchHistoryState;
};

export type ReserveQuoteDispatchResult =
  | { readonly kind: "none" }
  | { readonly kind: "invalid" }
  | { readonly kind: "reserved"; readonly dispatch: QuoteDispatch };

type QuoteMessageShape = {
  readonly role?: string;
  readonly metadata?: { readonly custom?: Record<string, unknown> };
};

const dispatches = new Map<string, QuoteDispatch>();
const listenersByThread = new Map<string, Set<() => void>>();
const snapshotsByThread = new Map<string, readonly QuoteDispatch[]>();
// Item-scoped terminal listeners. Entries exist only while a dispatch is live;
// the terminal notification removes them, so this is not an outcome ledger.
const outcomeListenersByContextId = new Map<
  string,
  Set<(outcome: QuoteDispatchOutcome) => void>
>();
const EMPTY_DISPATCHES: readonly QuoteDispatch[] = Object.freeze([]);

function announceQuoteDispatchOutcome(
  contextId: string,
  outcome: QuoteDispatchOutcome,
): void {
  const listeners = outcomeListenersByContextId.get(contextId);
  if (listeners === undefined) {
    return;
  }
  outcomeListenersByContextId.delete(contextId);
  for (const listener of [...listeners]) {
    listener(outcome);
  }
}

export function subscribeQuoteDispatchOutcome(
  contextId: string,
  listener: (outcome: QuoteDispatchOutcome) => void,
): () => void {
  let listeners = outcomeListenersByContextId.get(contextId);
  if (listeners === undefined) {
    listeners = new Set();
    outcomeListenersByContextId.set(contextId, listeners);
  }
  listeners.add(listener);
  return () => {
    const current = outcomeListenersByContextId.get(contextId);
    if (current === undefined) {
      return;
    }
    current.delete(listener);
    if (current.size === 0) {
      outcomeListenersByContextId.delete(contextId);
    }
  };
}

function notify(threadId: string): void {
  snapshotsByThread.delete(threadId);
  const listeners = listenersByThread.get(threadId);
  if (listeners === undefined) {
    return;
  }
  for (const listener of [...listeners]) {
    listener();
  }
}

function updateDispatch(
  contextId: string,
  update: (dispatch: QuoteDispatch) => QuoteDispatch | null,
): boolean {
  const dispatch = dispatches.get(contextId);
  if (dispatch === undefined) {
    return false;
  }
  const next = update(dispatch);
  if (next === null || next === dispatch) {
    return false;
  }
  dispatches.set(contextId, next);
  notify(dispatch.threadId);
  return true;
}

export function subscribeQuoteDispatches(
  threadId: string,
  listener: () => void,
): () => void {
  let listeners = listenersByThread.get(threadId);
  if (listeners === undefined) {
    listeners = new Set();
    listenersByThread.set(threadId, listeners);
  }
  listeners.add(listener);
  return () => {
    const current = listenersByThread.get(threadId);
    if (current === undefined) {
      return;
    }
    current.delete(listener);
    if (current.size === 0) {
      listenersByThread.delete(threadId);
    }
  };
}

export function getPendingQuoteDispatchesForThread(
  threadId: string,
): readonly QuoteDispatch[] {
  const cached = snapshotsByThread.get(threadId);
  if (cached !== undefined) {
    return cached;
  }
  const pending: QuoteDispatch[] = [];
  for (const dispatch of dispatches.values()) {
    if (dispatch.threadId === threadId) {
      pending.push(dispatch);
    }
  }
  const snapshot =
    pending.length === 0 ? EMPTY_DISPATCHES : Object.freeze(pending);
  snapshotsByThread.set(threadId, snapshot);
  return snapshot;
}

export function getQuoteDispatch(contextId: string): QuoteDispatch | null {
  return dispatches.get(contextId) ?? null;
}

/**
 * Snapshots only currently unreserved quotes. A duplicate ID is an invariant
 * violation, not something to silently deduplicate.
 */
export function reserveQuoteDispatch(
  threadId: string,
  origin: QuoteDispatchOrigin = "composer",
): ReserveQuoteDispatchResult {
  const quotes = getUnreservedQuotes(threadId);
  if (quotes.length === 0) {
    return { kind: "none" };
  }
  const envelope = buildQuoteContextEnvelope(quotes);
  if (envelope === null) {
    return { kind: "invalid" };
  }
  const quoteIds = envelope.quotes.map((quote) => quote.quoteId);
  if (!reserveQuoteIds(threadId, envelope.contextId, quoteIds)) {
    return { kind: "invalid" };
  }
  const dispatch: QuoteDispatch = Object.freeze({
    contextId: envelope.contextId,
    threadId,
    quoteIds: Object.freeze(quoteIds),
    envelope,
    origin,
    sendState: "not-started",
    historyState: "not-started",
  });
  dispatches.set(envelope.contextId, dispatch);
  notify(threadId);
  return { kind: "reserved", dispatch };
}

export function quoteInfoForDispatch(
  dispatch: QuoteDispatch,
): QuoteContextTransport {
  return quoteContextTransport(dispatch.envelope);
}

export function quoteContextIdForMessage(message: unknown): string | null {
  const typed = message as QuoteMessageShape | null;
  if (
    typed === null ||
    typeof typed !== "object" ||
    (typed.role !== undefined && typed.role !== "user")
  ) {
    return null;
  }
  const parsed = parseQuoteContext(typed.metadata?.custom?.quote);
  return parsed.kind === "valid" ? parsed.envelope.contextId : null;
}

export function markQuoteDispatchSendStarted(contextId: string): boolean {
  return updateDispatch(contextId, (dispatch) =>
    dispatch.sendState === "not-started"
      ? { ...dispatch, sendState: "pending" }
      : null,
  );
}

export function markQuoteDispatchSendSettled(contextId: string): boolean {
  return updateDispatch(contextId, (dispatch) =>
    dispatch.sendState === "pending"
      ? { ...dispatch, sendState: "settled" }
      : null,
  );
}

export function markQuoteDispatchSendFailed(contextId: string): boolean {
  return updateDispatch(contextId, (dispatch) =>
    dispatch.sendState === "pending" || dispatch.sendState === "not-started"
      ? { ...dispatch, sendState: "failed" }
      : null,
  );
}

/**
 * Restores a failed append attempt to the queued state for retry. The internal
 * guard is deliberate: a caller must not make a history-started dispatch look
 * queued again.
 */
export function resetQuoteDispatchForRetry(contextId: string): boolean {
  return updateDispatch(contextId, (dispatch) =>
    dispatch.sendState === "failed" && dispatch.historyState === "not-started"
      ? { ...dispatch, sendState: "not-started" }
      : null,
  );
}

/**
 * Queue cleanup may release only when no message and no history write can
 * exist: history never started, and the append either never began or has
 * already settled as failed. A pending send may still land.
 */
export function canReleaseQueuedQuoteDispatch(
  dispatch: QuoteDispatch | null,
): boolean {
  return (
    dispatch !== null &&
    dispatch.historyState === "not-started" &&
    (dispatch.sendState === "not-started" || dispatch.sendState === "failed")
  );
}

/** Queue snapshots are not "being sent" until their append actually starts. */
export function shouldShowQuoteDispatchInPill(
  dispatch: QuoteDispatch,
): boolean {
  return !(
    dispatch.origin === "queue" && dispatch.sendState === "not-started"
  );
}

export function markQuoteDispatchHistoryPending(contextId: string): boolean {
  return updateDispatch(contextId, (dispatch) =>
    dispatch.historyState === "not-started"
      ? { ...dispatch, historyState: "pending" }
      : null,
  );
}

export function markQuoteDispatchHistoryOutcome(contextId: string): boolean {
  return updateDispatch(contextId, (dispatch) =>
    dispatch.historyState === "pending"
      ? { ...dispatch, historyState: "settled" }
      : null,
  );
}

export function markQuoteDispatchHistoryFailed(contextId: string): boolean {
  return updateDispatch(contextId, (dispatch) =>
    dispatch.historyState === "pending"
      ? { ...dispatch, historyState: "failed" }
      : null,
  );
}

export function acceptQuoteDispatchForMessage(message: unknown): boolean {
  const contextId = quoteContextIdForMessage(message);
  if (contextId === null) {
    return false;
  }
  const dispatch = dispatches.get(contextId);
  if (dispatch === undefined) {
    return false;
  }
  const settled = settleQuoteIds(dispatch.threadId, contextId);
  // Announce before deletion: queue items consume this as the authoritative
  // accepted/released outcome rather than inferring it from the collection.
  announceQuoteDispatchOutcome(contextId, "accepted");
  dispatches.delete(contextId);
  notify(dispatch.threadId);
  return settled;
}

export function releaseQuoteDispatchForMessage(message: unknown): boolean {
  const contextId = quoteContextIdForMessage(message);
  return contextId === null ? false : releaseQuoteDispatch(contextId);
}

/**
 * Removes one pending dispatch and releases its quote IDs. A late send/history
 * callback finds no dispatch and becomes a no-op.
 */
export function releaseQuoteDispatch(contextId: string): boolean {
  const dispatch = dispatches.get(contextId);
  if (dispatch === undefined) {
    return false;
  }
  const released = releaseQuoteIds(dispatch.threadId, contextId);
  // Announce before deletion; see acceptQuoteDispatchForMessage.
  announceQuoteDispatchOutcome(contextId, "released");
  dispatches.delete(contextId);
  notify(dispatch.threadId);
  return released;
}

export function releaseQuoteDispatchManually(contextId: string): boolean {
  return releaseQuoteDispatch(contextId);
}
