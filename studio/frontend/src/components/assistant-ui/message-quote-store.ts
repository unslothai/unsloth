// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Session-scoped quote collections keyed by thread id. The store is outside
// the virtualized message list, so a card survives its source message being
// unmounted or remounted.

import {
  type CollectedQuote,
  type QuoteAnchor,
  normalizeQuoteComment,
} from "./message-quote-model.ts";

export type AddQuoteOptions = {
  readonly id?: string;
  readonly comment?: string;
  readonly createdAt?: number;
};

const quotesByThread = new Map<string, readonly CollectedQuote[]>();
const listenersByThread = new Map<string, Set<() => void>>();
// quoteId -> dispatch contextId. Reserved quotes stay in the collection but
// cannot be removed or included in another submission.
const reservationsByThread = new Map<string, Map<string, string>>();
const EMPTY_QUOTES: readonly CollectedQuote[] = Object.freeze([]);
const EMPTY_RESERVED_IDS: ReadonlySet<string> = new Set();

function createQuoteId(): string {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return cryptoApi.randomUUID();
  }
  return `quote-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2)}`;
}

/** A per-Thread fallback key, so unsaved conversations never share a bucket. */
export function createEphemeralQuoteThreadId(): string {
  return `unsaved-quote-${createQuoteId()}`;
}

function notify(threadId: string): void {
  const listeners = listenersByThread.get(threadId);
  if (listeners === undefined) {
    return;
  }
  for (const listener of [...listeners]) {
    listener();
  }
}

function usableThreadId(
  threadId: string | null | undefined,
): threadId is string {
  return typeof threadId === "string" && threadId.length > 0;
}

/**
 * Keeps one collection key for a Thread across the unsaved-to-saved
 * transition. assistant-ui keeps the local thread item id stable when a new
 * thread is initialized, while the active thread id switches to the remote
 * backend id. Preferring the runtime id avoids moving or losing quotes during
 * that switch. The fallback key is per Thread, never the shared empty string.
 */
export function resolveQuoteCollectionKey(
  runtimeThreadId: string | null | undefined,
  activeThreadId: string | null | undefined,
  ephemeralThreadId: string,
): string {
  if (usableThreadId(runtimeThreadId)) {
    return runtimeThreadId;
  }
  if (usableThreadId(activeThreadId)) {
    return activeThreadId;
  }
  return usableThreadId(ephemeralThreadId)
    ? ephemeralThreadId
    : createEphemeralQuoteThreadId();
}

export function getQuotes(threadId: string): readonly CollectedQuote[] {
  return quotesByThread.get(threadId) ?? EMPTY_QUOTES;
}

export function getReservedQuoteIds(threadId: string): ReadonlySet<string> {
  const reservations = reservationsByThread.get(threadId);
  return reservations === undefined
    ? EMPTY_RESERVED_IDS
    : new Set(reservations.keys());
}

export function getUnreservedQuotes(
  threadId: string,
): readonly CollectedQuote[] {
  const reservations = reservationsByThread.get(threadId);
  if (reservations === undefined || reservations.size === 0) {
    return getQuotes(threadId);
  }
  const quotes = getQuotes(threadId);
  const unreserved = quotes.filter((quote) => !reservations.has(quote.id));
  return unreserved.length === quotes.length ? quotes : unreserved;
}

/** Reserves all IDs atomically, or changes nothing. */
export function reserveQuoteIds(
  threadId: string,
  contextId: string,
  quoteIds: readonly string[],
): boolean {
  if (contextId.length === 0 || quoteIds.length === 0) {
    return false;
  }
  const quoteIdsInCollection = new Set(
    getQuotes(threadId).map((quote) => quote.id),
  );
  const reservations = new Map(reservationsByThread.get(threadId) ?? []);
  const seen = new Set<string>();
  for (const quoteId of quoteIds) {
    if (
      quoteId.length === 0 ||
      seen.has(quoteId) ||
      !quoteIdsInCollection.has(quoteId) ||
      reservations.has(quoteId)
    ) {
      return false;
    }
    seen.add(quoteId);
  }
  for (const quoteId of seen) {
    reservations.set(quoteId, contextId);
  }
  reservationsByThread.set(threadId, reservations);
  notify(threadId);
  return true;
}

/** Releases one dispatch's claim; the quotes themselves stay. */
export function releaseQuoteIds(threadId: string, contextId: string): boolean {
  const reservations = reservationsByThread.get(threadId);
  if (reservations === undefined) {
    return false;
  }
  let changed = false;
  for (const [quoteId, owner] of reservations) {
    if (owner === contextId) {
      reservations.delete(quoteId);
      changed = true;
    }
  }
  if (!changed) {
    return false;
  }
  if (reservations.size === 0) {
    reservationsByThread.delete(threadId);
  }
  notify(threadId);
  return true;
}

/** Removes exactly the quotes reserved by one accepted dispatch. */
export function settleQuoteIds(threadId: string, contextId: string): boolean {
  const reservations = reservationsByThread.get(threadId);
  if (reservations === undefined) {
    return false;
  }
  const acceptedIds = new Set<string>();
  for (const [quoteId, owner] of reservations) {
    if (owner === contextId) {
      acceptedIds.add(quoteId);
      reservations.delete(quoteId);
    }
  }
  if (acceptedIds.size === 0) {
    return false;
  }
  if (reservations.size === 0) {
    reservationsByThread.delete(threadId);
  }
  const remaining = getQuotes(threadId).filter(
    (quote) => !acceptedIds.has(quote.id),
  );
  if (remaining.length === 0) {
    quotesByThread.delete(threadId);
  } else {
    quotesByThread.set(threadId, remaining);
  }
  notify(threadId);
  return true;
}

export function subscribeQuotes(
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

/** Appends a quote; array order is the collection order. */
export function addQuote(
  threadId: string,
  anchor: QuoteAnchor,
  options: AddQuoteOptions = {},
): CollectedQuote {
  const quote: CollectedQuote = {
    id: options.id ?? createQuoteId(),
    anchor,
    comment: normalizeQuoteComment(options.comment ?? ""),
    createdAt: options.createdAt ?? Date.now(),
  };
  const next = [...getQuotes(threadId), quote];
  quotesByThread.set(threadId, next);
  notify(threadId);
  return quote;
}

/** Removes one card without reordering the rest. */
export function removeQuote(threadId: string, quoteId: string): boolean {
  if (reservationsByThread.get(threadId)?.has(quoteId) === true) {
    return false;
  }
  const current = quotesByThread.get(threadId);
  if (current === undefined) {
    return false;
  }
  const next = current.filter((quote) => quote.id !== quoteId);
  if (next.length === current.length) {
    return false;
  }
  if (next.length === 0) {
    quotesByThread.delete(threadId);
  } else {
    quotesByThread.set(threadId, next);
  }
  notify(threadId);
  return true;
}

export function clearQuotes(threadId: string): boolean {
  const hadQuotes = quotesByThread.delete(threadId);
  const hadReservations = reservationsByThread.delete(threadId);
  if (!(hadQuotes || hadReservations)) {
    return false;
  }
  notify(threadId);
  return true;
}
