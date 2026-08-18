// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * One fork-count subscription per rendered thread, shared by every message badge.
 *
 * The badge used to own a `CHAT_HISTORY_UPDATED_EVENT` listener and a request per
 * message, so a 200-message thread spent 200 requests on every history event, and
 * streaming raises one of those per chunk.
 */
import { CHAT_HISTORY_UPDATED_EVENT, getThreadForkCounts } from "../api/chat-api";

// Same reasoning as the sidebar's refresh: each quiet window costs one fetch.
export const FORK_COUNT_REFRESH_DEBOUNCE_MS = 300;

type Counts = ReadonlyMap<string, number>;

const EMPTY_COUNTS: Counts = new Map();

type Entry = {
  counts: Counts;
  subscribers: Set<() => void>;
  /** Discards a response overtaken by a later refresh of the same thread. */
  seq: number;
};

const entries = new Map<string, Entry>();
let pendingRefresh: ReturnType<typeof setTimeout> | null = null;
let listening = false;

async function refresh(threadId: string): Promise<void> {
  const entry = entries.get(threadId);
  if (!entry) return;
  const seq = ++entry.seq;
  let counts: Counts;
  try {
    counts = await getThreadForkCounts(threadId);
  } catch {
    return; // the badge is non-critical
  }
  if (entries.get(threadId) !== entry || entry.seq !== seq) return;
  entry.counts = counts;
  for (const notify of [...entry.subscribers]) notify();
}

function onHistoryUpdated(): void {
  // Clear and reschedule, as the sidebar refresh does. Returning while a timer exists would
  // make this a leading-edge throttle: streaming fires this event per chunk, so the timer
  // would expire mid-stream and the next chunk would start another window, costing one
  // whole-thread fetch every FORK_COUNT_REFRESH_DEBOUNCE_MS for as long as the reply runs.
  // Fork counts cannot change during generation, so every one of those is wasted.
  if (pendingRefresh) clearTimeout(pendingRefresh);
  pendingRefresh = setTimeout(() => {
    pendingRefresh = null;
    for (const threadId of entries.keys()) void refresh(threadId);
  }, FORK_COUNT_REFRESH_DEBOUNCE_MS);
}

/** Register a badge. The first subscriber of a thread pays for the fetch; the rest are free. */
export function subscribeForkCounts(
  threadId: string,
  onChange: () => void,
): () => void {
  let entry = entries.get(threadId);
  if (!entry) {
    entry = { counts: EMPTY_COUNTS, subscribers: new Set(), seq: 0 };
    entries.set(threadId, entry);
    void refresh(threadId);
  }
  const owner = entry;
  owner.subscribers.add(onChange);
  if (!listening && typeof window !== "undefined") {
    window.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
    listening = true;
  }
  return () => {
    owner.subscribers.delete(onChange);
    if (owner.subscribers.size > 0 || entries.get(threadId) !== owner) return;
    entries.delete(threadId);
    if (entries.size === 0 && listening) {
      window.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistoryUpdated);
      listening = false;
      if (pendingRefresh) {
        clearTimeout(pendingRefresh);
        pendingRefresh = null;
      }
    }
  };
}

/** The snapshot a `useSyncExternalStore` badge reads: a number, so identity never churns. */
export function forkCountFor(threadId: string, messageId: string): number {
  return entries.get(threadId)?.counts.get(messageId) ?? 0;
}
