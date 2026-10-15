// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One fork-count fetch per thread, shared by every badge, instead of one per message. */
import {
  CHAT_HISTORY_UPDATED_EVENT,
  getThreadForkCounts,
} from "../api/chat-api";
import { isThreadIncognito } from "./chat-history-storage";

export const FORK_COUNT_REFRESH_DEBOUNCE_MS = 300;

// Max-wait ceiling: streaming fires the history event per chunk, starving a pure debounce.
export const FORK_COUNT_REFRESH_MAX_WAIT_MS = 2000;

type Counts = ReadonlyMap<string, number>;

const EMPTY_COUNTS: Counts = new Map();

type Entry = {
  counts: Counts;
  subscribers: Set<() => void>;
  seq: number;
};

const entries = new Map<string, Entry>();
let pendingRefresh: ReturnType<typeof setTimeout> | null = null;
// Not restarted by later events, which makes it a ceiling rather than a second debounce.
let maxWaitTimer: ReturnType<typeof setTimeout> | null = null;
let listening = false;

async function refresh(threadId: string): Promise<void> {
  const entry = entries.get(threadId);
  if (!entry) return;
  // Temporary chats cannot have forks; the row decides that, not the `__LOCALID_` prefix.
  if (isThreadIncognito(threadId)) return;
  const seq = ++entry.seq;
  let counts: Counts;
  try {
    counts = await getThreadForkCounts(threadId);
  } catch {
    return;
  }
  if (entries.get(threadId) !== entry || entry.seq !== seq) return;
  entry.counts = counts;
  for (const notify of [...entry.subscribers]) notify();
}

function cancelTimers(): void {
  if (pendingRefresh) {
    clearTimeout(pendingRefresh);
    pendingRefresh = null;
  }
  if (maxWaitTimer) {
    clearTimeout(maxWaitTimer);
    maxWaitTimer = null;
  }
}

function runRefresh(): void {
  cancelTimers();
  for (const threadId of entries.keys()) void refresh(threadId);
}

function onHistoryUpdated(): void {
  // Clear and reschedule: a leading-edge throttle would refetch every window during streaming.
  if (pendingRefresh) clearTimeout(pendingRefresh);
  pendingRefresh = setTimeout(runRefresh, FORK_COUNT_REFRESH_DEBOUNCE_MS);
  if (!maxWaitTimer) {
    maxWaitTimer = setTimeout(runRefresh, FORK_COUNT_REFRESH_MAX_WAIT_MS);
  }
}

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
      cancelTimers();
    }
  };
}

export function forkCountFor(threadId: string, messageId: string): number {
  return entries.get(threadId)?.counts.get(messageId) ?? 0;
}
