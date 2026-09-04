// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One fork-count subscription per rendered thread, shared by every message badge. The badge used
 *  to own a `CHAT_HISTORY_UPDATED_EVENT` listener and a request per message, so a 200-message
 *  thread spent 200 requests on every history event, and streaming raises one per chunk. */
import {
  CHAT_HISTORY_UPDATED_EVENT,
  getThreadForkCounts,
} from "../api/chat-api";
import { isThreadIncognito } from "./chat-history-storage";

// Same reasoning as the sidebar's refresh: each quiet window costs one fetch.
export const FORK_COUNT_REFRESH_DEBOUNCE_MS = 300;

// The ceiling on how long a real fork change can sit unrendered. A trailing edge alone is a
// starvation hazard: CHAT_HISTORY_UPDATED_EVENT fires once per streaming chunk, so a reply
// running in a background thread resets the timer forever and a deleted fork keeps its old
// badge for minutes. The event is a bare Event with six other consumers, so telling fork
// changes apart from chunks would mean changing a contract well outside this store. 2000
// rather than tighter because the bound costs one whole-thread fetch per window while a
// stream runs; at 300ms that is the per-chunk traffic this store exists to remove.
export const FORK_COUNT_REFRESH_MAX_WAIT_MS = 2000;

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
// Started by the first event of a burst and NOT restarted by the ones after it. That is what
// makes it a ceiling rather than a second debounce.
let maxWaitTimer: ReturnType<typeof setTimeout> | null = null;
let listening = false;

async function refresh(threadId: string): Promise<void> {
  const entry = entries.get(threadId);
  if (!entry) return;
  // A temporary chat is the one thread whose forks cannot exist: ensureThreadRecord marks it and
  // returns without a row. The row decides that, not the id, since a `__LOCALID_` prefix belongs
  // to every chat the app creates. Skipped because the answer is already the empty map the entry
  // holds, not to dodge a failure: fork_counts_for_thread GROUPs without looking the source up,
  // so an unknown thread gets 200 and an empty map.
  if (isThreadIncognito(threadId)) return;
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
  // Both timers race for this; whichever loses must not fire afterwards.
  cancelTimers();
  for (const threadId of entries.keys()) void refresh(threadId);
}

function onHistoryUpdated(): void {
  // Clear and reschedule, as the sidebar refresh does. Returning while a timer exists would make
  // this a leading-edge throttle: streaming fires this event per chunk, so the timer would
  // expire mid-stream and the next chunk would start another window, costing one whole-thread
  // fetch per debounce window for as long as the reply runs. Fork counts cannot change during
  // generation, so every one of those is wasted.
  if (pendingRefresh) clearTimeout(pendingRefresh);
  pendingRefresh = setTimeout(runRefresh, FORK_COUNT_REFRESH_DEBOUNCE_MS);
  // The bound. Deliberately not restarted while it is already running, or a per-chunk event stream
  // would push it out the way it pushes out the trailing edge. A second timer rather than a
  // Date.now() deadline so it shares the debounce's clock and can be tested without a fake Date.
  if (!maxWaitTimer) {
    maxWaitTimer = setTimeout(runRefresh, FORK_COUNT_REFRESH_MAX_WAIT_MS);
  }
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
      cancelTimers();
    }
  };
}

/** The snapshot a `useSyncExternalStore` badge reads: a number, so identity never churns. */
export function forkCountFor(threadId: string, messageId: string): number {
  return entries.get(threadId)?.counts.get(messageId) ?? 0;
}
