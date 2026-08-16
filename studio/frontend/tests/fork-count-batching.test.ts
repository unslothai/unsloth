// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #8977: the chat got slower as the thread filled. The fork badge owned a
// CHAT_HISTORY_UPDATED_EVENT listener and a request per message, so a delete on a
// 200-message thread spent 200 requests before anything could repaint, and streaming
// raises that event once per chunk. One shared subscription, one request per thread.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test, { mock } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

const CHAT_HISTORY_UPDATED_EVENT = "unsloth-chat-history-updated";

type Store = {
  FORK_COUNT_REFRESH_DEBOUNCE_MS: number;
  subscribeForkCounts: (threadId: string, onChange: () => void) => () => void;
  forkCountFor: (threadId: string, messageId: string) => number;
};

const listeners = new Map<string, Set<() => void>>();
Object.assign(globalThis, {
  window: {
    addEventListener(type: string, fn: () => void) {
      const set = listeners.get(type) ?? new Set<() => void>();
      set.add(fn);
      listeners.set(type, set);
    },
    removeEventListener(type: string, fn: () => void) {
      listeners.get(type)?.delete(fn);
    },
  },
});

function historyListenerCount(): number {
  return listeners.get(CHAT_HISTORY_UPDATED_EVENT)?.size ?? 0;
}

function fireHistoryUpdated(): void {
  for (const fn of [...(listeners.get(CHAT_HISTORY_UPDATED_EVENT) ?? [])]) fn();
}

/** A fresh module instance plus the request log its fetches write to. */
function freshStore(counts: Record<string, number> = {}): {
  store: Store;
  requests: string[];
} {
  const requests: string[] = [];
  const store = loadWithStubs<Store>(
    new URL("../src/features/chat/utils/fork-count-store.ts", import.meta.url),
    {
      "../api/chat-api": {
        CHAT_HISTORY_UPDATED_EVENT,
        getThreadForkCounts: async (threadId: string) => {
          requests.push(threadId);
          return new Map(Object.entries(counts));
        },
      },
    },
  );
  return { store, requests };
}

/** Let the awaited fetch inside the store settle. */
async function flush(): Promise<void> {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

test("a 200-message thread costs one request, not one per message", async () => {
  const { store, requests } = freshStore({ m7: 3 });
  const seen = new Array<number>(200).fill(0);
  const unsubscribes = seen.map((_, i) =>
    store.subscribeForkCounts("thread-a", () => {
      seen[i] = store.forkCountFor("thread-a", "m7");
    }),
  );
  await flush();

  assert.deepEqual(requests, ["thread-a"]);
  assert.equal(historyListenerCount(), 1);
  // Every badge sees the count the single request brought back.
  assert.equal(seen.every((count) => count === 3), true);
  assert.equal(store.forkCountFor("thread-a", "m7"), 3);
  assert.equal(store.forkCountFor("thread-a", "m8"), 0);

  for (const unsubscribe of unsubscribes) unsubscribe();
  assert.equal(historyListenerCount(), 0);
});

test("a burst of history events collapses into one refresh", async (t) => {
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore();
  const unsubscribes = Array.from({ length: 200 }, () =>
    store.subscribeForkCounts("thread-a", () => {}),
  );
  await flush();
  assert.equal(requests.length, 1);

  // What one delete looks like: the sync PUT and every follow-up notify.
  for (let i = 0; i < 20; i++) fireHistoryUpdated();
  assert.equal(requests.length, 1, "nothing fires before the debounce window");
  mock.timers.tick(store.FORK_COUNT_REFRESH_DEBOUNCE_MS);
  await flush();
  assert.equal(requests.length, 2);

  fireHistoryUpdated();
  mock.timers.tick(store.FORK_COUNT_REFRESH_DEBOUNCE_MS);
  await flush();
  assert.equal(requests.length, 3, "a later window still refreshes");

  for (const unsubscribe of unsubscribes) unsubscribe();
  fireHistoryUpdated();
  mock.timers.tick(store.FORK_COUNT_REFRESH_DEBOUNCE_MS);
  await flush();
  assert.equal(requests.length, 3, "an unmounted thread stops fetching");
});

test("two threads on screen cost one request each", async () => {
  const { store, requests } = freshStore();
  const a = store.subscribeForkCounts("thread-a", () => {});
  const b = store.subscribeForkCounts("thread-b", () => {});
  await flush();
  assert.deepEqual(requests, ["thread-a", "thread-b"]);
  a();
  b();
});

test("the badge no longer owns a listener or a per-message request", () => {
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(thread, /getForkCount\(/);
  assert.doesNotMatch(thread, /addEventListener\(CHAT_HISTORY_UPDATED_EVENT/);
  assert.match(thread, /subscribeForkCounts\(remoteId, onChange\)/);
});
