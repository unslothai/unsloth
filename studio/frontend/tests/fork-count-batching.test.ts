// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test, { mock } from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

import { readSrc } from "./helpers/kit.ts";

const CHAT_HISTORY_UPDATED_EVENT = "unsloth-chat-history-updated";

type Store = {
  FORK_COUNT_REFRESH_DEBOUNCE_MS: number;
  FORK_COUNT_REFRESH_MAX_WAIT_MS: number;
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

function freshStore(
  counts: Record<string, number> = {},
  incognito: readonly string[] = [],
): {
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
      "./chat-history-storage": {
        isThreadIncognito: (threadId: string) => incognito.includes(threadId),
      },
    },
  );
  return { store, requests };
}

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
  assert.equal(
    seen.every((count) => count === 3),
    true,
  );
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

test("badges churning under the hover autohide do not refetch the thread", async () => {
  const { store, requests } = freshStore({ m7: 3 });
  const thread = store.subscribeForkCounts("thread-a", () => {});
  await flush();
  assert.deepEqual(requests, ["thread-a"]);

  for (let i = 0; i < 10; i++) {
    const badge = store.subscribeForkCounts("thread-a", () => {});
    assert.equal(store.forkCountFor("thread-a", "m7"), 3);
    badge();
  }
  await flush();
  assert.equal(
    requests.length,
    1,
    "hovering ten messages refetched the whole thread",
  );

  thread();
  assert.equal(store.forkCountFor("thread-a", "m7"), 0);
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
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.doesNotMatch(thread, /getForkCount\(/);
  assert.doesNotMatch(thread, /addEventListener\(CHAT_HISTORY_UPDATED_EVENT/);
  assert.match(thread, /subscribeForkCounts\(remoteId, onChange\)/);
  assert.match(
    thread,
    /const useThreadForkCounts[\s\S]*?subscribeForkCounts\(remoteId,/,
  );
  assert.match(thread, /^ {2}useThreadForkCounts\(\);$/m);
});

test("a continuous stream costs one refresh per ceiling, not one per debounce window", async (t) => {
  // Streaming fires the event per chunk, so this measures the max-wait refresh cost.
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore();
  const unsubscribe = store.subscribeForkCounts("thread-a", () => {});
  await flush();
  assert.equal(requests.length, 1, "the initial subscribe fetches once");

  const gap = Math.floor(store.FORK_COUNT_REFRESH_DEBOUNCE_MS / 2);
  const chunks = 20;
  for (let chunk = 0; chunk < chunks; chunk++) {
    fireHistoryUpdated();
    mock.timers.tick(gap);
    await flush();
  }
  const streamMs = chunks * gap;
  const midStream = requests.length - 1;
  const throttleWould = Math.floor(
    streamMs / store.FORK_COUNT_REFRESH_DEBOUNCE_MS,
  );
  const ceilingAllows = Math.floor(
    streamMs / store.FORK_COUNT_REFRESH_MAX_WAIT_MS,
  );
  assert.equal(
    midStream,
    ceilingAllows,
    `a ${chunks} chunk stream over ${streamMs}ms refetched ${midStream} time(s) mid-stream; ` +
      `the ceiling allows ${ceilingAllows}`,
  );
  assert.ok(
    midStream < throttleWould,
    `the per-chunk traffic is back: ${midStream} fetches against the ${throttleWould} a ` +
      "leading-edge throttle would have cost",
  );

  mock.timers.tick(store.FORK_COUNT_REFRESH_DEBOUNCE_MS);
  await flush();
  assert.equal(
    requests.length,
    2 + ceilingAllows,
    "the quiet window after the stream refreshes exactly once",
  );

  unsubscribe();
});

// A trailing-edge debounce alone starves under a per-chunk event stream.

test("a chunk every debounce window cannot postpone a refresh forever", async (t) => {
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore();
  const unsubscribe = store.subscribeForkCounts("thread-a", () => {});
  await flush();
  assert.equal(requests.length, 1, "the initial fetch");

  const step = store.FORK_COUNT_REFRESH_DEBOUNCE_MS - 1;
  const ticks = Math.ceil(store.FORK_COUNT_REFRESH_MAX_WAIT_MS / step) + 1;
  let firedAfter: number | null = null;
  for (let i = 0; i < ticks; i++) {
    fireHistoryUpdated();
    mock.timers.tick(step);
    await flush();
    if (requests.length > 1) {
      firedAfter = (i + 1) * step;
      break;
    }
  }
  assert.notEqual(
    firedAfter,
    null,
    "the refresh never happened: a chunk per window postponed it indefinitely",
  );
  assert.ok(
    (firedAfter as number) <= store.FORK_COUNT_REFRESH_MAX_WAIT_MS + step,
    `the refresh waited ${firedAfter}ms, past the ${store.FORK_COUNT_REFRESH_MAX_WAIT_MS}ms bound`,
  );
  unsubscribe();
});

test("the ceiling does not fire while the burst is still inside the debounce window", async (t) => {
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore();
  const unsubscribe = store.subscribeForkCounts("thread-a", () => {});
  await flush();
  assert.equal(requests.length, 1);

  for (let i = 0; i < 20; i++) fireHistoryUpdated();
  mock.timers.tick(store.FORK_COUNT_REFRESH_DEBOUNCE_MS);
  await flush();
  assert.equal(requests.length, 2, "the trailing edge refreshes once");
  mock.timers.tick(store.FORK_COUNT_REFRESH_MAX_WAIT_MS * 2);
  await flush();
  assert.equal(
    requests.length,
    2,
    "the max-wait timer fired again after the debounce had already refreshed",
  );
  unsubscribe();
});

test("the ceiling restarts for the next burst rather than firing once per lifetime", async (t) => {
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore();
  const unsubscribe = store.subscribeForkCounts("thread-a", () => {});
  await flush();

  for (let round = 0; round < 2; round++) {
    const step = store.FORK_COUNT_REFRESH_DEBOUNCE_MS - 1;
    const before = requests.length;
    const ticks = Math.ceil(store.FORK_COUNT_REFRESH_MAX_WAIT_MS / step) + 1;
    for (let i = 0; i < ticks && requests.length === before; i++) {
      fireHistoryUpdated();
      mock.timers.tick(step);
      await flush();
    }
    assert.equal(
      requests.length,
      before + 1,
      `round ${round + 1}: the bound did not apply, so it is a one-shot rather than a ceiling`,
    );
  }
  unsubscribe();
});

test("unsubscribing cancels the ceiling as well as the trailing edge", async (t) => {
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore();
  const first = store.subscribeForkCounts("thread-a", () => {});
  await flush();
  assert.equal(requests.length, 1);

  // A leaked timer is only observable once a later thread is subscribed.
  fireHistoryUpdated();
  first();

  const second = store.subscribeForkCounts("thread-b", () => {});
  await flush();
  assert.deepEqual(
    requests,
    ["thread-a", "thread-b"],
    "the new thread fetches once on subscribe",
  );

  mock.timers.tick(store.FORK_COUNT_REFRESH_MAX_WAIT_MS * 2);
  await flush();
  assert.deepEqual(
    requests,
    ["thread-a", "thread-b"],
    "a ceiling armed by the previous thread outlived it and refetched the new one",
  );
  second();
});

test("a chat created in the app asks for its fork counts", async (t) => {
  // `__LOCALID_` ids are permanent primary keys, so the prefix must not skip the fetch.
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore({ m1: 2 });

  const unsubscribe = store.subscribeForkCounts("__LOCALID_abc123", () => {});
  await flush();
  assert.deepEqual(requests, ["__LOCALID_abc123"]);
  assert.equal(store.forkCountFor("__LOCALID_abc123", "m1"), 2);

  unsubscribe();
});

test("a temporary chat is the one thread that never asks", async (t) => {
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore({ m1: 2 }, ["__LOCALID_temp"]);

  const stop = [
    store.subscribeForkCounts("__LOCALID_temp", () => {}),
    store.subscribeForkCounts("__LOCALID_saved", () => {}),
  ];
  await flush();
  assert.deepEqual(requests, ["__LOCALID_saved"]);

  for (let i = 0; i < 20; i++) fireHistoryUpdated();
  mock.timers.tick(store.FORK_COUNT_REFRESH_MAX_WAIT_MS);
  await flush();
  assert.equal(
    requests.filter((id) => id === "__LOCALID_temp").length,
    0,
    "nor must a burst of history events reach a temporary chat",
  );
  assert.equal(store.forkCountFor("__LOCALID_temp", "m1"), 0);
  assert.equal(store.forkCountFor("__LOCALID_saved", "m1"), 2);

  for (const unsubscribe of stop) unsubscribe();
});

test("every subscribed thread refreshes, whatever its id looks like", async (t) => {
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore({ m1: 2 });

  const stop = [
    store.subscribeForkCounts("__LOCALID_abc123", () => {}),
    store.subscribeForkCounts("thread-saved", () => {}),
  ];
  await flush();
  assert.deepEqual(requests.slice().sort(), [
    "__LOCALID_abc123",
    "thread-saved",
  ]);

  requests.length = 0;
  fireHistoryUpdated();
  mock.timers.tick(store.FORK_COUNT_REFRESH_DEBOUNCE_MS);
  await flush();
  assert.deepEqual(requests.slice().sort(), [
    "__LOCALID_abc123",
    "thread-saved",
  ]);
  assert.equal(store.forkCountFor("thread-saved", "m1"), 2);
  assert.equal(store.forkCountFor("__LOCALID_abc123", "m1"), 2);

  for (const unsubscribe of stop) unsubscribe();
});
