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

/** A fresh module instance plus the request log its fetches write to. */
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

test("badges churning under the hover autohide do not refetch the thread", async () => {
  const { store, requests } = freshStore({ m7: 3 });
  // The thread subscribes for as long as it is on screen. Its badges live inside action bars
  // that autohide, so they come and go with the pointer and at rest at most one is left.
  const thread = store.subscribeForkCounts("thread-a", () => {});
  await flush();
  assert.deepEqual(requests, ["thread-a"]);

  for (let i = 0; i < 10; i++) {
    const badge = store.subscribeForkCounts("thread-a", () => {});
    // Already there on the first render, so the badge does not arrive a round trip after the
    // bar it sits in and shift the buttons beside it.
    assert.equal(store.forkCountFor("thread-a", "m7"), 3);
    badge();
  }
  await flush();
  assert.equal(
    requests.length,
    1,
    "hovering ten messages refetched the whole thread",
  );

  // Closing the thread is what drops the counts.
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
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(thread, /getForkCount\(/);
  assert.doesNotMatch(thread, /addEventListener\(CHAT_HISTORY_UPDATED_EVENT/);
  assert.match(thread, /subscribeForkCounts\(remoteId, onChange\)/);
  // The badges all unmount at rest, so the thread has to hold the subscription itself.
  assert.match(
    thread,
    /const useThreadForkCounts[\s\S]*?subscribeForkCounts\(remoteId,/,
  );
  assert.match(thread, /^ {2}useThreadForkCounts\(\);$/m);
});

test("a continuous stream costs one refresh per ceiling, not one per debounce window", async (t) => {
  // The burst case above fires every event inside a single window, where a leading-edge
  // throttle and a trailing-edge debounce are indistinguishable. Streaming is the case that
  // separates them: the event arrives per chunk, so chunks keep landing after the window has
  // already expired. A throttle re-arms and fetches again, once per debounce window, for as long
  // as the reply runs.
  //
  // This used to assert ZERO mid-stream fetches. That was the #8992 review finding: a pure
  // trailing edge never expires under a stream, so a fork deleted from the sidebar kept a stale
  // badge until the unrelated reply finished. FORK_COUNT_REFRESH_MAX_WAIT_MS bounds that, and
  // the price of the bound is what this test now measures, so it is a number in the suite rather
  // than a claim in a comment.
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

// #8992 review: a trailing edge alone starves. CHAT_HISTORY_UPDATED_EVENT fires once per
// streaming chunk, so a reply running in a background thread resets the debounce forever. A fork
// deleted from the sidebar changes the badge on the thread the user is looking at, and without a
// ceiling that badge stays wrong until the unrelated stream goes quiet, which on a long or queued
// generation is minutes.

test("a chunk every debounce window cannot postpone a refresh forever", async (t) => {
  mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => mock.timers.reset());
  const { store, requests } = freshStore();
  const unsubscribe = store.subscribeForkCounts("thread-a", () => {});
  await flush();
  assert.equal(requests.length, 1, "the initial fetch");

  // A stream chunking just inside the debounce window, which is what resets the trailing edge.
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

  // The ordinary case must still collapse to ONE refresh on the trailing edge. If the max-wait
  // timer were left running after the debounce fired, this would fetch a second time later.
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

  // Unsubscribing with the ceiling armed. Asserting "no fetch happens" here would prove nothing:
  // the entries map is empty by then, so a leaked timer fires runRefresh over nothing and the
  // request log looks identical either way. The leak is only observable once a LATER thread is
  // on screen for the stale timer to refresh, which is also the case that actually costs a user
  // a request: open a thread, close it, open another inside the ceiling window.
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
  // A `__LOCALID_` id is the permanent primary key of every chat Studio creates, so skipping
  // the fetch on that prefix left their badges reading 0 for good.
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
  // An incognito thread has no row, so its forks cannot exist. The saved chat beside it still
  // asks: the guard is the row, not the `__LOCALID_` prefix both of them carry.
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
