// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The wiring, as opposed to the leaf helpers: chatSearchIndexHasRows has to answer for a
// history not yet indexed, which is the first open of every page load.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./chat-search-index-resolver.mjs", import.meta.url);
const { store } = installLocalStorageFake();

// The shared fake's addEventListener is a no-op, so the module-level listeners would register
// into nothing. Swapped before the import below registers them. dispatchEvent routes by type,
// since the storage listener re-raises the history event through it.
type Listener = (event: {
  key?: string | null;
  newValue?: string | null;
  type?: string;
}) => void;
const listeners = new Map<string, Set<Listener>>();
Object.assign(globalThis.window as object, {
  addEventListener: (type: string, fn: Listener) => {
    const forType = listeners.get(type) ?? new Set<Listener>();
    forType.add(fn);
    listeners.set(type, forType);
  },
  removeEventListener: (type: string, fn: Listener) => {
    listeners.get(type)?.delete(fn);
  },
  dispatchEvent: (event: { type: string }) => {
    for (const fn of listeners.get(event.type) ?? []) fn(event);
    return true;
  },
});
const fire = (
  type: string,
  event: { key?: string | null; newValue?: string | null } = {},
) => {
  for (const fn of listeners.get(type) ?? []) fn({ type, ...event });
};
const fireStorage = (key: string | null, newValue: string | null = "value") =>
  fire("storage", { key, newValue });

const {
  buildChatSearchIndex,
  chatSearchIndexHasRows,
  publishChatSearchBuild,
  shouldPostponeSearchRebuild,
  writeCachedIndex,
} = await import("../src/features/chat/hooks/use-chat-search-index.ts");
const { CHAT_HISTORY_REVISION_KEY, CHAT_HISTORY_UPDATED_EVENT } = await import(
  "./helpers/store-stubs/chat-search-history.ts"
);
const { configureChatSearchHistoryStub } = await import(
  "./helpers/store-stubs/chat-search-history.ts"
);
const { rememberChatSearchHasRows } = await import(
  "../src/features/chat/utils/chat-search-history-hint.ts"
);
const {
  AUTH_SESSION_CLEARED_EVENT,
  AUTH_SESSION_MARK_KEY,
  AUTH_TOKEN_KEY,
  setAuthSessionEpochForTest,
} = await import("./helpers/store-stubs/chat-search-auth.ts");

const row = {
  type: "single" as const,
  id: "t1",
  title: "Acme roadmap",
  userSearchText: "acme roadmap",
  searchText: "acme roadmap",
  createdAt: 1,
};

test("an unbuilt index falls back to the last completed build's hint", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex(null);
  // Nothing ever built, so the history is unknown rather than known-empty.
  assert.equal(chatSearchIndexHasRows(), null);

  // A completed build with rows, then the page reloads: the module cache is gone.
  writeCachedIndex([row]);
  assert.equal(chatSearchIndexHasRows(), true);
  writeCachedIndex(null);
  assert.equal(
    chatSearchIndexHasRows(),
    true,
    "150 stored chats must not read as an empty history on the next page load",
  );
});

test("an invalidated cache keeps a rows answer and drops an empty one", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);

  // Invalidation says stale, not empty.
  writeCachedIndex(null);
  assert.equal(chatSearchIndexHasRows(), true);

  // A completed build that found nothing is the one thing that means "no chats".
  writeCachedIndex([]);
  assert.equal(chatSearchIndexHasRows(), false);
  // ...but the next change may be that history's first chat, so it goes back to unknown.
  writeCachedIndex(null);
  assert.equal(chatSearchIndexHasRows(), null);
});

test("a session change inside one page load drops the previous account's hint", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);
  assert.equal(chatSearchIndexHasRows(), true);

  setAuthSessionEpochForTest(1);
  assert.equal(
    chatSearchIndexHasRows(),
    null,
    "the next account must not be sized by the previous one's history",
  );
});

test("an unbuilt index with no hint reads as unknown, not as empty", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex(null);
  // A profile that has never built the index, and one upgrading from before the hint
  // existed, both land here. Reading this as "empty" is what sized a populated dialog
  // compact and then grew it mid-open.
  assert.equal(chatSearchIndexHasRows(), null);
});

test("a completed empty build is remembered as empty, not as unknown", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([]);
  writeCachedIndex(null);
  assert.equal(
    chatSearchIndexHasRows(),
    null,
    "a history that just changed is unknown again, not still empty",
  );
  writeCachedIndex([]);
  assert.equal(chatSearchIndexHasRows(), false);
});

test("another tab's history change drops the cached rows", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);
  // Make the hint disagree with the cache so the two are distinguishable: cached rows answer
  // first, and only once dropped does the hint show through.
  rememberChatSearchHasRows(false);
  assert.equal(chatSearchIndexHasRows(), true);

  fireStorage(CHAT_HISTORY_REVISION_KEY);
  // Rows gone, so the next open rebuilds instead of offering a deleted chat, and the stale
  // hint goes with them. Without the listener this still reads true.
  assert.equal(chatSearchIndexHasRows(), null);
});

test("another tab's history change reaches an open dialog's rebuild", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);

  // What an open dialog subscribes to: dropping the cache alone leaves it on pre-change rows
  // with nothing scheduled.
  let rebuilds = 0;
  const onHistory = () => {
    rebuilds += 1;
  };
  (globalThis.window as Window).addEventListener(
    CHAT_HISTORY_UPDATED_EVENT,
    onHistory,
  );
  try {
    fireStorage(CHAT_HISTORY_REVISION_KEY);
    assert.equal(
      rebuilds,
      1,
      "the cross-tab change has to re-raise as a local one",
    );
  } finally {
    (globalThis.window as Window).removeEventListener(
      CHAT_HISTORY_UPDATED_EVENT,
      onHistory,
    );
  }
});

test("another tab's account switch drops this tab's rows and hint", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);
  assert.equal(chatSearchIndexHasRows(), true);

  // The epoch and its events are both this document's, so a sign-in elsewhere arrives as a
  // storage write alone and the epoch still matches: hence the cache would still answer.
  let rebuilds = 0;
  const onHistory = () => {
    rebuilds += 1;
  };
  // Mirrors the module-private event name. An open dialog listens for it to retire a build
  // in flight for the previous account, which the epoch check cannot do.
  let sessionChanges = 0;
  const onSession = () => {
    sessionChanges += 1;
  };
  const win = globalThis.window as Window;
  win.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistory);
  win.addEventListener("unsloth-chat-search-session-changed", onSession);
  try {
    fireStorage(AUTH_SESSION_MARK_KEY);
    assert.equal(
      chatSearchIndexHasRows(),
      null,
      "the previous account's titles must not survive the switch",
    );
    assert.equal(rebuilds, 1, "everything showing that account has to refresh");
    assert.equal(
      sessionChanges,
      1,
      "a build in flight for the previous account has to be retired",
    );
  } finally {
    win.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistory);
    win.removeEventListener("unsloth-chat-search-session-changed", onSession);
  }
});

test("a token refresh in another tab costs nothing", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);

  // refreshSession rewrites the token hourly with the session unchanged. Reading that as a
  // switch would throw away a warm cache for nothing.
  let sessionChanges = 0;
  const onSession = () => {
    sessionChanges += 1;
  };
  const win = globalThis.window as Window;
  win.addEventListener("unsloth-chat-search-session-changed", onSession);
  try {
    fireStorage(AUTH_TOKEN_KEY);
    assert.equal(
      chatSearchIndexHasRows(),
      true,
      "the cache survives a rotation",
    );
    assert.equal(sessionChanges, 0);
  } finally {
    win.removeEventListener("unsloth-chat-search-session-changed", onSession);
  }
});

test("a legacy session logout in another tab drops cached rows", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);

  let rebuilds = 0;
  const onHistory = () => {
    rebuilds += 1;
  };
  let sessionChanges = 0;
  const onSession = () => {
    sessionChanges += 1;
  };
  const win = globalThis.window as Window;
  win.addEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistory);
  win.addEventListener("unsloth-chat-search-session-changed", onSession);
  try {
    fireStorage(AUTH_TOKEN_KEY, null);
    assert.equal(
      chatSearchIndexHasRows(),
      null,
      "a session created before the marker existed still clears on logout",
    );
    assert.equal(rebuilds, 1);
    assert.equal(sessionChanges, 1);
  } finally {
    win.removeEventListener(CHAT_HISTORY_UPDATED_EVENT, onHistory);
    win.removeEventListener("unsloth-chat-search-session-changed", onSession);
  }
});

test("a history change in another tab is not treated as a session change", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);

  // A delete elsewhere must not tear rows out from under an open dialog: it schedules a
  // rebuild and lets the current rows stand.
  let sessionChanges = 0;
  const onSession = () => {
    sessionChanges += 1;
  };
  const win = globalThis.window as Window;
  win.addEventListener("unsloth-chat-search-session-changed", onSession);
  try {
    fireStorage(CHAT_HISTORY_REVISION_KEY);
    assert.equal(sessionChanges, 0);
  } finally {
    win.removeEventListener("unsloth-chat-search-session-changed", onSession);
  }
});

test("a logout takes the persisted hint with it", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);
  writeCachedIndex(null);
  assert.equal(chatSearchIndexHasRows(), true);

  // The hint outlives the page, so a reload on the login screen would hand this account's
  // history to whoever signs in next.
  fire(AUTH_SESSION_CLEARED_EVENT);
  assert.equal(
    chatSearchIndexHasRows(),
    null,
    "the next account's first open must not be sized by the previous one",
  );
});

test("an unrelated storage key leaves the cache alone", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);
  fireStorage("unsloth_theme");
  assert.equal(chatSearchIndexHasRows(), true);
});

test("an index too large to hold is rebuilt rather than cached", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  const heavy = Array.from({ length: 40 }, (_, i) => ({
    ...row,
    id: `t${i}`,
    // A tool-heavy thread: 200k characters of conversation text apiece.
    searchText: "x".repeat(200_000),
  }));
  writeCachedIndex(heavy);
  // The hint still answers, so the dialog sizes correctly without holding 8M characters.
  assert.equal(chatSearchIndexHasRows(), true);

  rememberChatSearchHasRows(false);
  assert.equal(
    chatSearchIndexHasRows(),
    false,
    "the rows were not retained, so the hint is what answers",
  );
});

test("an index within the budget is still cached", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);
  rememberChatSearchHasRows(false);
  assert.equal(chatSearchIndexHasRows(), true, "the cached rows still answer");
});

test("stream chunks do not postpone an already scheduled structural rebuild", () => {
  const structural = new Event(CHAT_HISTORY_UPDATED_EVENT);
  const stream = new CustomEvent(CHAT_HISTORY_UPDATED_EVENT, {
    detail: { coalesce: true },
  });
  assert.equal(shouldPostponeSearchRebuild(false, stream), true);
  assert.equal(shouldPostponeSearchRebuild(true, structural), true);
  assert.equal(
    shouldPostponeSearchRebuild(true, stream),
    false,
    "a stream chunk must keep the structural rebuild's original deadline",
  );
  assert.equal(
    shouldPostponeSearchRebuild(false, stream),
    true,
    "once that deadline fires, later chunks return to quiet-window coalescing",
  );
});

test("failed message reads are not persisted as a completed empty history", async () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([]);
  assert.equal(chatSearchIndexHasRows(), false);
  configureChatSearchHistoryStub({
    threads: [
      {
        id: "unreadable-thread",
        title: "History still exists",
        modelType: "text",
        archived: false,
        createdAt: 1,
      },
    ],
    batchFails: true,
    messageReadsFail: true,
  });

  const build = await buildChatSearchIndex();
  assert.deepEqual(build, { items: [], complete: false });
  publishChatSearchBuild(build);
  assert.equal(
    chatSearchIndexHasRows(),
    null,
    "a transient total read failure is unknown, not a known-empty history",
  );
  configureChatSearchHistoryStub({});
});

test("partial builds replace a stale empty hint with a rows answer", async () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([]);
  assert.equal(chatSearchIndexHasRows(), false);
  configureChatSearchHistoryStub({
    threads: [
      {
        id: "readable-thread",
        title: "Visible history",
        modelType: "text",
        archived: false,
        createdAt: 2,
      },
      {
        id: "unreadable-thread",
        title: "Unreadable history",
        modelType: "text",
        archived: false,
        createdAt: 1,
      },
    ],
    messagesByThread: new Map([
      [
        "readable-thread",
        [
          {
            id: "readable-message",
            threadId: "readable-thread",
            role: "user",
            content: [{ type: "text", text: "Recovered row" }],
            createdAt: 2,
          },
        ],
      ],
    ]),
    batchFails: true,
    messageReadFailures: new Set(["unreadable-thread"]),
  });

  const build = await buildChatSearchIndex();
  assert.equal(build.complete, false);
  assert.equal(build.items.length, 1);
  publishChatSearchBuild(build);
  assert.equal(
    chatSearchIndexHasRows(),
    true,
    "visible partial rows disprove the older completed-empty hint",
  );
  configureChatSearchHistoryStub({});
});
