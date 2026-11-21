// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./chat-search-index-resolver.mjs", import.meta.url);
const { store } = installLocalStorageFake();

// The shared fake's addEventListener is a no-op, so listeners are swapped in before import.
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
  assert.equal(chatSearchIndexHasRows(), null);

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

  writeCachedIndex(null);
  assert.equal(chatSearchIndexHasRows(), true);

  writeCachedIndex([]);
  assert.equal(chatSearchIndexHasRows(), false);
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
  rememberChatSearchHasRows(false);
  assert.equal(chatSearchIndexHasRows(), true);

  fireStorage(CHAT_HISTORY_REVISION_KEY);
  assert.equal(chatSearchIndexHasRows(), null);
});

test("another tab's history change reaches an open dialog's rebuild", () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  writeCachedIndex([row]);

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
    searchText: "x".repeat(200_000),
  }));
  writeCachedIndex(heavy);
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

test("a chat past the message bound is still found by its title", async () => {
  store.clear();
  setAuthSessionEpochForTest(0);
  const total = 201;
  const threads = Array.from({ length: total }, (_, i) => ({
    id: `thread-${i}`,
    title: i === total - 1 ? "Zanzibar ledger" : `Chat ${i}`,
    modelType: "text",
    archived: false,
    createdAt: total - i,
  }));
  configureChatSearchHistoryStub({
    threads,
    messagesByThread: new Map(
      threads.map((thread, i) => [
        thread.id,
        [
          {
            id: `message-${i}`,
            threadId: thread.id,
            role: "user",
            content: [{ type: "text", text: `body ${i} quokka` }],
            createdAt: thread.createdAt,
          },
        ],
      ]),
    ),
  });

  const build = await buildChatSearchIndex();
  assert.equal(build.complete, true);
  assert.equal(build.items.length, total);
  const oldest = build.items.find((item) => item.id === `thread-${total - 1}`);
  assert.ok(oldest, "the oldest chat must not drop out of the index");
  assert.equal(oldest.userSearchText, "zanzibar ledger");
  assert.equal(
    oldest.searchText,
    "zanzibar ledger",
    "its messages stay unloaded",
  );
  for (const i of [0, total - 2]) {
    assert.ok(
      build.items
        .find((item) => item.id === `thread-${i}`)
        ?.searchText.includes(`body ${i} quokka`),
      `chat ${i}, inside the bound, keeps its message text`,
    );
  }
  configureChatSearchHistoryStub({});
});
