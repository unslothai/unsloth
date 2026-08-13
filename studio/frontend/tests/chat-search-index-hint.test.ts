// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The wiring, as opposed to the two leaf helpers: chatSearchIndexHasRows has to answer for a
// history whose index has not been built yet, which is the first open of every page load.

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

import { installLocalStorageFake } from "./helpers/kit.ts";

register("./chat-search-index-resolver.mjs", import.meta.url);
const { store } = installLocalStorageFake();

// The shared fake's addEventListener is a no-op, so a module-level storage listener would
// register into nothing. Swap in one that keeps its listeners, before the import below
// registers them.
const storageListeners = new Set<(event: { key: string | null }) => void>();
Object.assign(globalThis.window as object, {
  addEventListener: (type: string, fn: (event: { key: string | null }) => void) => {
    if (type === "storage") storageListeners.add(fn);
  },
  removeEventListener: (type: string, fn: (event: { key: string | null }) => void) => {
    if (type === "storage") storageListeners.delete(fn);
  },
});
const fireStorage = (key: string | null) => {
  for (const fn of storageListeners) fn({ key });
};

const { chatSearchIndexHasRows, writeCachedIndex } = await import(
  "../src/features/chat/hooks/use-chat-search-index.ts"
);
const { CHAT_HISTORY_REVISION_KEY } = await import(
  "./helpers/store-stubs/chat-search-history.ts"
);
const { rememberChatSearchHasRows } = await import(
  "../src/features/chat/utils/chat-search-history-hint.ts"
);
const { setAuthSessionEpochForTest } = await import(
  "./helpers/store-stubs/chat-search-auth.ts"
);

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
  // Force the hint to disagree with the cache, so the two are distinguishable: while the
  // rows are cached they answer, and only once they are dropped does the hint show through.
  rememberChatSearchHasRows(false);
  assert.equal(chatSearchIndexHasRows(), true);

  fireStorage(CHAT_HISTORY_REVISION_KEY);
  // Rows gone, so the next open rebuilds instead of offering a chat another tab deleted,
  // and the stale empty hint goes with them. Without the listener this still reads true.
  assert.equal(chatSearchIndexHasRows(), null);
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
