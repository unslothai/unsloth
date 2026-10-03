// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { useBookmarkedTurnsStore } from "../src/features/chat/stores/bookmarked-turns-store.ts";

const store = () => useBookmarkedTurnsStore.getState();

test("deleted chats leave no bookmark records behind", () => {
  useBookmarkedTurnsStore.setState({ bookmarkedByThread: {} });
  store().toggleBookmarkedTurn("t1", "u1");
  store().toggleBookmarkedTurn("t1", "u2");
  store().toggleBookmarkedTurn("t2", "u9");
  store().toggleBookmarkedTurn("t3", "u5");
  store().forgetThreads(["t1", "t3", "never-bookmarked"]);
  assert.deepEqual(store().bookmarkedByThread, { t2: ["u9"] });
});

test("forgetting chats with no bookmarks keeps the state object", () => {
  useBookmarkedTurnsStore.setState({ bookmarkedByThread: { t2: ["u9"] } });
  const before = store().bookmarkedByThread;
  store().forgetThreads(["t1"]);
  assert.equal(store().bookmarkedByThread, before);
});

test("clear all chats forgets the cleared chats' bookmarks", async () => {
  // Settings -> Clear all chats skips deleteChatItems, so it forgets them itself.
  const clearAll = await readFile(
    new URL("../src/features/chat/utils/clear-all-chats.ts", import.meta.url),
    "utf8",
  );
  assert.match(clearAll, /forgetThreads\(result\.deletedThreadIds\)/);
});
