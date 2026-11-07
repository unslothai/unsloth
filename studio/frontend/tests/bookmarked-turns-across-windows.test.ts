// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";

const KEY = "unsloth_bookmarked_turns";
const data = new Map<string, string>();
const listeners: ((event: { key: string | null }) => void)[] = [];
const localStorage = {
  getItem: (key: string) => data.get(key) ?? null,
  setItem: (key: string, value: string) => void data.set(key, value),
  removeItem: (key: string) => void data.delete(key),
};
Object.assign(globalThis, {
  localStorage,
  window: {
    localStorage,
    addEventListener: (type: string, listener: (event: { key: string | null }) => void) => {
      if (type === "storage") listeners.push(listener);
    },
  },
});

const { useBookmarkedTurnsStore } = await import(
  "../src/features/chat/stores/bookmarked-turns-store.ts"
);

function otherWindowWrites(bookmarkedByThread: Record<string, string[]>) {
  data.set(KEY, JSON.stringify({ state: { bookmarkedByThread }, version: 0 }));
  for (const listener of listeners) listener({ key: KEY });
}

test("a bookmark made in another window survives a bookmark made in this one", () => {
  otherWindowWrites({ "chat-a": ["m1"] });
  useBookmarkedTurnsStore.getState().toggleBookmarkedTurn("chat-b", "m2");
  assert.deepEqual(JSON.parse(data.get(KEY) ?? "{}").state.bookmarkedByThread, {
    "chat-a": ["m1"],
    "chat-b": ["m2"],
  });
});

test("clearing storage in another window drops this window's bookmarks", () => {
  data.clear();
  for (const listener of listeners) listener({ key: null });
  assert.deepEqual(useBookmarkedTurnsStore.getState().bookmarkedByThread, {});
});
