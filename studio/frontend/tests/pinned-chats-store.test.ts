// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  discardUnboundPinEdits,
  setPinnedChatsActions,
  usePinnedChatsStore,
} from "../src/features/chat/stores/pinned-chats-store.ts";

const STORE = "../src/features/chat/stores/pinned-chats-store.ts";

test("pins made before the sync runtime binds are replayed into it, and sign-out drops them", () => {
  const calls: Array<[string[], boolean]> = [];
  usePinnedChatsStore.getState().togglePin("early");
  usePinnedChatsStore.getState().unpin("other");
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["early"]);
  setPinnedChatsActions({ setPinned: (ids, pinned) => calls.push([ids, pinned]) });
  assert.deepEqual(calls, [[["early"], true], [["other"], false]]);

  setPinnedChatsActions(null);
  usePinnedChatsStore.getState().togglePin("signed-out");
  discardUnboundPinEdits();
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, []);
  setPinnedChatsActions({ setPinned: (ids, pinned) => calls.push([ids, pinned]) });
  assert.equal(calls.length, 2);
  setPinnedChatsActions(null);
});

test("a pin made before the runtime binds survives a reload", async () => {
  const values = new Map<string, string>();
  const storage = {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => void values.set(key, value),
    removeItem: (key: string) => void values.delete(key),
  };
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true });
  try {
    const before = await import(`${STORE}?page=1`);
    before.usePinnedChatsStore.getState().togglePin("early");
    const after = await import(`${STORE}?page=2`);
    const calls: Array<[string[], boolean]> = [];
    after.setPinnedChatsActions({
      setPinned: (ids: string[], pinned: boolean) => calls.push([ids, pinned]),
    });
    assert.deepEqual(calls, [[["early"], true]]);
    assert.equal(values.size, 0);
  } finally {
    if (descriptor) Object.defineProperty(globalThis, "localStorage", descriptor);
    else delete (globalThis as { localStorage?: unknown }).localStorage;
  }
});
