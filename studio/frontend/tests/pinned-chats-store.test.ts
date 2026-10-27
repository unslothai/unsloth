// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  setPinnedChatsActions,
  usePinnedChatsStore,
} from "../src/features/chat/stores/pinned-chats-store.ts";

test("pins made before the sync runtime binds are replayed into it, and an unbind drops them", () => {
  const calls: Array<[string[], boolean]> = [];
  usePinnedChatsStore.getState().togglePin("early");
  usePinnedChatsStore.getState().unpin("other");
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["early"]);
  setPinnedChatsActions({ setPinned: (ids, pinned) => calls.push([ids, pinned]) });
  assert.deepEqual(calls, [[["early"], true], [["other"], false]]);

  setPinnedChatsActions(null);
  usePinnedChatsStore.getState().togglePin("signed-out");
  setPinnedChatsActions(null);
  setPinnedChatsActions({ setPinned: (ids, pinned) => calls.push([ids, pinned]) });
  assert.equal(calls.length, 2);
  setPinnedChatsActions(null);
});
