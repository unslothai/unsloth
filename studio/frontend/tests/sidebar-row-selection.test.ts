// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  rangeBetween,
  toggleSelected,
} from "../src/features/chat/utils/row-selection.ts";
import { usePinnedChatsStore } from "../src/features/chat/stores/pinned-chats-store.ts";

const rows = ["a", "b", "c", "d"];

test("shift click takes the block between anchor and row, either way round", () => {
  assert.deepEqual(rangeBetween(rows, "b", "d"), ["b", "c", "d"]);
  // Dragging the selection upward covers the same rows.
  assert.deepEqual(rangeBetween(rows, "d", "b"), ["b", "c", "d"]);
  assert.deepEqual(rangeBetween(rows, "c", "c"), ["c"]);
});

test("a lost anchor selects only the clicked row", () => {
  // The anchored chat was deleted or moved to another list; selecting a block
  // from nothing would grab rows the user never pointed at.
  assert.deepEqual(rangeBetween(rows, "gone", "c"), ["c"]);
  assert.deepEqual(rangeBetween(rows, "a", "gone"), []);
});

test("cmd click adds a row, and takes it back out", () => {
  const once = toggleSelected(new Set(["a"]), "b");
  assert.deepEqual([...once].sort(), ["a", "b"]);
  const twice = toggleSelected(once, "b");
  assert.deepEqual([...twice], ["a"]);
  // The input set is never mutated, so React sees a new reference.
  assert.deepEqual([...once].sort(), ["a", "b"]);
});

test("pinning a selection leads with the new pins and keeps the rest", () => {
  const store = usePinnedChatsStore.getState();
  store.setPinned([], false);
  usePinnedChatsStore.setState({ pinnedIds: ["old"] });

  usePinnedChatsStore.getState().setPinned(["a", "b"], true);
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["a", "b", "old"]);

  // Pinning again must not duplicate, and must not reshuffle.
  usePinnedChatsStore.getState().setPinned(["a", "old"], true);
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["a", "b", "old"]);

  usePinnedChatsStore.getState().setPinned(["a", "old"], false);
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["b"]);
});
