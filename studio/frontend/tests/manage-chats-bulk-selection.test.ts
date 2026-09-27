// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Bulk pin/unpin ordering and shift-range selection, run against the real store.

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const { usePinnedChatsStore } = await import(
  "../src/features/chat/stores/pinned-chats-store.ts"
);
const { rangeBetween } = await import(
  "../src/features/chat/utils/row-selection.ts"
);

function reset(ids: string[] = []): void {
  usePinnedChatsStore.setState({ pinnedIds: ids });
}

test("pinning a selection prepends only the chats that were not pinned", () => {
  reset(["b"]);
  usePinnedChatsStore.getState().setPinned(["a", "b", "c"], true);
  // b keeps its place; a and c lead, as a single pin does.
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["a", "c", "b"]);
});

test("pinning an already pinned selection is a no-op, not a reorder", () => {
  reset(["a", "b"]);
  const before = usePinnedChatsStore.getState().pinnedIds;
  usePinnedChatsStore.getState().setPinned(["a", "b"], true);
  assert.equal(usePinnedChatsStore.getState().pinnedIds, before);
});

test("unpinning a selection drops exactly those ids and keeps the rest ordered", () => {
  reset(["a", "b", "c", "d"]);
  usePinnedChatsStore.getState().setPinned(["b", "d"], false);
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["a", "c"]);
});

test("unpinning ids that were never pinned leaves the list alone", () => {
  reset(["a"]);
  usePinnedChatsStore.getState().setPinned(["x", "y"], false);
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["a"]);
});

test("an empty selection changes nothing in either direction", () => {
  reset(["a", "b"]);
  usePinnedChatsStore.getState().setPinned([], true);
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["a", "b"]);
  usePinnedChatsStore.getState().setPinned([], false);
  assert.deepEqual(usePinnedChatsStore.getState().pinnedIds, ["a", "b"]);
});

test("a shift range survives the list re-sorting between the two clicks", () => {
  // A background stream re-sorts the list by updatedAt mid-selection; the
  // anchor must still name the chat it was set on.
  const before = ["a", "b", "c", "d"];
  const after = ["d", "a", "b", "c"];
  // Anchor on "b", then shift-click "c" after the reorder: still exactly b..c.
  assert.deepEqual(rangeBetween(after, "b", "c"), ["b", "c"]);
  // An index anchor would instead sweep a..c, deleting "a" unasked.
  assert.equal(before.indexOf("b"), 1);
  assert.equal(after[1], "a");
  assert.deepEqual(rangeBetween(after, after[1], "c"), ["a", "b", "c"]);
});
