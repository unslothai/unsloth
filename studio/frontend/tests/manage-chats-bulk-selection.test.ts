// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Bulk pin/unpin ordering, and the scope the header checkbox claims. The view is
// .tsx and pulls in the whole app, so it is read as text the way
// chat-only-route-guard.test.ts does; the store is exercised for real.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
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

test("the header checkbox says it selects the visible chats, which is what it does", async () => {
  const src = await readFile(
    new URL(
      "../src/features/settings/components/manage-chats-view.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  // Rows sit behind a "Show more", so the control cannot reach the whole list
  // and the label must not promise that it does.
  assert.match(src, /onCheckedChange=\{toggleAllVisible\}/);
  assert.match(src, /aria-label="Select all visible chats"/);
  assert.doesNotMatch(src, /aria-label="Select all chats"/);
});
