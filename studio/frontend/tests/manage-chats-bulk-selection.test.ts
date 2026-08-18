// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Bulk pin/unpin ordering, and the scope the header checkbox claims. The .tsx
// view pulls in the whole app, so it is read as text; the store runs for real.

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
const { rangeBetween } = await import(
  "../src/features/chat/utils/row-selection.ts"
);

async function manageChatsSource(): Promise<string> {
  return await readFile(
    new URL(
      "../src/features/settings/components/manage-chats-view.tsx",
      import.meta.url,
    ),
    "utf8",
  );
}

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

test("the shift anchor is stored as a chat id, not a row index", async () => {
  const src = await manageChatsSource();
  assert.match(src, /useRef<string \| null>\(null\)/);
  assert.match(src, /rangeBetween\(/);
  // An index anchor addresses whatever row now sits at the saved position.
  assert.doesNotMatch(src, /lastToggledIndex/);
});

test("bulk delete passes the always-delete-files preference through", async () => {
  const src = await manageChatsSource();
  // Without this, deleteChatItems defaults args to {} and sends
  // delete_files:false, keeping every selected chat's sandbox.
  assert.match(
    src,
    /useChatPreferencesStore\(\s*\(s\) => s\.alwaysDeleteChatFiles,?\s*\)/,
  );
  assert.match(
    src,
    /deleteChatItems\(selectedItems, openChatId, resetView, \{\s*deleteFiles: alwaysDeleteChatFiles,\s*\}\)/,
  );
});

test("the header checkbox says it selects the visible chats, which is what it does", async () => {
  const src = await manageChatsSource();
  // Rows sit behind a "Show more", so the label must not promise the whole list.
  assert.match(src, /onCheckedChange=\{toggleAllVisible\}/);
  assert.match(src, /aria-label="Select all visible chats"/);
  assert.doesNotMatch(src, /aria-label="Select all chats"/);
});
