// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  adjacentChatItem,
  nextAttentionChatItem,
  recentChatItemAtSlot,
  useChatNavigationStore,
  visibleChatItems,
} from "../src/features/chat/stores/chat-navigation-store.ts";
import type { SidebarItem } from "../src/features/chat/hooks/use-chat-sidebar-items.ts";

function row(id: string): SidebarItem {
  return { id, title: id, type: "single", updatedAt: 0 } as SidebarItem;
}

const store = () => useChatNavigationStore.getState();

/** Four rows, visited D, C, B, A, so the stack reads A, B, C, D. */
function seed(): void {
  useChatNavigationStore.setState({
    recentlyViewedIds: [],
    traversal: null,
    unreadThreadIds: new Set(),
  });
  store().publishLists({
    pinnedItems: [],
    recentItems: ["A", "B", "C", "D"].map(row),
    attentionItemIds: [],
    activeItemId: "A",
  });
  for (const id of ["D", "C", "B", "A"]) store().noteViewed(id);
}

/** One press: step, open the row, and let the sidebar note the new chat. */
function press(delta: number): string | null {
  const next = store().stepRecentlyViewed(delta);
  if (!next) return null;
  store().noteViewed(next.id);
  useChatNavigationStore.setState({ activeItemId: next.id });
  return next.id;
}

test("a held walk reaches past the second chat", () => {
  seed();
  assert.deepEqual(store().recentlyViewedIds, ["A", "B", "C", "D"]);
  // Promoting each chat as the walk lands on it would swap the top two and
  // send the next press straight back, stranding everything below them.
  assert.deepEqual(
    [press(1), press(1), press(1), press(1)],
    ["B", "C", "D", "A"],
  );
});

test("releasing the modifier puts the chat it landed on on top", () => {
  seed();
  press(1);
  press(1);
  assert.deepEqual(store().recentlyViewedIds, ["A", "B", "C", "D"]);
  store().endTraversal();
  assert.deepEqual(store().recentlyViewedIds, ["C", "A", "B", "D"]);
  assert.equal(store().traversal, null);
});

test("tapping the chord toggles the two most recent chats", () => {
  seed();
  // Tap, release, tap: the same two chats, as an app switcher does.
  const taps: (string | null)[] = [];
  for (let i = 0; i < 3; i++) {
    taps.push(press(1));
    store().endTraversal();
  }
  assert.deepEqual(taps, ["B", "A", "B"]);
});

test("walking backwards is the mirror of walking forwards", () => {
  seed();
  assert.deepEqual([press(-1), press(-1)], ["D", "C"]);
});

test("opening a chat by hand ends the walk and promotes it", () => {
  seed();
  press(1);
  store().noteViewed("D");
  assert.equal(store().traversal, null);
  assert.deepEqual(store().recentlyViewedIds, ["D", "A", "B", "C"]);
});

test("a row that disappears mid-walk is stepped over", () => {
  seed();
  press(1);
  store().publishLists({
    pinnedItems: [],
    recentItems: ["A", "B", "D"].map(row),
    attentionItemIds: [],
    activeItemId: "B",
  });
  // C was next in the frozen order, and is gone.
  assert.equal(press(1), "D");
});

test("an empty stack has nothing to walk", () => {
  useChatNavigationStore.setState({ recentlyViewedIds: [], traversal: null });
  store().publishLists({
    pinnedItems: [],
    recentItems: [],
    attentionItemIds: [],
    activeItemId: null,
  });
  assert.equal(store().stepRecentlyViewed(1), null);
  // Ending a walk that never began is a no-op, not a crash.
  store().endTraversal();
  assert.deepEqual(store().recentlyViewedIds, []);
});

test("the other navigation selectors read the published lists", () => {
  seed();
  assert.deepEqual(
    visibleChatItems(store()).map((item) => item.id),
    ["A", "B", "C", "D"],
  );
  assert.equal(recentChatItemAtSlot(store(), 2)?.id, "B");
  assert.equal(adjacentChatItem(store(), 1)?.id, "B");
  // Wraps at the top rather than stopping.
  assert.equal(adjacentChatItem(store(), -1)?.id, "D");
  assert.equal(nextAttentionChatItem(store()), null);
});
