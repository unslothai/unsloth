// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  adjacentChatItem,
  nextAttentionChatItem,
  countUnreadRows,
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
    projectItems: [],
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

// A walk holds the stack still until it ends. If the release never arrives,
// because the window went away with the modifier still down, the walk is
// still running and the next press carries on from where it stopped.
test("a walk left running sends the next press onward, not back", () => {
  seed();
  assert.equal(press(1), "B");
  // The release lands in another app, so nothing ends the walk.
  assert.deepEqual(store().recentlyViewedIds, ["A", "B", "C", "D"]);
  assert.notEqual(store().traversal, null);
  assert.equal(press(1), "C");

  // Ending it on the way out is what makes the next press a toggle again.
  seed();
  assert.equal(press(1), "B");
  store().endTraversal();
  assert.deepEqual(store().recentlyViewedIds, ["B", "A", "C", "D"]);
  assert.equal(press(1), "A");
});

test("a chat outside the stack walks onto the end it started from", () => {
  seed();
  // A brand new chat is in no stack, so the first press has to reach the most
  // recently viewed one rather than stepping past it.
  useChatNavigationStore.setState({ activeItemId: "unsaved", traversal: null });
  assert.equal(store().stepRecentlyViewed(1)?.id, "A");
  useChatNavigationStore.setState({ activeItemId: "unsaved", traversal: null });
  assert.equal(store().stepRecentlyViewed(-1)?.id, "D");
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
    projectItems: [],
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
    projectItems: [],
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

test("a chat that lives in a project folder is still navigable", () => {
  useChatNavigationStore.setState({ recentlyViewedIds: [], traversal: null });
  // Organized by project, an unpinned project chat is drawn under its folder
  // and never reaches Recents, so it has to be published in its own right.
  store().publishLists({
    pinnedItems: [row("pinned")],
    projectItems: [row("in-project")],
    recentItems: [row("loose")],
    attentionItemIds: [],
    activeItemId: "in-project",
  });
  assert.deepEqual(
    visibleChatItems(store()).map((item) => item.id),
    ["pinned", "in-project", "loose"],
  );
  // Next from the project chat is its neighbour, not the top of the list.
  assert.equal(adjacentChatItem(store(), 1)?.id, "loose");
  assert.equal(adjacentChatItem(store(), -1)?.id, "pinned");
});

test("a row that moves project is republished, not kept by id", () => {
  const inProject = (projectId: string) => ({
    ...row("c1"),
    projectId,
  });
  store().publishLists({
    pinnedItems: [],
    projectItems: [inProject("a")],
    recentItems: [],
    attentionItemIds: [],
    activeItemId: "c1",
  });
  store().publishLists({
    pinnedItems: [],
    projectItems: [inProject("b")],
    recentItems: [],
    attentionItemIds: [],
    activeItemId: "c1",
  });
  // The chords route with this projectId, so a stale one opens the chat in
  // the project it just left.
  assert.equal(visibleChatItems(store())[0].projectId, "b");
});

test("a title-only change does not republish", () => {
  // The sidebar rebuilds its rows every render, so the guard has to hold for
  // the fields nothing here reads, or every render writes to the store.
  store().publishLists({
    pinnedItems: [],
    projectItems: [],
    recentItems: [row("c1")],
    attentionItemIds: [],
    activeItemId: "c1",
  });
  const before = visibleChatItems(store())[0];
  store().publishLists({
    pinnedItems: [],
    projectItems: [],
    recentItems: [{ ...row("c1"), title: "renamed" }],
    attentionItemIds: [],
    activeItemId: "c1",
  });
  assert.equal(visibleChatItems(store())[0], before);
});

test("a pinned project chat is drawn twice but walked once", () => {
  useChatNavigationStore.setState({ recentlyViewedIds: [], traversal: null });
  store().publishLists({
    pinnedItems: [row("both")],
    projectItems: [row("both"), row("other")],
    recentItems: [],
    attentionItemIds: [],
    activeItemId: "both",
  });
  assert.deepEqual(
    visibleChatItems(store()).map((item) => item.id),
    ["both", "other"],
  );
  assert.equal(adjacentChatItem(store(), 1)?.id, "other");
});

test("a Compare row counts once, not once per pane", () => {
  const compare = {
    id: "cmp",
    title: "cmp",
    type: "compare",
    updatedAt: 0,
    threadIds: ["left", "right"],
  } as SidebarItem;
  useChatNavigationStore.setState({ unreadThreadIds: new Set() });
  store().publishLists({
    pinnedItems: [],
    projectItems: [],
    recentItems: [compare, row("A")],
    attentionItemIds: [],
    activeItemId: null,
  });

  // Both panes finishing marks the one row unread twice.
  store().markThreadsUnread(["left", "right"]);
  assert.equal(store().unreadThreadIds.size, 2);
  assert.equal(countUnreadRows(store()), 1);

  store().markThreadsUnread(["A"]);
  assert.equal(countUnreadRows(store()), 2);
});

test("unreads whose row is gone still get a count", () => {
  useChatNavigationStore.setState({ unreadThreadIds: new Set() });
  store().publishLists({
    pinnedItems: [],
    projectItems: [],
    recentItems: [],
    attentionItemIds: [],
    activeItemId: null,
  });
  store().markThreadsUnread(["deleted-1", "deleted-2"]);
  assert.equal(countUnreadRows(store()), 2);
});

test("an unread chat that left the list still counts", () => {
  useChatNavigationStore.setState({ unreadThreadIds: new Set() });
  store().publishLists({
    pinnedItems: [],
    projectItems: [],
    recentItems: [row("A")],
    attentionItemIds: [],
    activeItemId: null,
  });
  // One listed, one whose row was archived out from under it. Clearing wipes
  // both, so reporting only the visible row undercounts what it did.
  store().markThreadsUnread(["A", "gone"]);
  assert.equal(countUnreadRows(store()), 2);
});

test("a hidden Compare row counts once, not once per pane", () => {
  const compare = {
    id: "cmp",
    title: "cmp",
    type: "compare",
    updatedAt: 0,
    threadIds: ["left", "right"],
  } as SidebarItem;
  useChatNavigationStore.setState({
    unreadThreadIds: new Set(),
    unreadRowIds: {},
  });
  // Marked while it was on screen, then the section closes and the published
  // lists no longer carry it. Both threads stay unread.
  store().publishLists({
    pinnedItems: [],
    projectItems: [],
    recentItems: [compare],
    attentionItemIds: [],
    activeItemId: null,
  });
  store().markThreadsUnread(["left", "right"], { left: "cmp", right: "cmp" });
  assert.equal(countUnreadRows(store()), 1);
  store().publishLists({
    pinnedItems: [],
    projectItems: [],
    recentItems: [],
    attentionItemIds: [],
    activeItemId: null,
  });
  assert.equal(store().unreadThreadIds.size, 2);
  assert.equal(countUnreadRows(store()), 1);
});

test("clearing an unread drops the row it was grouped under", () => {
  useChatNavigationStore.setState({
    unreadThreadIds: new Set(),
    unreadRowIds: {},
  });
  store().markThreadsUnread(["left"], { left: "cmp" });
  store().clearThreadsUnread(["left"]);
  assert.deepEqual(store().unreadRowIds, {});
});
