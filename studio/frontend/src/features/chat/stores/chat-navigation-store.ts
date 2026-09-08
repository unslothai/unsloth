// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";

/** What the navigation chords need: the ordered rows the sidebar renders, the open chat, the
 *  unread set and the rows wanting attention. The sidebar owns the sorting and publishes its
 *  finished lists here, so nothing recomputes a copy that could disagree with the screen. */

/** How many rows back ⌃Tab can walk, bounded so the array cannot grow all day. */
const RECENTLY_VIEWED_LIMIT = 24;

export interface ChatNavigationState {
  /** The pinned block, in the order the sidebar draws it. */
  pinnedItems: SidebarItem[];
  /** Project chats on screen. Empty when the sidebar is organized as one list, where those chats
   *  are in Recents instead. */
  projectItems: SidebarItem[];
  /** Recents, likewise. ⌥⌘1-6 indexes into this list alone. */
  recentItems: SidebarItem[];
  /** Rows generating, queued or unread, most urgent first. */
  attentionItemIds: string[];
  activeItemId: string | null;
  unreadThreadIds: Set<string>;
  /** Which row each unread thread belongs to, for rows no longer published. */
  unreadRowIds: Record<string, string>;
  /** Row ids, most recently opened first. */
  recentlyViewedIds: string[];
  /** A walk through the stack in progress: the order frozen at the first step, and where in it the walk has got to. */
  traversal: { order: string[]; index: number } | null;
  /** Registered by the sidebar, the only code that can route to a row. */
  openChatItem: ((item: SidebarItem) => void) | null;
  /** Rows or folders are selected, so Escape is already spoken for. Published because bare Escape
   *  also declines a waiting tool call, and one press must not do both. */
  selectionActive: boolean;

  publishLists: (next: {
    pinnedItems: SidebarItem[];
    projectItems: SidebarItem[];
    recentItems: SidebarItem[];
    attentionItemIds: string[];
    activeItemId: string | null;
  }) => void;
  setOpenChatItem: (fn: ((item: SidebarItem) => void) | null) => void;
  /** Drop one account's rows, unread set and walk. Called as the sidebar goes. */
  resetAccountState: () => void;
  setSelectionActive: (active: boolean) => void;
  /** `rowIdByThreadId` groups threads by their row, so a Compare row still counts once after a
   *  collapsed section takes it out of the published lists while its threads stay unread. */
  markThreadsUnread: (
    threadIds: string[],
    rowIdByThreadId?: Record<string, string>,
  ) => void;
  clearThreadsUnread: (threadIds: string[]) => void;
  clearAllUnreads: () => void;
  noteViewed: (itemId: string) => void;
  stepRecentlyViewed: (delta: number) => SidebarItem | null;
  endTraversal: () => void;
}

/** Everything the chords read off a row: which chat, where it routes, and the threads behind
 *  it. Compared by value, not identity, since the sidebar rebuilds its items; ids alone would
 *  keep a row whose project changed and route the next jump to the project it left. */
function sameRows(a: SidebarItem[], b: SidebarItem[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const before = a[i];
    const after = b[i];
    if (
      before.id !== after.id ||
      before.type !== after.type ||
      (before.projectId ?? null) !== (after.projectId ?? null) ||
      (before.threadIds ?? []).join("\u0000") !==
        (after.threadIds ?? []).join("\u0000")
    ) {
      return false;
    }
  }
  return true;
}

function sameStrings(a: string[], b: string[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/** One account's chats. The unread set used to live in component state and died with it; a
 *  module store does not, so signing out has to say so or the next account inherits these rows. */
const ACCOUNT_STATE = {
  pinnedItems: [] as SidebarItem[],
  projectItems: [] as SidebarItem[],
  recentItems: [] as SidebarItem[],
  attentionItemIds: [] as string[],
  activeItemId: null as string | null,
  unreadThreadIds: new Set<string>(),
  unreadRowIds: {} as Record<string, string>,
  recentlyViewedIds: [] as string[],
  traversal: null as { order: string[]; index: number } | null,
  selectionActive: false,
};

export const useChatNavigationStore = create<ChatNavigationState>(
  (set, get) => ({
    ...ACCOUNT_STATE,
    unreadThreadIds: new Set(),
    openChatItem: null,

    // Published from an effect on every render, so bail out when nothing moved.
    publishLists: (next) =>
      set((state) => {
        if (
          state.activeItemId === next.activeItemId &&
          sameRows(state.pinnedItems, next.pinnedItems) &&
          sameRows(state.projectItems, next.projectItems) &&
          sameRows(state.recentItems, next.recentItems) &&
          sameStrings(state.attentionItemIds, next.attentionItemIds)
        ) {
          return state;
        }
        return next;
      }),

    setOpenChatItem: (fn) => set({ openChatItem: fn }),

    // A fresh Set each time, or every account after the first shares one.
    resetAccountState: () =>
      set({ ...ACCOUNT_STATE, unreadThreadIds: new Set(), unreadRowIds: {} }),

    setSelectionActive: (active) =>
      set((state) =>
        state.selectionActive === active ? state : { selectionActive: active },
      ),

    markThreadsUnread: (threadIds, rowIdByThreadId) =>
      set((state) => {
        if (threadIds.length === 0) return state;
        const unreadThreadIds = new Set(state.unreadThreadIds);
        for (const id of threadIds) unreadThreadIds.add(id);
        const unreadRowIds = { ...state.unreadRowIds };
        for (const id of threadIds) {
          const rowId = rowIdByThreadId?.[id];
          if (rowId) unreadRowIds[id] = rowId;
        }
        return { unreadThreadIds, unreadRowIds };
      }),

    clearThreadsUnread: (threadIds) =>
      set((state) => {
        if (!threadIds.some((id) => state.unreadThreadIds.has(id))) {
          return state;
        }
        const unreadThreadIds = new Set(state.unreadThreadIds);
        const unreadRowIds = { ...state.unreadRowIds };
        for (const id of threadIds) {
          unreadThreadIds.delete(id);
          delete unreadRowIds[id];
        }
        return { unreadThreadIds, unreadRowIds };
      }),

    clearAllUnreads: () =>
      set((state) =>
        state.unreadThreadIds.size === 0
          ? state
          : { unreadThreadIds: new Set(), unreadRowIds: {} },
      ),

    noteViewed: (itemId) => {
      const { recentlyViewedIds, traversal } = get();
      // Mid-walk the stack holds still: promoting each chat it lands on would swap the top two, so
      // the next step comes straight back and everything below stays unreachable.
      if (traversal && traversal.order[traversal.index] === itemId) return;
      if (recentlyViewedIds[0] === itemId) {
        if (traversal) set({ traversal: null });
        return;
      }
      set({
        traversal: null,
        recentlyViewedIds: [
          itemId,
          ...recentlyViewedIds.filter((id) => id !== itemId),
        ].slice(0, RECENTLY_VIEWED_LIMIT),
      });
    },

    stepRecentlyViewed: (delta) => {
      const state = get();
      const byId = new Map(
        visibleChatItems(state).map((item) => [item.id, item]),
      );
      // Rows deleted or archived since the walk began would be dead stops.
      const order = (state.traversal?.order ?? state.recentlyViewedIds).filter(
        (id) => byId.has(id),
      );
      if (order.length === 0) return null;
      const from = state.traversal
        ? order.indexOf(state.traversal.order[state.traversal.index])
        : state.activeItemId
          ? order.indexOf(state.activeItemId)
          : -1;
      // Nothing open, or a chat outside the stack: the walk starts outside it, so the first step
      // lands on the end it walks from.
      const index =
        from === -1
          ? delta > 0
            ? 0
            : order.length - 1
          : (from + delta + order.length) % order.length;
      set({ traversal: { order, index } });
      return byId.get(order[index]) ?? null;
    },

    endTraversal: () => {
      const { traversal, recentlyViewedIds } = get();
      if (!traversal) return;
      // The walk is over, so the chat it finished on takes the top, the way it would have if it had
      // been opened by hand.
      const landed = traversal.order[traversal.index];
      set({
        traversal: null,
        recentlyViewedIds: [
          landed,
          ...recentlyViewedIds.filter((id) => id !== landed),
        ].slice(0, RECENTLY_VIEWED_LIMIT),
      });
    },
  }),
);

/** Every chat row the sidebar shows, in draw order: pinned, then project folders, then Recents.
 *  A pinned project chat is drawn twice, so the first of the pair wins and the walk does not
 *  stop on it again. */
export function visibleChatItems(state: ChatNavigationState): SidebarItem[] {
  const seen = new Set<string>();
  const out: SidebarItem[] = [];
  for (const item of [
    ...state.pinnedItems,
    ...state.projectItems,
    ...state.recentItems,
  ]) {
    if (seen.has(item.id)) continue;
    seen.add(item.id);
    out.push(item);
  }
  return out;
}

/** How many chats hold an unread thread. Not the set's size: a Compare row is backed by two
 *  threads and marked unread by both, so the set counted it twice. Unreads no row accounts
 *  for still count one each, since the wipe clears them. */
export function countUnreadRows(state: ChatNavigationState): number {
  const listed = new Set<string>();
  let rows = 0;
  for (const item of visibleChatItems(state)) {
    const own = (item.threadIds?.length ? item.threadIds : [item.id]).filter(
      (id) => state.unreadThreadIds.has(id),
    );
    if (own.length === 0) continue;
    rows += 1;
    for (const id of own) listed.add(id);
  }
  // Plus the unreads no row accounts for: a chat archived while unread leaves its thread in the
  // set, and the wipe clears those too. Grouped by row, so a hidden Compare row counts once
  // rather than once per pane; an id with no group recorded stands for itself.
  const hidden = new Set<string>();
  for (const id of state.unreadThreadIds) {
    if (!listed.has(id)) hidden.add(state.unreadRowIds[id] ?? id);
  }
  return rows + hidden.size;
}

/** The row `slot` (1-based) of Recents only, ignoring the pinned block. */
export function recentChatItemAtSlot(
  state: ChatNavigationState,
  slot: number,
): SidebarItem | null {
  return state.recentItems[slot - 1] ?? null;
}

/** The row `delta` places from the open one, wrapping. With nothing open, the first row or the
 *  last, so the chord still does something. */
export function adjacentChatItem(
  state: ChatNavigationState,
  delta: number,
): SidebarItem | null {
  const items = visibleChatItems(state);
  if (items.length === 0) return null;
  const current = items.findIndex((item) => item.id === state.activeItemId);
  if (current === -1) return delta > 0 ? items[0] : items[items.length - 1];
  const next = (current + delta + items.length) % items.length;
  return items[next];
}

/** The next row wanting attention after the open one, wrapping. The list arrives already
 *  ordered, so this only picks the next entry. */
export function nextAttentionChatItem(
  state: ChatNavigationState,
): SidebarItem | null {
  const { attentionItemIds } = state;
  if (attentionItemIds.length === 0) return null;
  const byId = new Map(visibleChatItems(state).map((item) => [item.id, item]));
  const current = state.activeItemId
    ? attentionItemIds.indexOf(state.activeItemId)
    : -1;
  for (let step = 1; step <= attentionItemIds.length; step++) {
    const id =
      attentionItemIds[
        (current + step + attentionItemIds.length) % attentionItemIds.length
      ];
    const item = byId.get(id);
    if (item && item.id !== state.activeItemId) return item;
  }
  return null;
}

/** Open a row through the callback the sidebar registered. */
export function openChatItemById(item: SidebarItem | null): void {
  if (!item) return;
  useChatNavigationStore.getState().openChatItem?.(item);
}
