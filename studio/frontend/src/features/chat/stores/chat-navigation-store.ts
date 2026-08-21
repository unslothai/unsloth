// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";

/**
 * What the navigation chords need: the ordered rows the sidebar renders, the
 * open chat, the unread set and the rows wanting attention. The sidebar owns
 * the sorting and publishes its finished lists here, so nothing recomputes a
 * second copy that could disagree with the screen.
 */

/** How many rows back ⌃Tab can walk, bounded so the array cannot grow all day. */
const RECENTLY_VIEWED_LIMIT = 24;

export interface ChatNavigationState {
  /** The pinned block, in the order the sidebar draws it. */
  pinnedItems: SidebarItem[];
  /** Recents, likewise. ⌥⌘1-6 indexes into this list alone. */
  recentItems: SidebarItem[];
  /** Rows generating, queued or unread, most urgent first. */
  attentionItemIds: string[];
  activeItemId: string | null;
  unreadThreadIds: Set<string>;
  /** Row ids, most recently opened first. */
  recentlyViewedIds: string[];
  /** Registered by the sidebar, the only code that can route to a row. */
  openChatItem: ((item: SidebarItem) => void) | null;

  publishLists: (next: {
    pinnedItems: SidebarItem[];
    recentItems: SidebarItem[];
    attentionItemIds: string[];
    activeItemId: string | null;
  }) => void;
  setOpenChatItem: (fn: ((item: SidebarItem) => void) | null) => void;
  markThreadsUnread: (threadIds: string[]) => void;
  clearThreadsUnread: (threadIds: string[]) => void;
  clearAllUnreads: () => void;
  noteViewed: (itemId: string) => void;
}

function sameIds(a: SidebarItem[], b: SidebarItem[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].id !== b[i].id) return false;
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

export const useChatNavigationStore = create<ChatNavigationState>(
  (set, get) => ({
    pinnedItems: [],
    recentItems: [],
    attentionItemIds: [],
    activeItemId: null,
    unreadThreadIds: new Set(),
    recentlyViewedIds: [],
    openChatItem: null,

    // Published from an effect on every render, so bail out when nothing moved.
    publishLists: (next) =>
      set((state) => {
        if (
          state.activeItemId === next.activeItemId &&
          sameIds(state.pinnedItems, next.pinnedItems) &&
          sameIds(state.recentItems, next.recentItems) &&
          sameStrings(state.attentionItemIds, next.attentionItemIds)
        ) {
          return state;
        }
        return next;
      }),

    setOpenChatItem: (fn) => set({ openChatItem: fn }),

    markThreadsUnread: (threadIds) =>
      set((state) => {
        if (threadIds.length === 0) return state;
        const unreadThreadIds = new Set(state.unreadThreadIds);
        for (const id of threadIds) unreadThreadIds.add(id);
        return { unreadThreadIds };
      }),

    clearThreadsUnread: (threadIds) =>
      set((state) => {
        if (!threadIds.some((id) => state.unreadThreadIds.has(id))) {
          return state;
        }
        const unreadThreadIds = new Set(state.unreadThreadIds);
        for (const id of threadIds) unreadThreadIds.delete(id);
        return { unreadThreadIds };
      }),

    clearAllUnreads: () =>
      set((state) =>
        state.unreadThreadIds.size === 0
          ? state
          : { unreadThreadIds: new Set() },
      ),

    noteViewed: (itemId) => {
      const { recentlyViewedIds } = get();
      if (recentlyViewedIds[0] === itemId) return;
      set({
        recentlyViewedIds: [
          itemId,
          ...recentlyViewedIds.filter((id) => id !== itemId),
        ].slice(0, RECENTLY_VIEWED_LIMIT),
      });
    },
  }),
);

/** Every chat row the sidebar shows, pinned block first. */
export function visibleChatItems(state: ChatNavigationState): SidebarItem[] {
  return [...state.pinnedItems, ...state.recentItems];
}

/** The row `slot` (1-based) of Recents only, ignoring the pinned block. */
export function recentChatItemAtSlot(
  state: ChatNavigationState,
  slot: number,
): SidebarItem | null {
  return state.recentItems[slot - 1] ?? null;
}

/** The row `delta` places from the open one, wrapping. With nothing open, the
 *  first row or the last, so the chord still does something. */
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

/** Walk the most-recently-viewed stack. Position 0 is the open chat. */
export function recentlyViewedChatItem(
  state: ChatNavigationState,
  delta: number,
): SidebarItem | null {
  const { recentlyViewedIds } = state;
  if (recentlyViewedIds.length === 0) return null;
  const byId = new Map(visibleChatItems(state).map((item) => [item.id, item]));
  // Deleted or archived rows would be dead stops.
  const live = recentlyViewedIds.filter((id) => byId.has(id));
  if (live.length === 0) return null;
  const current = state.activeItemId ? live.indexOf(state.activeItemId) : -1;
  const base = current === -1 ? 0 : current;
  const next = (base + delta + live.length) % live.length;
  return byId.get(live[next]) ?? null;
}

/** The next row wanting attention after the open one, wrapping. The list
 *  arrives already ordered, so this only picks the next entry. */
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
