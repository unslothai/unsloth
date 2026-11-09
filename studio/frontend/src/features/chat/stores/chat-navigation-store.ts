// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { SidebarItem } from "../hooks/use-chat-sidebar-items";

/** The sidebar publishes its finished lists here so nothing recomputes a divergent copy. */

const RECENTLY_VIEWED_LIMIT = 24;

export interface ChatNavigationState {
  pinnedItems: SidebarItem[];
  projectItems: SidebarItem[];
  /** ⌥⌘1-6 indexes into this list alone. */
  recentItems: SidebarItem[];
  attentionItemIds: string[];
  activeItemId: string | null;
  unreadThreadIds: Set<string>;
  unreadRowIds: Record<string, string>;
  recentlyViewedIds: string[];
  traversal: { order: string[]; index: number } | null;
  openChatItem: ((item: SidebarItem) => void) | null;
  /** Published because bare Escape also declines a waiting tool call; one press must not do both. */
  selectionActive: boolean;

  publishLists: (next: {
    pinnedItems: SidebarItem[];
    projectItems: SidebarItem[];
    recentItems: SidebarItem[];
    attentionItemIds: string[];
    activeItemId: string | null;
  }) => void;
  setOpenChatItem: (fn: ((item: SidebarItem) => void) | null) => void;
  resetAccountState: () => void;
  setSelectionActive: (active: boolean) => void;
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

/** Compared by value: the sidebar rebuilds items, and a row's project may change. */
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

/** Module-level, so sign-out must reset it or the next account inherits these rows. */
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

    // Published from an effect every render, so bail out when nothing moved.
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
      // Mid-walk the stack holds still, or each step would swap the top two.
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
      const order = (state.traversal?.order ?? state.recentlyViewedIds).filter(
        (id) => byId.has(id),
      );
      if (order.length === 0) return null;
      const from = state.traversal
        ? order.indexOf(state.traversal.order[state.traversal.index])
        : state.activeItemId
          ? order.indexOf(state.activeItemId)
          : -1;
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

/** A pinned project chat is drawn twice; the first wins so the walk does not stop twice. */
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

/** Counts rows, not set size: a Compare row is backed by two threads. */
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
  // Also count unreads no row accounts for (e.g. archived while unread), grouped by row.
  const hidden = new Set<string>();
  for (const id of state.unreadThreadIds) {
    if (!listed.has(id)) hidden.add(state.unreadRowIds[id] ?? id);
  }
  return rows + hidden.size;
}

export function recentChatItemAtSlot(
  state: ChatNavigationState,
  slot: number,
): SidebarItem | null {
  return state.recentItems[slot - 1] ?? null;
}

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

export function openChatItemById(item: SidebarItem | null): void {
  if (!item) return;
  useChatNavigationStore.getState().openChatItem?.(item);
}
