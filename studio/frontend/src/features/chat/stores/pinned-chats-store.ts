// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

export type PinnedChatPlacement = "before" | "after";

export interface PinnedChatsActions {
  setPinned: (ids: string[], pinned: boolean) => void;
  move: (id: string, targetId: string, placement: PinnedChatPlacement) => void;
}

export interface PinnedChatsState extends PinnedChatsActions {
  /** Stable chat IDs (or pair IDs for comparisons), in manual pin order. */
  pinnedIds: string[];
  togglePin: (id: string) => void;
  unpin: (id: string) => void;
}

let actions: PinnedChatsActions | null = null;

/** Bind persistence at runtime without pulling API/auth modules into this state store. */
export function setPinnedChatsActions(next: PinnedChatsActions | null): void {
  actions = next;
}

export const usePinnedChatsStore = create<PinnedChatsState>((set, get) => ({
  pinnedIds: [],
  togglePin: (id) => get().setPinned([id], !get().pinnedIds.includes(id)),
  unpin: (id) => get().setPinned([id], false),
  setPinned: (ids, pinned) => {
    if (actions) {
      actions.setPinned(ids, pinned);
      return;
    }
    set((state) => {
      if (!pinned) {
        const dropping = new Set(ids);
        const next = state.pinnedIds.filter((id) => !dropping.has(id));
        return next.length === state.pinnedIds.length
          ? state
          : { pinnedIds: next };
      }
      const additions = [...new Set(ids)].filter(
        (id) => !state.pinnedIds.includes(id),
      );
      return additions.length === 0
        ? state
        : { pinnedIds: [...additions, ...state.pinnedIds] };
    });
  },
  move: (id, targetId, placement) => {
    if (actions) {
      actions.move(id, targetId, placement);
      return;
    }
    set((state) => {
      if (
        id === targetId ||
        !state.pinnedIds.includes(id) ||
        !state.pinnedIds.includes(targetId)
      )
        return state;
      const next = state.pinnedIds.filter((entry) => entry !== id);
      next.splice(
        next.indexOf(targetId) + (placement === "after" ? 1 : 0),
        0,
        id,
      );
      return next.every((entry, index) => entry === state.pinnedIds[index])
        ? state
        : { pinnedIds: next };
    });
  },
}));
