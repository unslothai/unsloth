// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

// Here, not in the journal, so the startup hook can read it without loading the sync runtime.
export const PIN_JOURNAL_PREFIX = "unsloth_chat_pin_journal:";

export interface PinnedChatsActions {
  setPinned: (ids: string[], pinned: boolean) => void;
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
}));
