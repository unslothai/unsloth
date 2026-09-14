// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// Client-side pin state for chats, keyed by stable chat id. Kept in localStorage, not the chat DB.
// New pins are prepended so the most recently pinned chat sorts first in the Pinned section.
export interface PinnedChatsState {
  pinnedIds: string[];
  togglePin: (id: string) => void;
  unpin: (id: string) => void;
  /** Pins or unpins a whole selection in one write. */
  setPinned: (ids: string[], pinned: boolean) => void;
}

export const usePinnedChatsStore = create<PinnedChatsState>()(
  persist(
    (set) => ({
      pinnedIds: [],
      togglePin: (id) =>
        set((state) => ({
          pinnedIds: state.pinnedIds.includes(id)
            ? state.pinnedIds.filter((x) => x !== id)
            : [id, ...state.pinnedIds],
        })),
      unpin: (id) =>
        set((state) => ({
          pinnedIds: state.pinnedIds.filter((x) => x !== id),
        })),
      setPinned: (ids, pinned) =>
        set((state) => {
          if (!pinned) {
            const dropping = new Set(ids);
            return {
              pinnedIds: state.pinnedIds.filter((id) => !dropping.has(id)),
            };
          }
          // Already pinned chats keep their place; the rest lead, as one pin does.
          const additions = ids.filter((id) => !state.pinnedIds.includes(id));
          if (additions.length === 0) return state;
          return { pinnedIds: [...additions, ...state.pinnedIds] };
        }),
    }),
    {
      name: "unsloth_pinned_chats",
      merge: (persisted, current) => {
        const saved = persisted as Partial<PinnedChatsState> | undefined;
        return {
          ...current,
          pinnedIds: Array.isArray(saved?.pinnedIds) ? saved.pinnedIds : [],
        };
      },
    },
  ),
);
