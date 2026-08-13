// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// Client-side pin state for chats, keyed by stable chat id. Kept in
// localStorage, not the chat DB. New pins are prepended so the most
// recently pinned chat sorts first in the Pinned section.
export interface PinnedChatsState {
  pinnedIds: string[];
  togglePin: (id: string) => void;
  unpin: (id: string) => void;
  setManyPinned: (ids: string[], pinned: boolean) => void;
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
      setManyPinned: (ids, pinned) =>
        set((state) => {
          if (!pinned) {
            const drop = new Set(ids);
            return {
              pinnedIds: state.pinnedIds.filter((x) => !drop.has(x)),
            };
          }
          const additions = ids.filter((id) => !state.pinnedIds.includes(id));
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
