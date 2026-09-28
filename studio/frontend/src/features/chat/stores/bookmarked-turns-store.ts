// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// bookmarked turns per chat, keyed by thread id to the opening user message ids, kept in localstorage like pinned chats
export interface BookmarkedTurnsState {
  bookmarkedByThread: Record<string, string[]>;
  toggleBookmarkedTurn: (threadId: string, messageId: string) => void;
}

export const useBookmarkedTurnsStore = create<BookmarkedTurnsState>()(
  persist(
    (set) => ({
      bookmarkedByThread: {},
      toggleBookmarkedTurn: (threadId, messageId) =>
        set((state) => {
          const current = state.bookmarkedByThread[threadId] ?? [];
          const next = current.includes(messageId)
            ? current.filter((id) => id !== messageId)
            : [...current, messageId];
          const others = Object.fromEntries(
            Object.entries(state.bookmarkedByThread).filter(
              ([id]) => id !== threadId,
            ),
          );
          return {
            bookmarkedByThread:
              next.length > 0 ? { ...others, [threadId]: next } : others,
          };
        }),
    }),
    {
      name: "unsloth_bookmarked_turns",
      merge: (persisted, current) => {
        const saved = (persisted as Partial<BookmarkedTurnsState> | undefined)
          ?.bookmarkedByThread;
        return {
          ...current,
          bookmarkedByThread:
            saved && typeof saved === "object" && !Array.isArray(saved)
              ? saved
              : {},
        };
      },
    },
  ),
);
