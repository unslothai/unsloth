// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// thread id -> opening user message ids of its bookmarked turns
export interface BookmarkedTurnsState {
  bookmarkedByThread: Record<string, string[]>;
  toggleBookmarkedTurn: (threadId: string, messageId: string) => void;
  forgetThreads: (threadIds: readonly string[]) => void;
  forgetTurns: (threadId: string, messageIds: readonly string[]) => void;
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
      forgetThreads: (threadIds) =>
        set((state) => {
          if (!threadIds.some((id) => id in state.bookmarkedByThread)) {
            return state;
          }
          const forget = new Set(threadIds);
          return {
            bookmarkedByThread: Object.fromEntries(
              Object.entries(state.bookmarkedByThread).filter(
                ([id]) => !forget.has(id),
              ),
            ),
          };
        }),
      forgetTurns: (threadId, messageIds) =>
        set((state) => {
          const current = state.bookmarkedByThread[threadId];
          if (!current?.some((id) => messageIds.includes(id))) {
            return state;
          }
          const next = current.filter((id) => !messageIds.includes(id));
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
