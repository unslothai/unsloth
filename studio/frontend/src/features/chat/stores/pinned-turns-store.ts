// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

// pinned turns per chat, keyed by thread id to the opening user message ids, kept in localstorage like pinned chats
export interface PinnedTurnsState {
  pinnedByThread: Record<string, string[]>;
  togglePinnedTurn: (threadId: string, messageId: string) => void;
}

export const usePinnedTurnsStore = create<PinnedTurnsState>()(
  persist(
    (set) => ({
      pinnedByThread: {},
      togglePinnedTurn: (threadId, messageId) =>
        set((state) => {
          const current = state.pinnedByThread[threadId] ?? [];
          const next = current.includes(messageId)
            ? current.filter((id) => id !== messageId)
            : [...current, messageId];
          const others = Object.fromEntries(
            Object.entries(state.pinnedByThread).filter(
              ([id]) => id !== threadId,
            ),
          );
          return {
            pinnedByThread:
              next.length > 0 ? { ...others, [threadId]: next } : others,
          };
        }),
    }),
    {
      name: "unsloth_pinned_turns",
      merge: (persisted, current) => {
        const saved = (persisted as Partial<PinnedTurnsState> | undefined)
          ?.pinnedByThread;
        return {
          ...current,
          pinnedByThread:
            saved && typeof saved === "object" && !Array.isArray(saved)
              ? saved
              : {},
        };
      },
    },
  ),
);
