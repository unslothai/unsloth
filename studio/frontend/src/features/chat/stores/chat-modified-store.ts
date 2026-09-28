// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

export const CHAT_MODIFIED_STORAGE_KEY = "unsloth_chat_modified_at";
// Newest entries kept; older ones fall back to the thread's own updatedAt.
const MAX_ENTRIES = 2000;

/** When each thread was last renamed, moved or (un)archived. `updatedAt` only tracks messages. */
export interface ChatModifiedState {
  at: Record<string, number>;
  touch: (threadIds: string[], when?: number) => void;
}

export const useChatModifiedStore = create<ChatModifiedState>()(
  persist(
    (set) => ({
      at: {},
      touch: (threadIds, when = Date.now()) =>
        set((state) => {
          const next = { ...state.at };
          for (const id of threadIds) next[id] = when;
          const ids = Object.keys(next);
          if (ids.length <= MAX_ENTRIES) return { at: next };
          const kept = ids.sort((a, b) => next[b]! - next[a]!).slice(0, MAX_ENTRIES);
          return { at: Object.fromEntries(kept.map((id) => [id, next[id]!])) };
        }),
    }),
    {
      name: CHAT_MODIFIED_STORAGE_KEY,
      partialize: (state) => ({ at: state.at }),
    },
  ),
);

/** Thread patch fields that count as a user edit. */
export function isChatEdit(patch: object): boolean {
  return "title" in patch || "projectId" in patch || "archived" in patch;
}
