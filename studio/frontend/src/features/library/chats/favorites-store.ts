// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

export const LIBRARY_CHAT_FAVORITES_STORAGE_KEY = "unsloth_library_chat_favorites";

// Starred chats, projects and sections, stored locally like pins. Listed only in Favorites and Chats.
export interface ChatFavoritesState {
  /** Chat row ids: a thread id, or a compare pair's id. Newest star first. */
  chatIds: string[];
  projectIds: string[];
  sectionIds: string[];
  setChats: (ids: string[], favorite: boolean) => void;
  setProjects: (ids: string[], favorite: boolean) => void;
  setSections: (ids: string[], favorite: boolean) => void;
}

function withIds(current: string[], ids: string[], on: boolean): string[] {
  if (!on) {
    const dropping = new Set(ids);
    return current.filter((id) => !dropping.has(id));
  }
  const additions = ids.filter((id) => !current.includes(id));
  return additions.length ? [...additions, ...current] : current;
}

function readIds(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((id): id is string => typeof id === "string") : [];
}

export const useChatFavoritesStore = create<ChatFavoritesState>()(
  persist(
    (set) => ({
      chatIds: [],
      projectIds: [],
      sectionIds: [],
      setChats: (ids, favorite) =>
        set((state) => ({ chatIds: withIds(state.chatIds, ids, favorite) })),
      setProjects: (ids, favorite) =>
        set((state) => ({ projectIds: withIds(state.projectIds, ids, favorite) })),
      setSections: (ids, favorite) =>
        set((state) => ({ sectionIds: withIds(state.sectionIds, ids, favorite) })),
    }),
    {
      name: LIBRARY_CHAT_FAVORITES_STORAGE_KEY,
      merge: (persisted, current) => {
        const saved = persisted as Partial<ChatFavoritesState> | undefined;
        return {
          ...current,
          chatIds: readIds(saved?.chatIds),
          projectIds: readIds(saved?.projectIds),
          sectionIds: readIds(saved?.sectionIds),
        };
      },
    },
  ),
);
