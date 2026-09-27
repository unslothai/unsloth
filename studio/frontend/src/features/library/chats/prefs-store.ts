// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { LibraryView } from "../settings-store";
import type { ChatGroupBy, ChatSort, ProjectSort, SectionSort } from "./model";

export const LIBRARY_CHATS_PREFS_STORAGE_KEY = "unsloth_library_chats_prefs";

export interface ChatsPrefs {
  view: LibraryView;
  sort: ChatSort;
  projectSort: ProjectSort;
  sectionSort: SectionSort;
  groupBy: ChatGroupBy;
  pinnedFirst: boolean;
}

export const DEFAULT_CHATS_PREFS: ChatsPrefs = {
  view: "list",
  sort: { key: "updated", desc: true },
  projectSort: { key: "updated", desc: true },
  sectionSort: { key: "updated", desc: true },
  groupBy: "date",
  pinnedFirst: true,
};

// Separate from the Library store so file-tab view and sort settings never apply to chats.
export const useChatsPrefsStore = create<ChatsPrefs & { set: (patch: Partial<ChatsPrefs>) => void }>()(
  persist(
    (set) => ({
      ...DEFAULT_CHATS_PREFS,
      set: (patch) => set(patch),
    }),
    {
      name: LIBRARY_CHATS_PREFS_STORAGE_KEY,
      version: 2,
      // v2 made list the default view: reset grids saved earlier.
      migrate: (persisted, version) => {
        const saved = (persisted ?? {}) as Partial<ChatsPrefs>;
        return version < 2 ? { ...saved, view: "list" } : saved;
      },
    },
  ),
);
