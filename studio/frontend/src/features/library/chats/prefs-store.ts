// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { LibraryView } from "../settings-store";
import type { ChatGroupBy, ChatSort, DateField, ProjectSort, SectionSort } from "./model";

export const LIBRARY_CHATS_PREFS_STORAGE_KEY = "unsloth_library_chats_prefs";

export interface ChatsPrefs {
  view: LibraryView;
  sort: ChatSort;
  projectSort: ProjectSort;
  sectionSort: SectionSort;
  dateField: DateField;
  groupBy: ChatGroupBy;
  pinnedFirst: boolean;
}

export const DEFAULT_CHATS_PREFS: ChatsPrefs = {
  view: "list",
  sort: { key: "modified", desc: true },
  projectSort: { key: "modified", desc: true },
  sectionSort: { key: "modified", desc: true },
  dateField: "modified",
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
      version: 3,
      // v2: list is the default view. v3: Last modified is the default date, so an untouched
      // Last active moves with it.
      migrate: (persisted, version) => {
        const saved = { ...((persisted ?? {}) as Partial<ChatsPrefs>) };
        if (version < 2) saved.view = "list";
        if (version < 3) {
          const modified = { key: "modified", desc: true } as const;
          const untouched = (sort?: { key: string; desc: boolean }) =>
            !sort || (sort.key === "updated" && sort.desc);
          if (!saved.dateField || saved.dateField === "updated") saved.dateField = "modified";
          if (untouched(saved.sort)) saved.sort = modified;
          if (untouched(saved.projectSort)) saved.projectSort = modified;
          if (untouched(saved.sectionSort)) saved.sectionSort = modified;
        }
        return saved;
      },
    },
  ),
);
