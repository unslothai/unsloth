// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { LibraryTab } from "./search";

export const LIBRARY_SETTINGS_STORAGE_KEY = "unsloth_library_settings";
export const LIBRARY_VIEW_STORAGE_KEY = "unsloth_library_view";

type LibrarySort = "recent" | "oldest" | "name" | "size";
export type LibraryTabVisibility = "always" | "auto" | "hidden";

export interface LibrarySettings {
  cardSize: "small" | "medium" | "large";
  imageLayout: "masonry" | "square";
  showCardDates: boolean;
  sort: LibrarySort;
  startTab: "last" | "suggested" | "favorites" | "folders" | "all";
  lastTab: LibraryTab;
  tabs: Record<LibraryTab, LibraryTabVisibility>;
  suggestedLimit: number;
  showChatAttachments: boolean;
  showChatToolFiles: boolean;
  showGeneratedMedia: boolean;
  showFineTunes: boolean;
  confirmDelete: boolean;
}

export const DEFAULT_LIBRARY_SETTINGS: LibrarySettings = {
  cardSize: "medium",
  imageLayout: "masonry",
  showCardDates: true,
  sort: "recent",
  startTab: "suggested",
  lastTab: "suggested",
  tabs: {
    suggested: "always",
    favorites: "always",
    folders: "always",
    images: "auto",
    videos: "auto",
    audio: "auto",
    models: "auto",
    all: "always",
  },
  suggestedLimit: 40,
  showChatAttachments: true,
  showChatToolFiles: true,
  showGeneratedMedia: true,
  showFineTunes: true,
  confirmDelete: true,
};

export const SUGGESTED_LIMITS = [20, 40, 80] as const;

export const CARD_COLUMNS: Record<LibrarySettings["cardSize"], { minWidth: number; max: number }> = {
  small: { minWidth: 140, max: 6 },
  medium: { minWidth: 170, max: 5 },
  large: { minWidth: 210, max: 4 },
};

export type LibraryView = "grid" | "list";

export const useLibraryViewStore = create<{
  view: LibraryView;
  setView: (view: LibraryView) => void;
}>()(
  persist(
    (set) => ({
      view: "grid",
      setView: (view) => set({ view }),
    }),
    { name: LIBRARY_VIEW_STORAGE_KEY },
  ),
);

interface LibrarySettingsState extends LibrarySettings {
  set: (patch: Partial<LibrarySettings>) => void;
  reset: () => void;
}

/** v1 had one mediaTabs switch for Images, Videos and Audio together; before v3, Fine-tunes
 * always showed. */
export function migrateLibrarySettings(persisted: unknown, version: number): Record<string, unknown> {
  const state = { ...(persisted as Record<string, unknown>) };
  if (version < 2) {
    const media = state.mediaTabs === "always" ? "always" : "auto";
    state.tabs = { ...DEFAULT_LIBRARY_SETTINGS.tabs, images: media, videos: media, audio: media };
    delete state.mediaTabs;
  }
  if (version < 3) {
    const tabs = state.tabs as Record<string, string> | undefined;
    if (tabs?.models === "always") state.tabs = { ...tabs, models: "auto" };
  }
  return state;
}

export const useLibrarySettingsStore = create<LibrarySettingsState>()(
  persist(
    (set) => ({
      ...DEFAULT_LIBRARY_SETTINGS,
      set: (patch) => set(patch),
      reset: () => set(DEFAULT_LIBRARY_SETTINGS),
    }),
    {
      name: LIBRARY_SETTINGS_STORAGE_KEY,
      version: 3,
      migrate: (persisted, version) =>
        migrateLibrarySettings(persisted, version) as unknown as LibrarySettingsState,
    },
  ),
);

const SOURCE_SETTING: Record<string, keyof LibrarySettings> = {
  attachment: "showChatAttachments",
  sandbox: "showChatToolFiles",
  image: "showGeneratedMedia",
  video: "showGeneratedMedia",
  audio: "showGeneratedMedia",
  model: "showFineTunes",
};

export function includedBySettings(itemId: string, settings: LibrarySettings): boolean {
  const key = SOURCE_SETTING[itemId.slice(0, itemId.indexOf(":"))];
  return !key || Boolean(settings[key]);
}

export type LibrarySortKey = "name" | "modified" | "size";
export type LibrarySortState = { key: LibrarySortKey; desc: boolean };

export type LibraryUrlSort = LibrarySort | "name-desc" | "size-asc";

export const SORT_STATES: Record<LibraryUrlSort, LibrarySortState> = {
  recent: { key: "modified", desc: true },
  oldest: { key: "modified", desc: false },
  name: { key: "name", desc: false },
  "name-desc": { key: "name", desc: true },
  size: { key: "size", desc: true },
  "size-asc": { key: "size", desc: false },
};

export const LIBRARY_URL_SORTS = Object.keys(SORT_STATES) as LibraryUrlSort[];

export function sortParam(state: LibrarySortState): LibraryUrlSort {
  return LIBRARY_URL_SORTS.find(
    (sort) => SORT_STATES[sort].key === state.key && SORT_STATES[sort].desc === state.desc,
  )!;
}

export function nextSort(current: LibrarySortState, key: LibrarySortKey): LibrarySortState {
  return current.key === key ? { key, desc: !current.desc } : { key, desc: key !== "name" };
}

type Sortable = { name: string; updatedAt: number; sizeBytes?: number | null };

export function lastActivity(item: { updatedAt: number; openedAt?: number | null }): number {
  return Math.max(item.updatedAt, item.openedAt ?? 0);
}

export function compareBySort({ key, desc }: LibrarySortState): (a: Sortable, b: Sortable) => number {
  const ascending =
    key === "name"
      ? (a: Sortable, b: Sortable) => a.name.localeCompare(b.name, undefined, { numeric: true })
      : key === "size"
        ? (a: Sortable, b: Sortable) => (a.sizeBytes ?? -1) - (b.sizeBytes ?? -1)
        : (a: Sortable, b: Sortable) => a.updatedAt - b.updatedAt;
  return desc ? (a, b) => ascending(b, a) : ascending;
}

export const useLibraryVisitStore = create<{ visit: number; restart: () => void }>((set) => ({
  visit: 0,
  restart: () => set((state) => ({ visit: state.visit + 1 })),
}));
