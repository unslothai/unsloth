// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { LibraryTab } from "./search";

export const LIBRARY_SETTINGS_STORAGE_KEY = "unsloth_library_settings";
export const LIBRARY_VIEW_STORAGE_KEY = "unsloth_library_view";

export type LibraryCardSize = "small" | "medium" | "large";
export type LibraryImageLayout = "masonry" | "square";
export type LibrarySort = "recent" | "oldest" | "name" | "size";
export type LibraryStartTab = "last" | "suggested" | "favorites" | "folders" | "all";
/** "auto" shows a tab once it has something in it. */
export type LibraryTabVisibility = "always" | "auto" | "hidden";

export interface LibrarySettings {
  cardSize: LibraryCardSize;
  imageLayout: LibraryImageLayout;
  showCardDates: boolean;
  sort: LibrarySort;
  startTab: LibraryStartTab;
  /** Where "Last visited" reopens. */
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
    models: "always",
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

/** Smallest card width and most columns per row, per card size. */
export const CARD_COLUMNS: Record<LibraryCardSize, { minWidth: number; max: number }> = {
  small: { minWidth: 170, max: 6 },
  medium: { minWidth: 200, max: 5 },
  large: { minWidth: 240, max: 4 },
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

/** v1 had one mediaTabs switch for Images, Videos and Audio together. */
export function migrateLibrarySettings(persisted: unknown, version: number): Record<string, unknown> {
  const state = { ...(persisted as Record<string, unknown>) };
  if (version < 2) {
    const media = state.mediaTabs === "always" ? "always" : "auto";
    state.tabs = { ...DEFAULT_LIBRARY_SETTINGS.tabs, images: media, videos: media, audio: media };
    delete state.mediaTabs;
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
      version: 2,
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

/** Whether the settings let this item into the Library. Library uploads always show. */
export function includedBySettings(itemId: string, settings: LibrarySettings): boolean {
  const key = SOURCE_SETTING[itemId.slice(0, itemId.indexOf(":"))];
  return !key || Boolean(settings[key]);
}

/** A list column and direction; the Sort setting is one of these. */
export type LibrarySortKey = "name" | "modified" | "size";
export interface LibrarySortState {
  key: LibrarySortKey;
  desc: boolean;
}

export const LIBRARY_SORTS: readonly LibrarySort[] = ["recent", "oldest", "name", "size"];

const SORT_STATES: Record<LibrarySort, LibrarySortState> = {
  recent: { key: "modified", desc: true },
  oldest: { key: "modified", desc: false },
  name: { key: "name", desc: false },
  size: { key: "size", desc: true },
};

export function sortState(sort: LibrarySort): LibrarySortState {
  return SORT_STATES[sort];
}

/** Clicking a column flips it, or starts a new one in its natural direction. */
export function nextSort(current: LibrarySortState, key: LibrarySortKey): LibrarySortState {
  return current.key === key ? { key, desc: !current.desc } : { key, desc: key !== "name" };
}

type Sortable = { name: string; updatedAt: number; sizeBytes?: number | null };

/** Suggested's Last activity: the later of modified and opened. */
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
