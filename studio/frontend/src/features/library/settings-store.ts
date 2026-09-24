// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

export const LIBRARY_SETTINGS_STORAGE_KEY = "unsloth_library_settings";

export type LibraryCardSize = "small" | "medium" | "large";
export type LibraryImageLayout = "masonry" | "square";
export type LibrarySort = "recent" | "oldest" | "name" | "size";
export type LibraryStartTab = "suggested" | "favorites" | "folders" | "all";
export type LibraryMediaTabs = "auto" | "always";

export interface LibrarySettings {
  cardSize: LibraryCardSize;
  imageLayout: LibraryImageLayout;
  showCardDates: boolean;
  sort: LibrarySort;
  startTab: LibraryStartTab;
  mediaTabs: LibraryMediaTabs;
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
  mediaTabs: "auto",
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

interface LibrarySettingsState extends LibrarySettings {
  set: (patch: Partial<LibrarySettings>) => void;
  reset: () => void;
}

export const useLibrarySettingsStore = create<LibrarySettingsState>()(
  persist(
    (set) => ({
      ...DEFAULT_LIBRARY_SETTINGS,
      set: (patch) => set(patch),
      reset: () => set(DEFAULT_LIBRARY_SETTINGS),
    }),
    { name: LIBRARY_SETTINGS_STORAGE_KEY, version: 1 },
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

type Sortable = { name: string; updatedAt: number; sizeBytes?: number | null };

export function compareBySort(sort: LibrarySort): (a: Sortable, b: Sortable) => number {
  switch (sort) {
    case "oldest":
      return (a, b) => a.updatedAt - b.updatedAt;
    case "name":
      return (a, b) => a.name.localeCompare(b.name, undefined, { numeric: true });
    case "size":
      return (a, b) => (b.sizeBytes ?? 0) - (a.sizeBytes ?? 0);
    default:
      return (a, b) => b.updatedAt - a.updatedAt;
  }
}
