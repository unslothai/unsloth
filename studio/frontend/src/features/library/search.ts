// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LIBRARY_URL_SORTS, type LibraryUrlSort } from "./settings-store";

// Kept apart from the page so the route can validate its URL without loading the page chunk.
export const LIBRARY_TABS = [
  "suggested",
  "favorites",
  "folders",
  "images",
  "videos",
  "audio",
  "models",
  "all",
] as const;
export type LibraryTab = (typeof LIBRARY_TABS)[number];

export interface LibrarySearch {
  show?: LibraryTab;
  folder?: string;
  item?: string;
  sort?: LibraryUrlSort;
  filter?: "files";
}

export function validateLibrarySearch(search: Record<string, unknown>): LibrarySearch {
  const show = LIBRARY_TABS.find((entry) => entry === search.show);
  return {
    ...(show ? { show } : {}),
    ...(typeof search.folder === "string" ? { folder: search.folder } : {}),
    ...(typeof search.item === "string" ? { item: search.item } : {}),
    ...(LIBRARY_URL_SORTS.includes(search.sort as LibraryUrlSort)
      ? { sort: search.sort as LibraryUrlSort }
      : {}),
    ...(search.filter === "files" ? { filter: "files" as const } : {}),
  };
}
