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

// `show`, not `tab`: search params share one type across routes, and the hub owns `tab`.
export interface LibrarySearch {
  show?: LibraryTab;
  /** Open this folder instead of a tab. */
  folder?: string;
  /** Preview this item on top of whatever is showing. */
  item?: string;
  /** Sorted this way instead of the Sort setting: a column click, or Settings > Library > Storage. */
  sort?: LibraryUrlSort;
  /** Start filtered to files that are not media or models, for the Storage Files row. */
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
