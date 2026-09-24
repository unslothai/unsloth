// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
}

export function validateLibrarySearch(search: Record<string, unknown>): LibrarySearch {
  const show = LIBRARY_TABS.find((entry) => entry === search.show);
  return {
    ...(show ? { show } : {}),
    ...(typeof search.folder === "string" ? { folder: search.folder } : {}),
    ...(typeof search.item === "string" ? { item: search.item } : {}),
  };
}
