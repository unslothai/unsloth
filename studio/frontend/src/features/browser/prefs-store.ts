// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { SearchEngineId } from "./address";

/** When the bookmarks toolbar shows, as Firefox offers it. */
export type BookmarksToolbarMode = "always" | "newtab" | "never";

interface BrowserPrefsState {
  openLinksInBrowser: boolean;
  openFilesInBrowser: boolean;
  searchEngine: SearchEngineId;
  showFullUrl: boolean;
  bookmarksToolbar: BookmarksToolbarMode;
  /** Whether saving a bookmark opens its name and location editor. */
  showBookmarkEditor: boolean;
  /** Sites taken off the new tab's Suggested, by host. */
  hiddenSuggestions: string[];
  setOpenLinksInBrowser: (value: boolean) => void;
  setOpenFilesInBrowser: (value: boolean) => void;
  setSearchEngine: (value: SearchEngineId) => void;
  setShowFullUrl: (value: boolean) => void;
  setBookmarksToolbar: (value: BookmarksToolbarMode) => void;
  setShowBookmarkEditor: (value: boolean) => void;
  hideSuggestion: (host: string) => void;
  restoreSuggestions: () => void;
}

export const useBrowserPrefsStore = create<BrowserPrefsState>()(
  persist(
    (set) => ({
      openLinksInBrowser: false,
      openFilesInBrowser: true,
      searchEngine: "duckduckgo",
      showFullUrl: false,
      bookmarksToolbar: "newtab",
      showBookmarkEditor: true,
      hiddenSuggestions: [],
      setOpenLinksInBrowser: (openLinksInBrowser) => set({ openLinksInBrowser }),
      setOpenFilesInBrowser: (openFilesInBrowser) => set({ openFilesInBrowser }),
      setSearchEngine: (searchEngine) => set({ searchEngine }),
      setShowFullUrl: (showFullUrl) => set({ showFullUrl }),
      setBookmarksToolbar: (bookmarksToolbar) => set({ bookmarksToolbar }),
      setShowBookmarkEditor: (showBookmarkEditor) => set({ showBookmarkEditor }),
      hideSuggestion: (host) =>
        set((state) =>
          state.hiddenSuggestions.includes(host)
            ? state
            : { hiddenSuggestions: [...state.hiddenSuggestions, host] },
        ),
      restoreSuggestions: () => set({ hiddenSuggestions: [] }),
    }),
    {
      name: "unsloth_browser_prefs",
      // v0 defaulted links to the panel; reset to the default browser.
      version: 1,
      migrate: (persisted, version) => {
        const state = (persisted ?? {}) as Partial<BrowserPrefsState>;
        return version < 1 ? { ...state, openLinksInBrowser: false } : state;
      },
    },
  ),
);
