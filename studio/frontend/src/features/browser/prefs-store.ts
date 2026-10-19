// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { SearchEngineId } from "./address";

/** When the bookmarks toolbar shows, as Firefox offers it. */
export type BookmarksToolbarMode = "always" | "newtab" | "never";

/** Whether sending annotations also attaches a screenshot of the page. */
export type AnnotationScreenshots = "always" | "never";

/** Page zoom a new tab starts at, and that resetting zoom returns to. */
export const DEFAULT_ZOOM_STEPS = [0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2];

/** How long visits are kept, in days; 0 keeps them until cleared. */
export const HISTORY_RETENTION_DAYS = [0, 90, 30, 7, 1];

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
  /** Show ⌘/Ctrl-clicked links' tabs at once instead of in the background. */
  switchToNewTabs: boolean;
  defaultZoom: number;
  showSuggestedSites: boolean;
  showRecentPages: boolean;
  saveHistory: boolean;
  /** Days visits are kept (HISTORY_RETENTION_DAYS); 0 for until cleared. */
  historyRetentionDays: number;
  saveDownloadHistory: boolean;
  askWhereToSave: boolean;
  askBeforeDownloading: boolean;
  annotationScreenshots: AnnotationScreenshots;
  setOpenLinksInBrowser: (value: boolean) => void;
  setOpenFilesInBrowser: (value: boolean) => void;
  setSearchEngine: (value: SearchEngineId) => void;
  setShowFullUrl: (value: boolean) => void;
  setBookmarksToolbar: (value: BookmarksToolbarMode) => void;
  setShowBookmarkEditor: (value: boolean) => void;
  hideSuggestion: (host: string) => void;
  restoreSuggestions: () => void;
  setSwitchToNewTabs: (value: boolean) => void;
  setDefaultZoom: (value: number) => void;
  setShowSuggestedSites: (value: boolean) => void;
  setShowRecentPages: (value: boolean) => void;
  setSaveHistory: (value: boolean) => void;
  setHistoryRetentionDays: (value: number) => void;
  setSaveDownloadHistory: (value: boolean) => void;
  setAskWhereToSave: (value: boolean) => void;
  setAskBeforeDownloading: (value: boolean) => void;
  setAnnotationScreenshots: (value: AnnotationScreenshots) => void;
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
      switchToNewTabs: false,
      defaultZoom: 1,
      showSuggestedSites: true,
      showRecentPages: true,
      saveHistory: true,
      historyRetentionDays: 0,
      saveDownloadHistory: true,
      askWhereToSave: false,
      askBeforeDownloading: true,
      annotationScreenshots: "never",
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
      setSwitchToNewTabs: (switchToNewTabs) => set({ switchToNewTabs }),
      setDefaultZoom: (defaultZoom) => set({ defaultZoom }),
      setShowSuggestedSites: (showSuggestedSites) => set({ showSuggestedSites }),
      setShowRecentPages: (showRecentPages) => set({ showRecentPages }),
      setSaveHistory: (saveHistory) => set({ saveHistory }),
      setHistoryRetentionDays: (historyRetentionDays) => set({ historyRetentionDays }),
      setSaveDownloadHistory: (saveDownloadHistory) => set({ saveDownloadHistory }),
      setAskWhereToSave: (askWhereToSave) => set({ askWhereToSave }),
      setAskBeforeDownloading: (askBeforeDownloading) => set({ askBeforeDownloading }),
      setAnnotationScreenshots: (annotationScreenshots) => set({ annotationScreenshots }),
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

/** The zoom a page is at unless the reader zooms it. */
export function defaultZoom(): number {
  return useBrowserPrefsStore.getState().defaultZoom;
}
